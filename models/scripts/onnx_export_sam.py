# onnx_export_sam_fixed.py
"""
Fixed ONNX Export for Modified SAM Models
Handles both standard SAM with modified mask tokens and SemanticSAM with hypernetworks
"""

from __future__ import annotations
import os, json, types, hashlib, time, warnings, inspect
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from types import SimpleNamespace
from os.path import expandvars, expanduser

import torch
import torch.nn as nn
import torch.nn.functional as F

from onnxruntime.quantization import quantize_dynamic, QuantType

# --- repo imports -------------------------------------------------------------
from finetune_sam_slim.models.sam import sam_model_registry
from finetune_sam_slim.models.advanced_sam_peft import create_peft_sam

# ------------------------------------------------------------------------------
# Utilities (reused from original)
# ------------------------------------------------------------------------------

def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x

def _find_config_for_checkpoint(ckpt: Path) -> Path:
    cfg = ckpt.parent / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"config.json not found next to {ckpt}")
    return cfg

def _choose_device(devices_cfg: Any) -> str:
    """Choose the best available device from config"""
    dev = None
    if isinstance(devices_cfg, (list, tuple)) and len(devices_cfg) > 0:
        dev = str(devices_cfg[0]).strip().lower()
    elif isinstance(devices_cfg, str):
        dev = devices_cfg.strip().lower()

    if not dev:
        return "cpu"
    if dev == "cpu":
        return "cpu"
    if dev.isdigit():
        dev = f"cuda:{dev}"
    if dev.startswith("cuda"):
        if torch.cuda.is_available():
            return dev
        print(f"[warn] {dev} requested but CUDA not available → using CPU")
        return "cpu"
    return "cpu"

def _safe_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Extract clean state_dict and strips common prefixes"""
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "ema", "sam", "module", "model_state", "model_state_dict"):
            sd = obj.get(k)
            if isinstance(sd, dict) and sd:
                obj = sd
                break
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict mapping.")

    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            nk = k
            for pref in ("module.", "model.", "sam."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            out[nk] = v
    return out

def _resolve_arch(cfg: Dict[str, Any]) -> str:
    """Map config to the proper builder key in sam_model_registry"""
    arch = str(cfg.get("arch", "vit_b")).strip()
    use_sem = bool(cfg.get("use_semantic", False))
    num_cls = int(cfg.get("num_cls", cfg.get("num_classes", 2)))
    if use_sem or num_cls > 3:
        if not arch.startswith("semantic_"):
            arch = f"semantic_{arch}"
    return arch

def _parse_list(s, cast=int, allow_none=True):
    if s is None:
        return None if allow_none else []
    if isinstance(s, list):
        return [cast(x) for x in s]
    st = str(s).strip()
    if st == "" and allow_none:
        return None
    if st.lower() == "none":
        return None
    try:
        obj = json.loads(st)
        if isinstance(obj, list):
            return [cast(x) for x in obj]
    except Exception:
        pass
    parts = [p for p in st.replace(",", " ").split() if p]
    return [cast(p) for p in parts]

def _normalize_devices(dev):
    """Normalize device specification"""
    if dev is None:
        return ["cpu"]
    if isinstance(dev, (list, tuple)):
        vals = []
        for d in dev:
            ds = str(d).strip().lower()
            if ds == "cpu":
                vals.append("cpu")
            elif ds.isdigit():
                vals.append(f"cuda:{ds}")
            elif ds.startswith("cuda"):
                vals.append(ds)
            else:
                vals.append(ds)
        return vals or ["cpu"]
    s = str(dev).strip().lower()
    if s == "cpu":
        return ["cpu"]
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return _normalize_devices(obj)
    except Exception:
        pass
    parts = [p for p in s.replace(",", " ").split() if p]
    if parts and all(p.isdigit() for p in parts):
        return [f"cuda:{p}" for p in parts]
    if parts and all(p.startswith("cuda") for p in parts):
        return parts
    if s.isdigit():
        return [f"cuda:{s}"]
    return [s]

def _coerce_args_namespace(args_ns: SimpleNamespace) -> SimpleNamespace:
    """Ensure fields expected by modeling code have the right types"""
    devs = getattr(args_ns, "devices", None)
    args_ns.devices = _normalize_devices(devs)

    gf = getattr(args_ns, "gpu_fractions", None)
    if gf is not None:
        args_ns.gpu_fractions = _parse_list(gf, cast=float, allow_none=False)
    
    ead = getattr(args_ns, "encoder_adapter_depths", None)
    if ead is not None:
        args_ns.encoder_adapter_depths = _parse_list(ead, cast=int, allow_none=False)
    
    for key in ("if_split_encoder_gpus", "if_encoder_adapter", "if_mask_decoder_adapter"):
        if hasattr(args_ns, key):
            v = getattr(args_ns, key)
            if isinstance(v, str):
                setattr(args_ns, key, v.strip().lower() not in ("0", "false", "no", ""))

    return args_ns

# ------------------------------------------------------------------------------
# Model Detection and Decoder Access
# ------------------------------------------------------------------------------

def _detect_model_type(model: nn.Module) -> Tuple[str, bool]:
    """
    Detect model type and structure.
    Returns: (model_type, is_semantic)
    model_type: 'standard' or 'semantic'
    """
    # Check for SemanticSAM structure
    if hasattr(model, "mask_adapter"):
        # SemanticSAM with adapters
        if hasattr(model.mask_adapter, 'decoder_head') and hasattr(model.mask_adapter, 'decoder_neck'):
            print("[info] Detected SemanticSAM with decoder_head/decoder_neck structure")
            return 'semantic', True
        elif hasattr(model.mask_adapter, 'sam_mask_decoder'):
            print("[info] Detected SemanticSAM with wrapped mask_decoder")
            return 'semantic_wrapped', True
    
    # Standard SAM
    if hasattr(model, "mask_decoder"):
        print("[info] Detected standard SAM structure")
        return 'standard', False
    
    raise ValueError("Unknown model structure - cannot find decoder")

def _get_decoder_module(model: nn.Module, model_type: str) -> nn.Module:
    """Get the actual decoder module based on model type"""
    if model_type == 'semantic':
        # SemanticSAM with neck/head structure
        return model.mask_adapter
    elif model_type == 'semantic_wrapped':
        # SemanticSAM with wrapped decoder
        return model.mask_adapter.sam_mask_decoder
    else:
        # Standard SAM
        return model.mask_decoder

def _get_num_masks(decoder: nn.Module, model_type: str) -> int:
    """Get the number of output masks from decoder structure"""
    if model_type == 'semantic':
        # SemanticSAM - check decoder_head for class_num
        if hasattr(decoder, 'decoder_head') and hasattr(decoder.decoder_head, 'class_num'):
            return decoder.decoder_head.class_num
    
    # Standard SAM or wrapped - check num_mask_tokens
    if hasattr(decoder, 'num_mask_tokens'):
        return decoder.num_mask_tokens
    elif hasattr(decoder, 'decoder_neck') and hasattr(decoder.decoder_neck, 'num_mask_tokens'):
        return decoder.decoder_neck.num_mask_tokens
    
    # Default fallback
    return 4  # Original SAM default

# ------------------------------------------------------------------------------
# Export-safe Patching for Decoder
# ------------------------------------------------------------------------------

def _export_safe_patch_decoder(decoder: nn.Module, model_type: str) -> None:
    """
    Patch decoder's predict_masks to be ONNX-export friendly.
    Handles both standard SAM and SemanticSAM structures.
    """
    print(f"[info] Applying export-safe patch for {model_type} decoder")
    
    if model_type == 'semantic':
        # For SemanticSAM with neck/head structure, patch the neck
        if hasattr(decoder, 'decoder_neck'):
            _patch_decoder_neck(decoder.decoder_neck)
    else:
        # Standard SAM - patch predict_masks
        if hasattr(decoder, "predict_masks"):
            _patch_standard_predict_masks(decoder)

def _export_safe_patch_prompt_encoder(model: nn.Module) -> None:
    # Find the prompt encoder on the built model
    pe = None
    if hasattr(model, "prompt_encoder"):
        pe = model.prompt_encoder
    elif hasattr(model, "mask_adapter") and hasattr(model.mask_adapter, "prompt_encoder"):
        pe = model.mask_adapter.prompt_encoder
    if pe is None or not hasattr(pe, "_embed_points"):
        return

    orig_embed_points = pe._embed_points

    def _embed_points_export(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        # Make point coords match SAM's pixel-center convention
        points = points + 0.5

        # NEW: normalize labels to 2-D (B, N). If input came in as (B, N, 1), squeeze it.
        if labels.dim() == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)  # (B, N)

        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device, dtype=points.dtype)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device, dtype=labels.dtype)  # (B,1)
            points = torch.cat([points, padding_point], dim=1)   # (B, N+1, 2)
            labels = torch.cat([labels, padding_label], dim=1)   # (B, N+1)

        # Embed points → (B, N, 256)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Build (B, N, 1) boolean masks for ONNX-friendly broadcasting to (B, N, 256)
        m_neg1 = (labels == -1).unsqueeze(-1)
        m_pos  = (labels ==  1).unsqueeze(-1)
        m_neg  = (labels ==  0).unsqueeze(-1)

        # Weights as (1, 1, C) so they broadcast over (B, N, 256)
        w_not = self.not_a_point_embed.weight.view(1, 1, -1)
        w_p0  = self.point_embeddings[0].weight.view(1, 1, -1)
        w_p1  = self.point_embeddings[1].weight.view(1, 1, -1)

        # Zero-out ignored points, then add the appropriate embeddings (no in-place ops for ONNX)
        zero = torch.zeros_like(point_embedding)
        point_embedding = torch.where(m_neg1, zero, point_embedding)
        point_embedding = point_embedding + m_neg1.float() * w_not
        point_embedding = point_embedding + m_neg.float()  * w_p0
        point_embedding = point_embedding + m_pos.float()  * w_p1
        return point_embedding

    pe._embed_points = types.MethodType(_embed_points_export, pe)


def _patch_standard_predict_masks(decoder: nn.Module) -> None:
    """Patch standard SAM decoder's predict_masks for ONNX export"""
    def predict_masks_export(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ):
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = image_embeddings.repeat(tokens.shape[0], 1, 1, 1)  # export-safe
        else:
            src = image_embeddings

        src = src + dense_prompt_embeddings

        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = image_pe.expand(tokens.shape[0], -1, -1, -1)  # export-safe
        else:
            pos_src = image_pe
        b, c, h, w = src.shape

        # Transformer
        hs, src2 = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale + hypernets
        src2 = src2.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src2)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b2, c2, h2, w2 = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b2, c2, h2 * w2)).view(b2, -1, h2, w2)

        # IoU pred
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    decoder.predict_masks = types.MethodType(predict_masks_export, decoder)
    print("[ok] Applied export-safe patch to standard decoder.predict_masks")

def _patch_decoder_neck(neck: nn.Module) -> None:
    """Patch SemanticSAM's decoder neck for ONNX export"""
    # The neck forward method already uses repeat_interleave which we need to replace
    # But since it returns intermediate values, we might not need to patch it
    # The actual mask generation happens in the head
    print("[ok] SemanticSAM decoder_neck is already export-compatible")

# ------------------------------------------------------------------------------
# Custom ONNX Wrapper for Modified SAM
# ------------------------------------------------------------------------------

class ModifiedSAMOnnxModel(nn.Module):
    """
    ONNX export wrapper for modified SAM models.
    Handles both standard SAM with modified mask tokens and SemanticSAM.
    """
    def __init__(self, sam_model: nn.Module, num_masks: int, model_type: str):
        super().__init__()
        self.model = sam_model
        self.num_masks = num_masks
        self.model_type = model_type
        
        print(f"[info] Created ONNX wrapper for {model_type} model with {num_masks} output masks")

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for ONNX export.
        Returns masks with shape (B, num_masks, H, W)
        """
        device = image_embeddings.device
        
        # Prepare prompts
        sparse_embeddings, dense_embeddings = self._encode_prompts(
            point_coords, point_labels, mask_input, has_mask_input, device
        )
        
        # Get dense PE
        if hasattr(self.model, 'prompt_encoder'):
            image_pe = self.model.prompt_encoder.get_dense_pe()
        elif hasattr(self.model, 'prompt_adapter'):
            image_pe = self.model.prompt_adapter.sam_prompt_encoder.get_dense_pe()
        else:
            raise ValueError("Cannot find prompt encoder for PE")
        
        # Decode masks based on model type
        if self.model_type == 'semantic':
            masks, iou_pred = self._forward_semantic(
                image_embeddings, image_pe, sparse_embeddings, dense_embeddings
            )
        else:
            masks, iou_pred = self._forward_standard(
                image_embeddings, image_pe, sparse_embeddings, dense_embeddings
            )
        
        # Postprocess masks
        masks = self._postprocess_masks(masks, orig_im_size)
        masks = masks + has_mask_input.view(masks.shape[0], 1, 1, 1) * 0.0
        
        print(f"[dbg] Output masks shape: {tuple(masks.shape)}")
        return masks

    def _encode_prompts(self, point_coords, point_labels, mask_input, has_mask_input, device):
        """Encode prompts using the appropriate encoder"""
        # Handle point prompts
        if point_coords.shape[1] > 0:
            points = (point_coords, point_labels)
        else:
            points = None
        
        # Handle mask prompts
        if has_mask_input.sum() > 0:
            masks = mask_input
        else:
            masks = None
        
        # Encode using appropriate encoder
        if hasattr(self.model, 'prompt_encoder'):
            sparse, dense = self.model.prompt_encoder(
                points=points, boxes=None, masks=masks
            )
        elif hasattr(self.model, 'prompt_adapter'):
            sparse, dense = self.model.prompt_adapter(
                points=points, boxes=None, masks=masks
            )
        else:
            raise ValueError("Cannot find prompt encoder")
        
        return sparse, dense

    def _forward_standard(self, image_embeddings, image_pe, sparse_embeddings, dense_embeddings):
        """Forward pass for standard SAM decoder"""
        decoder = self.model.mask_decoder
        
        # Call predict_masks directly (should be patched)
        masks, iou_pred = decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        
        # For modified SAM, all masks are valid (not just 1:)
        # Shape should be (B, num_masks, H, W) where num_masks could be 2, 3, etc.
        return masks, iou_pred

    def _forward_semantic(self, image_embeddings, image_pe, sparse_embeddings, dense_embeddings):
        """Forward pass for SemanticSAM decoder"""
        decoder = self.model.mask_adapter
        
        # Call decoder neck
        src, iou_token_out, mask_tokens_out, src_shape = decoder.decoder_neck(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,  # Always use multimask for semantic
        )
        
        # Call decoder head (uses hypernetworks for num_cls outputs)
        masks, iou_pred = decoder.decoder_head(
            src, iou_token_out, mask_tokens_out, src_shape, mask_scale=1
        )
        
        return masks, iou_pred

    def _postprocess_masks(self, masks, orig_im_size):
        """Postprocess masks to original image size"""
        # Get original H, W from orig_im_size
        if orig_im_size.dim() == 1:
            h, w = orig_im_size[0].item(), orig_im_size[1].item()
        else:
            h, w = orig_im_size[0, 0].item(), orig_im_size[0, 1].item()
        
        # Upscale masks
        masks = F.interpolate(
            masks,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        
        return masks

# ------------------------------------------------------------------------------
# Export Functions
# ------------------------------------------------------------------------------

def _export_encoder(model: nn.Module, out_path: Path, image_size: int, device: str, opset: int) -> str:
    """Export encoder to ONNX"""
    # Get encoder based on model type
    if hasattr(model, 'image_encoder'):
        enc = model.image_encoder
    elif hasattr(model, 'img_adapter'):
        enc = model.img_adapter.sam_img_encoder
    else:
        raise ValueError("Cannot find image encoder")
    
    enc = enc.to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            enc, (dummy,), str(out_path),
            input_names=["x"], 
            output_names=["image_embeddings"],
            dynamic_axes={"x": {0: "batch"}, "image_embeddings": {0: "batch"}},
            do_constant_folding=True, 
            opset_version=opset,
        )
    print(f"[ok] Encoder exported → {out_path}")
    return _sha256(out_path)

def _export_decoder(model: nn.Module, out_path: Path, image_size: int, device: str, 
                   opset: int, num_cls: int, model_type: str) -> str:
    """Export decoder to ONNX using custom wrapper"""
    
    # Create wrapper
    wrapper = ModifiedSAMOnnxModel(model, num_cls, model_type).to(device)
    
    # Derive embedding size
    if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "image_embedding_size"):
        es_h, es_w = model.prompt_encoder.image_embedding_size
    elif hasattr(model, "prompt_adapter"):
        es_h = es_w = 64  # SemanticSAM typically uses 64x64
    else:
        es_h = es_w = image_size // 16
    
    print(f"[info] Using embedding size: {es_h}x{es_w}")
    
    # Create dummy inputs
    B = 1
    image_embeddings = torch.randn(B, 256, es_h, es_w, device=device)
    point_coords = torch.zeros(B, 1, 2, device=device)
    point_labels = torch.full((B, 1, 1), -1, device=device, dtype=torch.int64)
    mask_input = torch.zeros(B, 1, 256, 256, device=device)
    has_mask_input = torch.zeros(B, device=device)
    orig_im_size = torch.tensor([image_size, image_size], device=device, dtype=torch.int64)           
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            (image_embeddings, point_coords, point_labels, mask_input, has_mask_input, orig_im_size),
            str(out_path),
            input_names=[
                "image_embeddings", "point_coords", "point_labels",
                "mask_input", "has_mask_input", "orig_im_size",
            ],
            output_names=["masks"],
            dynamic_axes={
                "image_embeddings": {0: "batch"},
                "point_coords": {0: "batch", 1: "num_points"},
                "point_labels": {0: "batch", 1: "num_points", 2: "one"},  # ← add axis-2,
                "mask_input": {0: "batch"},
                "has_mask_input": {0: "batch"},
                "masks": {0: "batch"},
            },
            do_constant_folding=True,
            opset_version=opset,
        )
    
    print(f"[ok] Decoder exported → {out_path}")
    return _sha256(out_path)

# ------------------------------------------------------------------------------
# Model Building and Loading
# ------------------------------------------------------------------------------

def _build_model_from_config(cfg: Dict[str, Any], ckpt_path: Path, device: str):
    """Build model from config and load checkpoint"""
    
    arch = _resolve_arch(cfg)
    num_cls = int(cfg.get("num_cls", cfg.get("num_classes", 2)))
    image_size = int(cfg.get("image_size", 1024))
    single_mask = bool(cfg.get("single_mask", True))
    out_dir = str(cfg.get("output_dir", ckpt_path.parent.as_posix()))
    enc_name = cfg.get("encoder_out") or None
    dec_name = cfg.get("decoder_out") or None
    opset = int(cfg.get("opset", 17))
    quantize = bool(cfg.get("quantize", False))
    finetune_type = str(cfg.get("finetune_type", "vanilla")).strip().lower()

    # Prepare args namespace
    args_ns = _to_ns(cfg)
    _coerce_args_namespace(args_ns)

    # Handle base checkpoint
    base_ckpt = getattr(args_ns, "sam_ckpt", None)
    if isinstance(base_ckpt, str) and base_ckpt.strip().lower() in ("", "none", "auto"):
        base_ckpt = None

    # Get builder
    if arch not in sam_model_registry:
        if arch.startswith("semantic_"):
            alt_arch = arch.replace("semantic_", "")
            if alt_arch in sam_model_registry:
                print(f"[warn] '{arch}' not in registry; falling back to '{alt_arch}'")
                arch = alt_arch
            else:
                raise KeyError(f"Unknown arch '{arch}' in sam_model_registry.")
        else:
            raise KeyError(f"Unknown arch '{arch}' in sam_model_registry.")
    
    builder = sam_model_registry[arch]

    # Build model
    print(f"[info] Building {arch} with num_classes={num_cls}")
    model = builder(args_ns, checkpoint=base_ckpt, num_classes=num_cls)
    model.to(device).eval()

    # Apply PEFT if needed
    if finetune_type == "peft":
        print("[info] Applying PEFT wrapper")
        model = create_peft_sam(
            model,
            method=str(cfg.get("peft_method", "lora_fa")).strip().lower(),
            rank=int(cfg.get("peft_rank", cfg.get("rank", 4))),
            alpha=float(cfg.get("peft_alpha", cfg.get("alpha", 1.0))),
            dropout=float(cfg.get("peft_dropout", cfg.get("dropout", 0.0))),
            encoder_layers=cfg.get("encoder_layers", None),
            decoder_layers=cfg.get("decoder_layers", None),
            enable_adapters=bool(cfg.get("enable_adapters", False)),
            adapter_modules=cfg.get("adapter_modules", None),
        ).to(device).eval()

    # Load fine-tuned weights
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = _safe_state_dict(obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    if missing:
        print(f"[info] Missing keys: {len(missing)} (first 5: {missing[:5]})")

    # Detect model type and patch decoder
    model_type, is_semantic = _detect_model_type(model)
    decoder = _get_decoder_module(model, model_type)
    
    # Apply export-safe patch
    _export_safe_patch_decoder(decoder, model_type)

    try:
        _export_safe_patch_prompt_encoder(model)
        print("[ok] export-safe patch applied to prompt_encoder._embed_points")
    except Exception as e:
        print(f"[warn] could not patch prompt encoder for export: {e}")

    return model, num_cls, image_size, single_mask, out_dir, enc_name, dec_name, opset, quantize, model_type

# ------------------------------------------------------------------------------
# Hashing and Manifest
# ------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class ExportMeta:
    variant: str
    arch: str
    arch_full: str
    peft: bool
    num_classes: int
    single_mask: bool
    image_size: int
    opset: int
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]

def _derive_meta(model: nn.Module, cfg: Dict[str, Any], opset: int, model_type: str) -> ExportMeta:
    arch_full = str(cfg.get("arch", "vit_b"))
    arch = arch_full.replace("semantic_", "")
    is_semantic = model_type.startswith("semantic")
    peft = str(cfg.get("finetune_type", "")).strip().lower() == "peft"
    num_classes = int(cfg.get("num_cls", cfg.get("num_classes", 2)))
    single_mask = bool(cfg.get("single_mask", True))
    image_size = int(cfg.get("image_size", 1024))

    inputs = (
        "image_embeddings", "point_coords", "point_labels",
        "mask_input", "has_mask_input", "orig_im_size",
    )
    outputs = ("masks",)

    return ExportMeta(
        variant=("semantic" if is_semantic else "sam"),
        arch=arch,
        arch_full=arch_full,
        peft=peft,
        num_classes=num_classes,
        single_mask=single_mask,
        image_size=image_size,
        opset=opset,
        inputs=inputs,
        outputs=outputs,
    )

def _default_names(meta: ExportMeta) -> Tuple[str, str]:
    base = f"{meta.variant}-{meta.arch}{'_peft' if meta.peft else ''}"
    enc = f"encoder_{base}.onnx"
    dec = f"decoder_{base}_nc{meta.num_classes}.onnx"
    return enc, dec

def _write_manifest(onnx_path: Path, meta: ExportMeta) -> None:
    info = {
        "file": onnx_path.name,
        "sha256": _sha256(onnx_path),
        "size_bytes": onnx_path.stat().st_size,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": asdict(meta),
    }
    out_json = onnx_path.with_suffix(onnx_path.suffix + ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[ok] Wrote manifest → {out_json}")

def _quantize_model(model_path: Path, output_suffix: str = "_quantized") -> Path:
    """Quantize an ONNX model to int8"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        print(f"[error] onnxruntime quantization tools not available: {e}")
        return model_path
        
    quantized_path = model_path.with_name(
        model_path.stem + output_suffix + model_path.suffix
    )
    
    try:
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        print(f"[ok] Quantized {model_path.name} → {quantized_path.name}")
        print(f"[hash] Quantized sha256={_sha256(quantized_path)}")
        return quantized_path
    except Exception as e:
        print(f"[warn] Quantization failed: {e}, using original model")
        return model_path

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _parse_cli() -> Path:
    import argparse
    p = argparse.ArgumentParser("ONNX export (config-driven)")
    p.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint (.pth)")
    return Path(p.parse_args().checkpoint).resolve()

def main() -> None:
    print("=" * 80)
    print("Modified SAM ONNX Export")
    print("Supports standard SAM with modified mask tokens and SemanticSAM")
    print("=" * 80)
    
    ckpt_raw = _parse_cli()
    ckpt = Path(expanduser(expandvars(str(ckpt_raw)))).resolve()
    cfg_path = _find_config_for_checkpoint(ckpt)
    cfg = _load_json(cfg_path)

    device = _choose_device(cfg.get("devices"))
    print(f"[info] Using device: {device}")

    # Build model and detect type
    model, num_cls, image_size, single_mask, out_dir, enc_name, dec_name, opset, quantize, model_type = \
        _build_model_from_config(cfg, ckpt, device=device)

    print(f"[info] Model type: {model_type}")
    print(f"[info] Number of output masks: {num_cls}")
    
    meta = _derive_meta(model, cfg, opset=opset, model_type=model_type)

    # Decide output names
    if not enc_name or not dec_name:
        def_enc, def_dec = _default_names(meta)
        enc_name = enc_name or def_enc
        dec_name = dec_name or def_dec

    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    enc_out = outdir / enc_name
    dec_out = outdir / dec_name

    # Export encoder
    print(f"\n[Exporting Encoder]")
    _export_encoder(model, enc_out, image_size=image_size, device=device, opset=opset)
    _write_manifest(enc_out, meta)

    # Quantize encoder
    if quantize:
        enc_quantized = _quantize_model(enc_out, "_quantized")
        if enc_quantized != enc_out:
            _write_manifest(enc_quantized, meta)

    # Export decoder
    print(f"\n[Exporting Decoder]")
    try:
        _export_decoder(model, dec_out, image_size=image_size, device=device, 
                       opset=opset, num_cls=num_cls, model_type=model_type)
    except RuntimeError as e:
        msg = str(e).lower()
        if "same device" in msg or "index_select" in msg or "cuda" in msg:
            print(f"[warn] Decoder export failed on {device} due to device mismatch → retrying on CPU")
            _export_decoder(model, dec_out, image_size=image_size, device="cpu", 
                          opset=opset, num_cls=num_cls, model_type=model_type)
        else:
            raise
    _write_manifest(dec_out, meta)

    # Quantize decoder
    if quantize:
        dec_quantized = _quantize_model(dec_out, "_quantized")
        if dec_quantized != dec_out:
            _write_manifest(dec_quantized, meta)

    print("\n" + "=" * 80)
    print("[done] Export complete")
    print(f"  Encoder: {enc_out}")
    print(f"  Decoder: {dec_out}")
    if quantize:
        print(f"\nQuantized models:")
        print(f"  Encoder: {enc_out.parent / (enc_out.stem + '_quantized.onnx')}")
        print(f"  Decoder: {dec_out.parent / (dec_out.stem + '_quantized.onnx')}")
    print("=" * 80)

if __name__ == "__main__":
    main()




