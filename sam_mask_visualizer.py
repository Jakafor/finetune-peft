"""
SAM Mask Visualizer Module
Loads custom SAM models with checkpoint weights and visualizes predictions
Supports vanilla, adapter, and PEFT model variants
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from types import SimpleNamespace
import warnings
from PIL import Image
import cv2
from os.path import expandvars, expanduser

# Import SAM components (adjust paths as needed)
from finetune_sam_slim.models.sam import sam_model_registry
from finetune_sam_slim.models.advanced_sam_peft import create_peft_sam
from finetune_sam_slim.models.sam.modeling.extend_sam import SemanticSam

# ------------------------------------------------------------------------------
# Utility Functions (adapted from onnx_export_sam.py)
# ------------------------------------------------------------------------------

def _expand_path(p) -> Path:
    # Make $HOME work on Windows even if it's not set
    if "HOME" not in os.environ and "USERPROFILE" in os.environ:
        os.environ["HOME"] = os.environ["USERPROFILE"]
    return Path(expanduser(expandvars(str(p)))).resolve()

def _parse_cli():
    import argparse
    p = argparse.ArgumentParser("SAM Mask Visualizer")
    p.add_argument("--checkpoint", "--checkpoint-path", required=True,
                   help="Path to fine-tuned checkpoint (.pth)")
    p.add_argument("--image", "--image-path", required=True,
                   help="Path to input image")
    p.add_argument("--out", "--output-dir", default=None,
                   help="Directory to save visualization (optional)")
    a = p.parse_args()
    return (
        _expand_path(a.checkpoint),
        _expand_path(a.image),
        _expand_path(a.out) if a.out else None,
    )


def _load_json(p: Path) -> Dict[str, Any]:
    """Load JSON configuration file"""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_ns(x):
    """Convert dict to SimpleNamespace recursively"""
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x

def _parse_list(s, cast=int, allow_none=True):
    """Parse list from various formats (JSON, comma-separated, etc.)"""
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
    """Normalize device specification to list of strings"""
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
    # String cases
    s = str(dev).strip().lower()
    if s == "cpu":
        return ["cpu"]
    if s.isdigit():
        return [f"cuda:{s}"]
    return [s]

def _choose_device(devices_cfg: Any) -> str:
    """Choose the best available device from config"""
    devices = _normalize_devices(devices_cfg)
    dev = devices[0] if devices else "cpu"
    
    if dev == "cpu":
        return "cpu"
    if dev.startswith("cuda"):
        if torch.cuda.is_available():
            return dev
        print(f"[warn] {dev} requested but CUDA not available → using CPU")
        return "cpu"
    return "cpu"

def _safe_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Extract clean state_dict from checkpoint object.
    Handles various checkpoint formats and removes common prefixes.
    """
    # Handle wrapped state dicts (common in checkpoints)
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "ema", "sam", "module", 
                 "model_state", "model_state_dict"):
            sd = obj.get(k)
            if isinstance(sd, dict) and sd:
                obj = sd
                break
    
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain valid state_dict")
    
    # Clean up key prefixes
    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            nk = k
            # Remove common prefixes
            for pref in ("module.", "model."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            out[nk] = v
    return out

def _resolve_arch(cfg: Dict[str, Any]) -> str:
    """
    Determine the correct architecture key for sam_model_registry.
    Handles semantic SAM variants based on use_semantic flag or num_cls.
    """
    arch = str(cfg.get("arch", "vit_b")).strip()
    use_sem = bool(cfg.get("use_semantic", False))
    num_cls = int(cfg.get("num_cls", cfg.get("num_classes", 2)))
    
    # Use semantic variant if flag is set or if more than 3 classes
    if use_sem or num_cls > 3:
        if not arch.startswith("semantic_"):
            arch = f"semantic_{arch}"
    return arch

def _coerce_args_namespace(args_ns: SimpleNamespace) -> SimpleNamespace:
    """
    Ensure args namespace has correct types for model building.
    Handles devices, lists, and boolean conversions.
    """
    # Normalize devices
    devs = getattr(args_ns, "devices", None)
    args_ns.devices = _normalize_devices(devs)
    
    # Handle gpu_fractions
    gf = getattr(args_ns, "gpu_fractions", None)
    if gf is not None:
        args_ns.gpu_fractions = _parse_list(gf, cast=float, allow_none=False)
    
    # Handle encoder_adapter_depths
    ead = getattr(args_ns, "encoder_adapter_depths", None)
    if ead is not None:
        args_ns.encoder_adapter_depths = _parse_list(ead, cast=int, allow_none=False)
    
    # Handle optional lists for PEFT
    args_ns.encoder_layers = _parse_list(getattr(args_ns, "encoder_layers", None), 
                                         cast=int, allow_none=True)
    args_ns.decoder_layers = _parse_list(getattr(args_ns, "decoder_layers", None), 
                                         cast=int, allow_none=True)
    
    # Convert string booleans
    for key in ("if_split_encoder_gpus", "if_encoder_adapter", 
                "if_mask_decoder_adapter", "enable_adapters"):
        if hasattr(args_ns, key):
            v = getattr(args_ns, key)
            if isinstance(v, str):
                setattr(args_ns, key, v.strip().lower() not in ("0", "false", "no", ""))
    
    return args_ns

# ------------------------------------------------------------------------------
# Model Building Functions
# ------------------------------------------------------------------------------

def build_sam_from_config(cfg: Dict[str, Any], device: str) -> nn.Module:
    """
    Build SAM model based on configuration.
    Supports vanilla, adapter, and PEFT variants.
    
    Returns model with base weights loaded (before fine-tuning checkpoint).
    """
    # Determine architecture
    arch = _resolve_arch(cfg)
    num_cls = int(cfg.get("num_cls", cfg.get("num_classes", 2)))
    image_size = int(cfg.get("image_size", 1024))
    finetune_type = str(cfg.get("finetune_type", "vanilla")).strip().lower()
    
    # Create args namespace for builders
    args_ns = _to_ns(cfg)
    _coerce_args_namespace(args_ns)
    
    # Handle base checkpoint path
    base_ckpt = getattr(args_ns, "sam_ckpt", None)
    if isinstance(base_ckpt, str) and base_ckpt.strip().lower() in ("", "none", "auto"):
        base_ckpt = None  # Will trigger auto-download of pretrained weights
    
    print(f"\n[Build Model]")
    print(f"  Architecture: {arch}")
    print(f"  Finetune type: {finetune_type}")
    print(f"  Num classes: {num_cls}")
    print(f"  Device: {device}")
    
    # Check if architecture exists in registry
    if arch not in sam_model_registry:
        # Try fallback for semantic variants
        if arch.startswith("semantic_"):
            alt_arch = arch.replace("semantic_", "")
            if alt_arch in sam_model_registry:
                print(f"  [info] '{arch}' not in registry, using '{alt_arch}'")
                arch = alt_arch
            else:
                raise KeyError(f"Unknown architecture: {arch}")
        else:
            raise KeyError(f"Unknown architecture: {arch}")
    
    # Get builder function
    builder = sam_model_registry[arch]
    
    # Build base model with pretrained weights
    print(f"  Loading base pretrained weights...")
    model = builder(args_ns, checkpoint=base_ckpt, num_classes=num_cls)
    model = model.to(device).eval()
    
    # Apply PEFT if specified
    if finetune_type == "peft":
        print(f"  Applying PEFT wrapper...")
        # Extract PEFT parameters from config
        peft_config = {
            "method": str(cfg.get("peft_method", "lora_fa")).strip().lower(),
            "rank": int(cfg.get("peft_rank", cfg.get("rank", 4))),
            "alpha": float(cfg.get("peft_alpha", cfg.get("alpha", 1.0))),
            "dropout": float(cfg.get("peft_dropout", cfg.get("dropout", 0.0))),
            "encoder_layers": getattr(args_ns, "encoder_layers", None),
            "decoder_layers": getattr(args_ns, "decoder_layers", None),
            "enable_adapters": bool(cfg.get("enable_adapters", False)),
            "adapter_modules": cfg.get("adapter_modules", None),
        }
        
        model = create_peft_sam(model, **peft_config)
        model = model.to(device).eval()
        print(f"    Method: {peft_config['method']}")
        print(f"    Rank: {peft_config['rank']}")
        print(f"    Alpha: {peft_config['alpha']}")
    
    elif finetune_type == "adapter":
        print(f"  Model has adapter layers enabled via config")
        # Adapters are already included if if_encoder_adapter and if_mask_decoder_adapter are True
    
    else:  # vanilla
        print(f"  Using vanilla fine-tuning setup")
    
    return model

def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path, device: str) -> Dict[str, Any]:
    """
    Load fine-tuned weights from checkpoint and overlay on model.
    Returns detailed information about loaded parameters.
    """
    print(f"\n[Load Checkpoint]")
    print(f"  Path: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_sd = _safe_state_dict(checkpoint)
    
    print(f"  Checkpoint keys: {len(checkpoint_sd)}")
    
    # Check if model is PEFT wrapped
    is_peft = hasattr(model, 'sam') and hasattr(model, 'config')
    
    # Get the actual model state dict
    if is_peft:
        # For PEFT model, we need to handle the wrapper
        model_sd = model.state_dict()
        print(f"  Model parameters (PEFT): {len(model_sd)}")
    else:
        model_sd = model.state_dict()
        print(f"  Model parameters: {len(model_sd)}")

    # --- Canonicalize and de-duplicate namespace aliases ---
    def _canonicalize_and_dedup(sd: Dict[str, torch.Tensor], is_peft: bool) -> Dict[str, torch.Tensor]:
        """
        Map legacy aliases to today's canonical names and drop duplicates.
        Canonical:
        PEFT   -> 'sam.image_encoder.*', 'sam.mask_decoder.*', 'sam.prompt_encoder.*'
        Non-PEFT -> 'image_encoder.*', 'mask_decoder.*', 'prompt_encoder.*'
        """
        out: Dict[str, torch.Tensor] = {}

        def prefer_sam(existing_key: str, new_key: str, src_key: str) -> bool:
            # Prefer keys that already carry the 'sam.' prefix (PEFT) if both exist
            return src_key.startswith("sam.")

        for k, v in sd.items():
            nk = k

            # Fix accidental "sam.encoder.*" / "sam.decoder.*" (seen in older runs)
            if k.startswith("sam.encoder."):
                nk = "sam.image_encoder." + k[len("sam.encoder."):]
            elif k.startswith("sam.decoder."):
                nk = "sam.mask_decoder." + k[len("sam.decoder."):]

            # Map bare aliases to canonical names
            elif k.startswith("encoder."):
                tail = k[len("encoder."):]
                nk = ("sam.image_encoder." if is_peft else "image_encoder.") + tail
            elif k.startswith("decoder."):
                tail = k[len("decoder."):]
                nk = ("sam.mask_decoder." if is_peft else "mask_decoder.") + tail
            elif k.startswith("prompt.") or k.startswith("prompt_enc."):
                tail = k.split(".", 1)[1]
                nk = ("sam.prompt_encoder." if is_peft else "prompt_encoder.") + tail

            # For non-PEFT loads, drop 'sam.' if present
            if not is_peft and nk.startswith("sam."):
                nk = nk[4:]

            # Keep the canonical copy only once; if both alias and canonical exist,
            # let the 'sam.' version win for PEFT models.
            if nk not in out or prefer_sam(nk, nk, k):
                out[nk] = v

        return out

    
    # Track loading statistics
    loaded_params = []
    skipped_params = []
    shape_mismatches = []
    missing_in_ckpt = []
    
    # Handle different checkpoint structures
    # Check if checkpoint has PEFT structure (keys like 'sam.xxx')
    checkpoint_sd = _canonicalize_and_dedup(checkpoint_sd, is_peft=is_peft)

    # Ensure PEFT/non-PEFT prefix consistency just once
    has_sam_prefix = any(k.startswith("sam.") for k in checkpoint_sd)
    if is_peft and not has_sam_prefix:
        checkpoint_sd = { (f"sam.{k}" if not k.startswith("peft_layers") else k): v
                        for k, v in checkpoint_sd.items() }
    elif not is_peft and has_sam_prefix:
        checkpoint_sd = { (k[4:] if k.startswith("sam.") else k): v
                        for k, v in checkpoint_sd.items()
                        if not k.startswith("peft_layers") }
    
    # Now attempt to load each parameter
    for name, param in model_sd.items():
        if name in checkpoint_sd:
            ckpt_param = checkpoint_sd[name]
            if param.shape == ckpt_param.shape:
                # Successfully load parameter
                param.data.copy_(ckpt_param.data)
                loaded_params.append(name)
            else:
                # Shape mismatch
                shape_mismatches.append({
                    "name": name,
                    "model_shape": param.shape,
                    "ckpt_shape": ckpt_param.shape
                })
        else:
            # Parameter not in checkpoint (will keep base weights)
            missing_in_ckpt.append(name)
    
    # Find unexpected keys in checkpoint
    unexpected_keys = [k for k in checkpoint_sd.keys() if k not in model_sd]
    
    # Print detailed loading report
    print(f"\n[Loading Report]")
    print(f"  Successfully loaded: {len(loaded_params)} parameters")
    print(f"  Missing in checkpoint (using base): {len(missing_in_ckpt)} parameters")
    print(f"  Shape mismatches: {len(shape_mismatches)} parameters")
    print(f"  Unexpected in checkpoint: {len(unexpected_keys)} parameters")
    
    # Show some examples of each category
    if loaded_params:
        print(f"\n  Examples of loaded parameters:")
        for name in loaded_params[:5]:
            print(f"    ✓ {name}")
        if len(loaded_params) > 5:
            print(f"    ... and {len(loaded_params) - 5} more")
    
    if missing_in_ckpt:
        print(f"\n  Examples of parameters using base weights:")
        for name in missing_in_ckpt[:5]:
            print(f"    ○ {name}")
        if len(missing_in_ckpt) > 5:
            print(f"    ... and {len(missing_in_ckpt) - 5} more")
    
    if shape_mismatches:
        print(f"\n  Shape mismatches (not loaded):")
        for mismatch in shape_mismatches[:3]:
            print(f"    ✗ {mismatch['name']}")
            print(f"      Model: {mismatch['model_shape']}, Checkpoint: {mismatch['ckpt_shape']}")
    
    if unexpected_keys:
        print(f"\n  Unexpected keys in checkpoint:")
        for key in unexpected_keys[:5]:
            print(f"    ? {key}")
    
    # Move model to device
    model = model.to(device)
    
    return {
        "loaded": loaded_params,
        "missing": missing_in_ckpt,
        "mismatches": shape_mismatches,
        "unexpected": unexpected_keys
    }

# ------------------------------------------------------------------------------
# Image Processing and Inference
# ------------------------------------------------------------------------------

def preprocess_image(image_path: str, image_size: int = 1024) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load and preprocess image for SAM input.
    Returns preprocessed tensor and original size.
    """
    print(f"\n[Preprocess Image]")
    print(f"  Path: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)
    print(f"  Original size: {original_size}")
    
    # Convert to numpy
    image_np = np.array(image)
    
    # Resize to model input size
    image_resized = cv2.resize(image_np, (image_size, image_size))
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
    
    # Apply SAM normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)
    
    print(f"  Preprocessed shape: {image_tensor.shape}")
    return image_tensor, (image_np.shape[1], image_np.shape[0])  # Return (W, H)

def generate_random_prompt(image_size: int = 1024, num_points: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random positive point prompts.
    Returns point coordinates and labels.
    """
    # Generate random points in [0, image_size]
    points = torch.rand(1, num_points, 2) * image_size  # (1, N, 2)
    labels = torch.ones(1, num_points, dtype=torch.long)  # All positive labels (1)
    
    print(f"\n[Generated Prompt]")
    print(f"  Points: {points.squeeze().tolist()}")
    print(f"  Labels: {labels.squeeze().tolist()} (1=positive, 0=negative)")
    
    return points, labels

def run_sam_inference(model: nn.Module, image: torch.Tensor, points: torch.Tensor, 
                      labels: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run SAM model inference with given image and prompts.
    Returns masks and IoU predictions.
    """
    print(f"\n[Running Inference]")
    
    # Move inputs to device
    image = image.to(device)
    points = points.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        # Check if this is a PEFT model
        if hasattr(model, 'sam') and hasattr(model, 'config'):
            # PEFT wrapped model - get the actual SAM model
            print("  Detected PEFT wrapped model")
            actual_model = model.sam
        else:
            actual_model = model
        
        # Now check the actual model structure
        if hasattr(actual_model, 'img_adapter'):
            # SemanticSAM structure
            print("  Using SemanticSAM inference path")
            image_embeddings = actual_model.img_adapter(image)
            sparse_embeddings, dense_embeddings = actual_model.prompt_adapter(
                points=(points, labels), boxes=None, masks=None
            )
            masks, iou_pred = actual_model.mask_adapter(
                image_embeddings=image_embeddings,
                prompt_adapter=actual_model.prompt_adapter,
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                multimask_output=True
            )
        elif hasattr(actual_model, 'image_encoder'):
            # Standard SAM structure
            print("  Using standard SAM inference path")
            # Encode image
            image_embeddings = actual_model.image_encoder(image)
            print(f"  Image embeddings shape: {image_embeddings.shape}")
            
            # Encode prompts
            sparse_embeddings, dense_embeddings = actual_model.prompt_encoder(
                points=(points, labels),
                boxes=None,
                masks=None
            )
            
            # Get dense positional encoding
            pe = actual_model.prompt_encoder.get_dense_pe()
            
            # Decode masks
            masks, iou_pred = actual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
        else:
            # Unknown structure
            print(f"  Error: Unrecognized model structure")
            print(f"  Model type: {type(actual_model)}")
            print(f"  Model attributes: {dir(actual_model)[:10]}...")  # Show first 10 attributes
            raise NotImplementedError(f"Unknown model structure for {type(actual_model)}")
    
    print(f"  Output masks shape: {masks.shape}")
    print(f"  IoU predictions shape: {iou_pred.shape}")
    print(f"  IoU values: {iou_pred.squeeze().tolist()}")
    
    return masks, iou_pred

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

def visualize_masks(image_path: str, masks: torch.Tensor, iou_scores: torch.Tensor, 
                    points: torch.Tensor, save_path: Optional[str] = None):
    """
    Visualize masks overlaid on original image with IoU scores.
    """
    print(f"\n[Visualization]")
    
    # Load original image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    # Process masks (resize to original size)
    masks_np = masks.cpu().numpy()
    num_masks = masks_np.shape[1]
    
    # Create subplot grid
    fig, axes = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))
    if num_masks == 1:
        axes = [axes]
    
    # Show original image with prompt point
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image with Prompt")
    
    # Add prompt point
    point = points.cpu().numpy()[0, 0]  # First point
    # Scale point to original image size
    point_scaled = point * np.array([w, h]) / 1024
    axes[0].scatter(point_scaled[0], point_scaled[1], c='red', s=100, marker='*')
    axes[0].axis('off')
    
    # Show each mask
    for i in range(num_masks):
        mask = masks_np[0, i]  # (H, W)
        
        # Resize mask to original size
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply sigmoid to get probabilities
        mask_prob = 1 / (1 + np.exp(-mask_resized))
        
        # Threshold
        mask_binary = mask_prob > 0.5
        
        # Create overlay
        overlay = image_np.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = mask_binary * 255  # Green channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Show with IoU score
        ax = axes[i + 1]
        ax.imshow(overlay)
        iou_score = iou_scores.cpu().numpy()[0, i]
        ax.set_title(f"Mask {i+1}\nIoU: {iou_score:.3f}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")
    
    plt.show()

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

def visualize(checkpoint_path: str, image_path: str, output_dir: Optional[str] = None):
    """
    Main function to load model, run inference, and visualize results.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint
        image_path: Path to input image
        output_dir: Optional directory to save results
    """
    # Resolve paths
    checkpoint_path = _expand_path(checkpoint_path)
    image_path = _expand_path(image_path)
    output_dir = _expand_path(output_dir) if output_dir else None
    
    # Find config.json in same directory as checkpoint
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    print("=" * 80)
    print("SAM Mask Visualizer")
    print("=" * 80)
    
    # Load configuration
    cfg = _load_json(config_path)
    device = _choose_device(cfg.get("devices"))
    
    # Build model with base weights
    model = build_sam_from_config(cfg, device)
    
    # Load fine-tuned checkpoint weights
    load_info = load_checkpoint_weights(model, checkpoint_path, device)
    
    # Set to eval mode
    model.eval()
    
    # Preprocess image
    image_tensor, _ = preprocess_image(str(image_path), 
                                    image_size=cfg.get("image_size", 1024))
    
    # Generate random prompt
    points, labels = generate_random_prompt(image_size=cfg.get("image_size", 1024))
    
    # Run inference
    masks, iou_scores = run_sam_inference(model, image_tensor, points, labels, device)
    
    # Visualize results
    save_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"mask_visualization_{checkpoint_path.stem}.png"
    
    visualize_masks(str(image_path), masks, iou_scores, points, save_path)
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    
    return model, masks, iou_scores

def main():  # zero-arg entry for console_script
    ckpt, img, out = _parse_cli()
    return visualize(str(ckpt), str(img), str(out) if out else None)


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    main()

