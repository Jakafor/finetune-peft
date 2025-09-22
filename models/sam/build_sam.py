# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
from urllib.request import urlretrieve
import torch
import torch.nn.functional as F
import hashlib, os
from torch.hub import download_url_to_file
from models.sam.modeling.extend_sam import SemanticSam

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
    TinyViT,
)

CKPTS = {
    "vit_b": ("sam_vit_b_01ec64.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
              "01ec64"),  # optional short tag – or full sha256 if you want
    "vit_l": ("sam_vit_l_0b3195.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
              "0b3195"),
    "vit_h": ("sam_vit_h_4b8939.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
              "4b8939"),
    "vit_t": ("mobile_sam.pt",
              "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt",
              None),
}

def _ensure_ckpt(path: Path, url: str, sha256: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    # hash_prefix can be a full sha256 string; it just checks that the file's sha256 starts with it
    download_url_to_file(url, str(tmp), hash_prefix=(sha256 or ""), progress=True)
    os.replace(tmp, path)
    return path

def _safe_load(path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)
    
def _as_state_dict(obj):
    """
    Return a pure {str: Tensor} state_dict whether the checkpoint is raw or wrapped.
    """
    import torch as _torch
    if isinstance(obj, dict) and all(isinstance(v, _torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state", "model_state_dict", "net", "module", "sam"):
            sd = obj.get(k)
            if isinstance(sd, dict) and all(isinstance(v, _torch.Tensor) for v in sd.values()):
                return sd
    raise ValueError("Could not find a state_dict in checkpoint")

    
#######################################################################################################################
''' mazurowski semantic sam builders (vit_b / vit_l / vit_h)'''
#######################################################################################################################


def build_sam_vit_h(args = None, checkpoint=None, num_classes = 1):
    if checkpoint is None:
        name, url, _ = CKPTS["vit_h"]
        checkpoint = Path(os.path.expanduser("~/.cache/segment_anything")) / name
        _ensure_ckpt(checkpoint, url)

    return _build_sam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes = num_classes,
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args, checkpoint=None, num_classes = 1):
    if checkpoint is None:
        name, url, _ = CKPTS["vit_l"]
        checkpoint = Path(os.path.expanduser("~/.cache/segment_anything")) / name
        _ensure_ckpt(checkpoint, url)

    return _build_sam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes = num_classes,
        checkpoint=checkpoint,
    )


def build_sam_vit_b(args, checkpoint=None, num_classes = 1):
    if checkpoint is None:
        name, url, _ = CKPTS["vit_b"]
        checkpoint = Path(os.path.expanduser("~/.cache/segment_anything")) / name
        _ensure_ckpt(checkpoint, url)

    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes = num_classes,
        checkpoint=checkpoint,
    )

def build_sam_vit_t(args, checkpoint=None,  num_classes = 1):

    if checkpoint is None:
        name, url, _ = CKPTS["vit_t"]
        checkpoint = Path(os.path.expanduser("~/.cache/segment_anything")) / name
        _ensure_ckpt(checkpoint, url)
        
    prompt_embed_dim = 256
    #image_size = 1024
    vit_patch_size = 16
    image_embedding_size = 1024 // vit_patch_size
    mobile_sam = Sam(
            args, 
            image_encoder=TinyViT(args,img_size=args.image_size, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(args.image_size, args.image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=num_classes,
                    transformer=TwoWayTransformer(
                    args = args,
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    #print(mobile_sam)
    if checkpoint is not None:
       state_obj = _safe_load(checkpoint)
       state_dict = _as_state_dict(state_obj)
       try:
            mobile_sam.load_state_dict(state_dict, strict = False)
       except:
            new_state_dict = load_from_mobile(mobile_sam, state_dict)
            mobile_sam.load_state_dict(new_state_dict)
    return mobile_sam


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
}

#########################################################################################################################################   
'''semantic sam builders (vit_b / vit_l / vit_h / vit_t)'''
#########################################################################################################################################


# --- semantic builders (vit_b / vit_l / vit_h) ---
def build_semantic_sam_vit_b(args, checkpoint=None, num_classes=20):
    # We keep base decoder with 3 masks so checkpoint shapes match and mask_tokens load.
    # num_classes here is the semantic class count.
    if checkpoint is None:
        name, url, _ = CKPTS["vit_b"]
        checkpoint = Path(os.path.expanduser("~/.cache/segment_anything")) / name
        _ensure_ckpt(checkpoint, url)

    base = _build_sam_skeleton_vit(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_multimask_outputs=3,
    )

    state_obj = _safe_load(checkpoint)
    state_dict = _as_state_dict(state_obj)
    new_sd = load_from_semantic(base, state_dict, image_size=1024, vit_patch_size=16)
    base.load_state_dict(new_sd, strict=False)

    sem = SemanticSam(ori_sam=base, class_num=num_classes, model_type='vit_b')
    return sem


def build_semantic_sam_vit_l(args, checkpoint=None, num_classes=20):
    if checkpoint is None:
        name, url, _ = CKPTS["vit_l"]
        checkpoint = Path(os.path.expanduser("~/.cache/segment_anything")) / name
        _ensure_ckpt(checkpoint, url)

    base = _build_sam_skeleton_vit(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_multimask_outputs=3,
    )

    state_obj = _safe_load(checkpoint)
    state_dict = _as_state_dict(state_obj)
    new_sd = load_from_semantic(base, state_dict, image_size=1024, vit_patch_size=16)
    base.load_state_dict(new_sd, strict=False)

    sem = SemanticSam(ori_sam=base, class_num=num_classes, model_type='vit_l')
    return sem


def build_semantic_sam_vit_h(args, checkpoint=None, num_classes=20):
    if checkpoint is None:
        name, url, _ = CKPTS["vit_h"]
        checkpoint = Path(os.path.expanduser("~/.cache/segment_anything")) / name
        _ensure_ckpt(checkpoint, url)

    base = _build_sam_skeleton_vit(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_multimask_outputs=3,
    )

    state_obj = _safe_load(checkpoint)
    state_dict = _as_state_dict(state_obj)
    new_sd = load_from_semantic(base, state_dict, image_size=1024, vit_patch_size=16)
    base.load_state_dict(new_sd, strict=False)

    sem = SemanticSam(ori_sam=base, class_num=num_classes, model_type='vit_h')
    return sem


# Optional: Mobile-SAM (TinyViT). We can just wrap the existing builder.
def build_semantic_sam_vit_t(args, checkpoint=None, num_classes=20):
    mobile = build_sam_vit_t(args, checkpoint=checkpoint, num_classes=3)  # keep 3 mask tokens
    sem = SemanticSam(ori_sam=mobile, class_num=num_classes, model_type='vit_t')
    return sem


# Register new builders
sam_model_registry.update({
    "semantic_vit_b": build_semantic_sam_vit_b,
    "semantic_vit_l": build_semantic_sam_vit_l,
    "semantic_vit_h": build_semantic_sam_vit_h,
    "semantic_vit_t": build_semantic_sam_vit_t,
})

#################################################################################################################################
''' basic build function for SAM and Semantic SAM'''
#################################################################################################################################

def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes = 1,
    checkpoint=None,
):
    dev = args.devices
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        args,
        image_encoder=ImageEncoderViT(
            args = args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            #num_multimask_outputs=3,
            num_multimask_outputs = num_classes,
            transformer=TwoWayTransformer(
                args = args,
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
        

    #print(sam)
    if checkpoint is not None:
        state_dict = _safe_load(checkpoint)
        try:
           sam.load_state_dict(state_dict, strict=False)
        except Exception:
           new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
           sam.load_state_dict(new_state_dict)
            
    
    if args.if_split_encoder_gpus:
        sam.prompt_encoder = sam.prompt_encoder.to(dev[1])
        sam.mask_decoder = sam.mask_decoder.to(dev[1])
    return sam




# from https://github.com/11yxk/SAM-LST/blob/main/segment_anything/build_sam.py
def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict


def load_from_mobile(sam, state_dict):
    sam_dict = sam.state_dict()
    #except_keys = ['patch_embed','mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    sam_dict.update(new_state_dict)
    return sam_dict

# --- helpers to build base ViT SAM without loading (so we can load with our variant) ---
def _build_sam_skeleton_vit(args, encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes,
                            num_multimask_outputs=3):
    """Create a standard ViT SAM skeleton; leave checkpoint loading to the caller."""
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        args,
        image_encoder=ImageEncoderViT(
            args=args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_multimask_outputs,  # keep 3-way as in pretrain
            transformer=TwoWayTransformer(
                args=args,
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if getattr(args, "if_split_encoder_gpus", False):
        dev = args.devices
        sam.prompt_encoder = sam.prompt_encoder.to(dev[1])
        sam.mask_decoder = sam.mask_decoder.to(dev[1])
    return sam


# --- keep-mask-tokens loader (for semantic head) ---
def load_from_semantic(sam, state_dict, image_size, vit_patch_size):
    """
    Variant of load_from() that PRESERVES mask_tokens but drops the original
    decoder heads (output_hypernetworks_mlps, iou_prediction_head).
    This is what we want for SemanticSam.
    """
    sam_dict = sam.state_dict()
    except_keys = ['output_hypernetworks_mlps', 'iou_prediction_head']  # keep mask_tokens
    new_state_dict = {k: v for k, v in state_dict.items()
                      if k in sam_dict.keys() and not any(ex in k for ex in except_keys)}

    # Handle pos embedding resize if image size differs
    pos_key = 'image_encoder.pos_embed'
    if pos_key in new_state_dict:
        pos_embed = new_state_dict[pos_key]
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict[pos_key] = pos_embed

        # Resize relative position params for global attention layers
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            if k in new_state_dict:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w),
                                               mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict
