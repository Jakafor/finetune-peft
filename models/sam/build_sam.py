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
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
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