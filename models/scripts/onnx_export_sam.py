
"""
Unified ONNX exporter for SAM / MobileSAM with optional EnhancedSAMPEFT.

- Builds a base SAM/MobileSAM from build_sam.sam_model_registry using args.arch
- Optionally wraps it with EnhancedSAMPEFT from sam_Peft.create_peft_sam
- Exports two ONNX files:
    * encoder: input 'x' -> output 'image_embeddings'
    * decoder: using models.sam.utils.onnx.SamOnnxModel (or Meta fallback)

Usage example:
  python onnx_export.py --arch vit_t --checkpoint /path/to/mobile_sam.pt --use_peft --rank 8 --alpha 8 --output_dir ./exported
"""
import os, hashlib, warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from cfg import parse_args
from models.sam import sam_model_registry
from models.sam_Peft import create_peft_sam

try:
    from models.sam.utils.onnx import SamOnnxModel
except Exception:
    from segment_anything.utils.onnx import SamOnnxModel  # type: ignore


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class EncoderWrapper(nn.Module):
    """Expose encoder with input 'x' and output 'image_embeddings' for ONNX."""
    def __init__(self, sam):
        super().__init__()
        self.image_encoder = sam.image_encoder
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)


def build_model(args):
    # Build base SAM / MobileSAM (vit_h/l/b or vit_t)
    builder = sam_model_registry[args.arch]
    sam = builder(args, checkpoint=args.checkpoint, num_classes=args.num_classes)
    sam.eval()

    name_tag = args.arch

    # Optionally wrap with EnhancedSAMPEFT
    if args.use_peft:
        peft = create_peft_sam(
            sam_model=sam,
            method="lora",  # default method; adjust if you expose it via cfg
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            enable_adapters=args.enable_adapters,
            adapter_modules=args.adapter_modules,
        )
        # Export the underlying patched SAM graph
        sam = peft.sam
        name_tag += ".peft"

    return sam, name_tag


def export_encoder(sam, out_path: str, opset: int, image_size: int):
    enc = EncoderWrapper(sam).eval()
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    torch.onnx.export(
        enc,
        dummy,
        out_path,
        input_names=["x"],
        output_names=["image_embeddings"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"x": {0: "batch"}, "image_embeddings": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(out_path))
    print("Encoder:", out_path, sha256(out_path))


def export_decoder(sam, out_path: str, opset: int, return_single_mask: bool):
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=False,
        return_extra_metrics=False,
    ).eval()

    embed_dim = sam.prompt_encoder.embed_dim
    es_h, es_w = sam.prompt_encoder.image_embedding_size
    mask_h, mask_w = 4 * es_h, 4 * es_w

    dummy = {
        "image_embeddings": torch.randn(1, embed_dim, es_h, es_w, dtype=torch.float32),
        "point_coords": torch.randn(1, 5, 2, dtype=torch.float32),       # N+1 incl. dummy
        "point_labels": torch.tensor([[1, 0, 1, 0, -1]], dtype=torch.float32),
        "mask_input": torch.zeros(1, 1, mask_h, mask_w, dtype=torch.float32),
        "has_mask_input": torch.tensor([0.0], dtype=torch.float32),      # 0 => no prior
        "orig_im_size": torch.tensor([float(4*es_h*4), float(4*es_w*4)], dtype=torch.float32),
    }
    _ = onnx_model(**dummy)  # materialize

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            onnx_model,
            tuple(dummy.values()),
            out_path,
            input_names=list(dummy.keys()),
            output_names=["masks", "iou_predictions", "low_res_masks"],
            dynamic_axes={
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(out_path))
    print("Decoder:", out_path, sha256(out_path))


def quantize(in_path: str, out_path: str):
    try:
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
    except Exception as e:
        print("Quantization skipped:", e)
        return None
    print(f"Quantizing {os.path.basename(in_path)} → {os.path.basename(out_path)} ...")
    quantize_dynamic(
        model_input=in_path,
        model_output=out_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )
    print("Quantized:", out_path, sha256(out_path))
    return out_path


def self_test(enc_path: str, dec_path: str, image_size: int):
    print("\n[ORT self-test]")
    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])

    x = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    (emb,) = enc.run(None, {"x": x})
    es_h, es_w = emb.shape[2], emb.shape[3]
    mask_h, mask_w = 4 * es_h, 4 * es_w

    feeds = {
        "image_embeddings": emb,
        "point_coords": np.array([[[image_size/2, image_size/2], [0, 0]]], dtype=np.float32),
        "point_labels": np.array([[1, -1]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, mask_h, mask_w), dtype=np.float32),
        "has_mask_input": np.array([0.0], dtype=np.float32),
        "orig_im_size": np.array([image_size, image_size], dtype=np.float32),
    }
    masks, scores, lowres = dec.run(None, feeds)
    print("Emb:", emb.shape, "Masks:", masks.shape, "Scores:", scores.shape, "Lowres:", lowres.shape)
    print("[ORT self-test] OK\n")


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sam, name_tag = build_model(args)

    enc_out = args.encoder_out or str(Path(args.output_dir) / f"{name_tag}.encoder.onnx")
    dec_out = args.decoder_out or str(Path(args.output_dir) / f"{name_tag}.decoder.onnx")

    export_encoder(sam, enc_out, opset=args.opset, image_size=args.image_size)
    export_decoder(sam, dec_out, opset=args.opset, return_single_mask=args.single_mask)

    if args.quantize:
        enc_q = os.path.splitext(enc_out)[0] + ".quantized.onnx"
        dec_q = os.path.splitext(dec_out)[0] + ".quantized.onnx"
        quantize(enc_out, enc_q)
        quantize(dec_out, dec_q)
        enc_out, dec_out = enc_q, dec_q

    self_test(enc_out, dec_out, image_size=args.image_size)

    print("Encoder hash:", sha256(enc_out))
    print("Decoder hash:", sha256(dec_out))


if __name__ == "__main__":
    main()
