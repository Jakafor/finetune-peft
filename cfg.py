
import argparse
import json

def _parse_list(s, cast=int, allow_none=True):
    """
    Parse a list from various formats:
      - Python list: [0,1,10,11]
      - JSON string: "[0,1,10,11]"
      - Comma/space list: "0,1,10,11" or "0 1 10 11"
      - "None" or "" -> None (if allow_none)
    """
    if s is None:
        return None if allow_none else []
    if isinstance(s, list):
        return [cast(x) for x in s]
    s = str(s).strip()
    if s == "" and allow_none:
        return None
    if s.lower() == "none":
        return None
    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [cast(x) for x in obj]
    except Exception:
        pass
    # Fallback: split by commas/spaces
    parts = [p for p in s.replace(",", " ").split() if p]
    return [cast(p) for p in parts]

def parse_args():
    parser = argparse.ArgumentParser(
        description="cfg for building SAM/MobileSAM or EnhancedSAMPEFT and exporting ONNX"
    )
    
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-arch', type=str, default='vit_b',
                        choices=['vit_h','vit_l','vit_b','vit_t'],
                        help='architecture: vit_h/l/b (SAM) or vit_t (MobileSAM)')
    # ---- Core model args needed by build_sam / modeling ----
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to base SAM/MobileSAM checkpoint (.pth/.pt)')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Encoder input size (square).')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='num_multimask_outputs for decoder.')
    parser.add_argument('--if_split_encoder_gpus', type=bool, default=False,
                        help='Split encoder across GPUs (if supported).')
    parser.add_argument('--devices', type=str, default='[0,1]',
                        help='GPU device indices, e.g. "[0,1]" or "0,1".')
    parser.add_argument('--gpu_fractions', type=str, default='[0.5,0.5]',
                        help='Fractions per device for split.')

    # ---- Adapter flags for SAM init (as you requested) ----
    parser.add_argument('--if_encoder_adapter', type=bool, default=True,
                        help='Whether to add adapters to the encoder.')
    parser.add_argument('--encoder_adapter_depths', type=str, default='[0,1,10,11]',
                        help='Encoder blocks to add adapters (JSON or comma/space list).')
    parser.add_argument('--if_mask_decoder_adapter', type=bool, default=False,
                        help='Whether to add adapters to every decoder transformer (aka if_adapter).')

    # ---- EnhancedSAMPEFT knobs (as you requested) ----
    parser.add_argument('--use_peft', action='store_true',
                        help='Wrap with EnhancedSAMPEFT before export.')
    parser.add_argument('--encoder_layers', type=str, default=None,
                        help='Encoder layers to apply PEFT (None => all).')
    parser.add_argument('--decoder_layers', type=str, default=None,
                        help='Decoder layers to apply PEFT (None => all).')
    parser.add_argument('--enable_adapters', action='store_true', default=False,
                        help='Enable existing adapters in the model (default False).')
    parser.add_argument('--adapter_modules', type=str, default=None,
                        help='Adapter module name patterns to enable (e.g. "Adapter MLP_Adapter").')
    parser.add_argument('--rank', type=int, default=4, help='PEFT rank.')
    parser.add_argument('--alpha', type=float, default=1.0, help='PEFT alpha.')
    parser.add_argument('--dropout', type=float, default=0.0, help='PEFT dropout.')

    # ---- Export knobs ----
    parser.add_argument('--output_dir', type=str, default='C:\\Users\\Jan Karl Forstner\\experiment\\exported_models',
                        help='Directory to save ONNX files.')
    parser.add_argument('--encoder_out', type=str, default=None,
                        help='Encoder ONNX path (defaults based on arch/name).')
    parser.add_argument('--decoder_out', type=str, default=None,
                        help='Decoder ONNX path (defaults based on arch/name).')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset.')
    parser.add_argument('--quantize', action='store_true', help='Also write *.quantized.onnx')
    parser.add_argument('--single_mask', action='store_true',
                        help='Export decoder that returns a single best mask.')

    args = parser.parse_args()

    # Post-processing for list-like fields
    args.devices = _parse_list(args.devices, cast=int, allow_none=False)
    args.gpu_fractions = _parse_list(args.gpu_fractions, cast=float, allow_none=False)
    args.encoder_adapter_depths = _parse_list(args.encoder_adapter_depths, cast=int, allow_none=False)
    # Optional lists
    args.encoder_layers = _parse_list(args.encoder_layers, cast=int, allow_none=True)
    args.decoder_layers = _parse_list(args.decoder_layers, cast=int, allow_none=True)
    if args.adapter_modules is not None:
        args.adapter_modules = _parse_list(args.adapter_modules, cast=str, allow_none=True)

    return args
