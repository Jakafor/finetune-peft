
import argparse
from email import parser
import json
from pathlib import Path
import importlib.util
import pickle

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

# method used for overwriting the num_classes/num_cls: token number/hypernetworks number adjusted according to the number of classes + 1 for the
# background. The information needed for that is imported from the label_mapping.pkl file we created when running the 
# label_mapping_utils from the annotator

def _infer_num_cls(args):
    ignore = getattr(args, "ignore_index", 255)
    pkl = getattr(args, "label_map_file", None) or getattr(args, "label_mapping", None)
    if pkl and Path(pkl).exists():
        lm = pickle.load(open(pkl, "rb"))
        ids = []
        for v in lm.values():
            if isinstance(v, int):
                ids.append(v)
            elif isinstance(v, dict) and "index" in v:
                ids.append(int(v["index"]))
        ids = [i for i in ids if i != ignore]
        return (max(ids) + 1) if ids else None
    elif (txt := getattr(args, "labels_file", None)) and Path(txt).exists():
        lines = [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]
        if len(lines) >= 2 and lines[0] == "__ignore__" and lines[1] == "_background_":
            return (len(lines) - 2) + 1  # fg + background
    return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="cfg for building SAM/MobileSAM or EnhancedSAMPEFT and exporting ONNX"
    )

    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-arch', type=str, default='vit_b',
                        choices=['vit_h','vit_l','vit_b','vit_t'],
                        help='architecture: vit_h/l/b (SAM) or vit_t (MobileSAM)')
    
    # ---- Core model args needed by build_sam / modeling ----
    parser.add_argument('--sam_ckpt', '--checkpoint', dest='sam_ckpt', type=str, default='auto',
                        help='Path to SAM/MobileSAM checkpoint (.pth/.pt)')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Encoder input size (square).')
    
    # number of classes: background class is 0, rest of classes have values 1...n: number of created classes is thus n+1
    parser.add_argument('--num_cls', '--num_classes', dest='num_cls', type=int, default=3,
                        help='Number of classes for segmentation (train baseline name)')
    parser.add_argument('--if_split_encoder_gpus', type=bool, default=False,
                        help='Split encoder across GPUs (if supported).')
    parser.add_argument('--devices', type=str, default='[0]',
                        help='GPU device indices, e.g. "[0,1]" or "0,1".')
    parser.add_argument('--gpu_fractions', type=str, default='[0.5,0.5]',
                        help='Fractions per device for split.')

    # ---- Adapter flags for SAM init -------------------------------------------------------------

    # Set to true if you want to add adapters during sam creation - to encoder
    parser.add_argument('--if_encoder_adapter', type=bool, default=True,
                        help='Whether to add adapters to the encoder.')
    
    # List of encoder blocks to add adapters to (default: all except 6,7,8,9) please familiarize yourself with the vit architecture if you want to change this!!
    parser.add_argument('--encoder_adapter_depths', type=str, default='[0,1,2,3,4,10,11]',
                        help='Encoder blocks to add adapters (JSON or comma/space list).')
    
    # Set to true if you want to add adapters during sam creation - to decoder
    parser.add_argument('--if_mask_decoder_adapter', type=bool, default=True,
                        help='Whether to add adapters to every decoder transformer (aka if_adapter).')
    
    # Decoder blocks to add adapters to (default: first two) - 2 is maximum, as there are only two transformer blocks in the decoder
    parser.add_argument('-decoder_adapt_depth', type=int, default=2, help='the depth of the decoder adapter')



    
    # ---- EnhancedSAMPEFT knobs ----


    # choose to apply peft to encoder
    parser.add_argument('--encoder_layers', type=str, default=None,
                        help='Encoder layers to apply PEFT (None => all).')
    
    # choose to apply peft to decoder
    parser.add_argument('--decoder_layers', type=str, default=None,
                        help='Decoder layers to apply PEFT (None => all).')
    
    # choose to enable adapter layers if present - only works for sam initialization with peft!
    parser.add_argument('--enable_adapters', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable adapter layers if present')
    
    # choose adapter modules to enable if present - only works for sam initialization with peft!
    parser.add_argument('--adapter_modules', type=str, default=None,
                        help='Adapter module name patterns to enable (e.g. "Adapter MLP_Adapter").')
    
    # in peft, choose to shut on/off the following parameters:
     # SemanticSam: 'decoder_head', 'decoder_neck', 'prompt_encoder', 'image_encoder', 'decoder' (both), 'all'
     # Regular SAM: 'encoder', 'decoder', 'image_encoder', 'prompt_encoder', 'all'
    parser.add_argument("--module_patterns", type = str, default = None,
                        help = "unfreeze whichever modules you like")
       

    
    # PEFT hyperparameters
    parser.add_argument('--peft_rank', '--rank', dest='peft_rank', type=int, default=4, help='PEFT rank (train baseline)')
    parser.add_argument('--peft_alpha', '--alpha', dest='peft_alpha', type=float, default=1.0, help='PEFT alpha (train baseline)')
    parser.add_argument('--peft_dropout', '--dropout', dest='peft_dropout', type=float, default=0.0, help='PEFT dropout (train baseline)')

    # choose between any one of the four peft methods implemented
    parser.add_argument('--peft_method', '--method', dest='peft_method', type=str, default='lora',
                        choices=['lora', 'lora_fa', 'vera', 'delta_lora'],
                        help='PEFT method (train baseline names)')
    



    # ---- Export knobs for ONNX ----

    parser.add_argument('--output_dir', type=str, default='C:\\Users\\Jan Karl Forstner\\experiment\\exported_models',
                        help='Directory to save ONNX files.')
    parser.add_argument('--encoder_out', type=str, default=None,
                        help='Encoder ONNX path (defaults based on arch/name).')
    parser.add_argument('--decoder_out', type=str, default=None,
                        help='Decoder ONNX path (defaults based on arch/name).')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset.')
    parser.add_argument('--quantize', action=argparse.BooleanOptionalAction,
                        default=True,help="Decoder returns a single best mask (use --no-single-mask for multi).",)
    parser.add_argument('--single_mask', action=argparse.BooleanOptionalAction,
                        default=True,help='Export decoder that returns a single best mask.')
    
    # ------- dataset / dataloader params -------------------------------------------------------------------
    # we use three parser arguments from the rock_annotator2 cfg:
    # --ann_dir (annotations dir, default "annotations")
    # --labels_file (labels.txt path, default "annotations/labels.txt")
    # --label_map_file (label_mapping.pkl path, default "annotations/label_mapping.pkl")
    # we will populate the defaults below if not set with the paths found in the rock_annotator2 cfg!
    # --------------------------------------------------------------------------------------------------------
    # ── From outer env (optional; can be inherited) ──────────────────────
    # --label_mapping as well but is dataset arg so defined below

    parser.add_argument('--ann_dir', type=Path, default=None,
                        help='Root annotations dir (inherits from outer cfg if available).')
    parser.add_argument('--labels_file', type=Path, default=None,
                        help='labels.txt path (inherits from outer cfg if available).')
    
   

# --------------------------call the cfg of rock_annotator2 to inherit the three args --ann_dir, --labels_file, --label_mapping--------------------------
    # VERY IMPORTANT: PASS THE PATH TO THE OUTER CFG TO INHERIT THE ABOVE THREE PARAMS (THE PATH IS ALWAYS DRIVE:/YOUR/PATH/GIT_REPO/cfg.py)
    #
    parser.add_argument('--outer_cfg', type=Path, default=Path("C:/Users/Jan Karl Forstner/git_repo/cfg.py"),
                        help='Path to parent env cfg.py to inherit ann_dir/labels_file/label_map_file.')
    
#----------------------------------------------------------------------------------------------
# -----------------------dataset args begin----------------------------------------------------
# ---------------------------------------------------------------------------------------------

    # chooses the classes we want to assign prompts to: 
    parser.add_argument('--targets', type=str, default=["bad"],
                    help='List of target class names or special tokens: "combine_all","multi_all",custom labels')

    # dataset label mapping file (label_mapping.pkl), also the dataset argument for Public_dataset class
    parser.add_argument('--label_mapping', '--label_map_file', dest='label_mapping',
                        type=Path, default=None,
                        help='label_mapping.pkl path (inherits from outer cfg if available).')
    parser.add_argument('--img_list', type=Path, default=None,
                    help='CSV with lines: "rel_img_path,rel_mask_path". Defaults to <ann_dir>/img_list.csv')
    
    # alters the training image for training augmentation
    parser.add_argument('--phase', type=str, default='train',
                    choices=['train', "test"],
                    help='Dataset phase.')

    # Number of points to sample for prompt generation
    parser.add_argument('--sample_num', type=int, default=3,
                    help='Number of points to sample for prompt generation.')

    # Ratio of negative points to positive points (e.g., 0.5 means half as many negative points as positive points)
    parser.add_argument('--neg_prompt_ratio', type=float, default=0.0,
                    help='Ratio of negative points to positive points.')
    


    #second round of optional cropping after transforms
    parser.add_argument('--crop', action=argparse.BooleanOptionalAction, default=False,
                    help='Enable random crop after transforms.')
    
    #used by both transforms spatial cropping and second round of cropping if --crop is enabled, adjust, at least for transforms - last size of image has to be 1024,1024 for sam
    parser.add_argument('--crop_size', type=int, default=1024,
                    help='Crop size used when --crop is enabled.')
    
    # -1 ignores the argument, any other number looks for the corresponding class and creates a binary semantic clasification
    parser.add_argument('--cls', type=int, default=1,
                    help='If >0, use this single class id for a binary mask.')

    # define if prompts are to be used for training: points are class agnostic
    parser.add_argument('--if_prompt', action=argparse.BooleanOptionalAction, default=True,
                    help='Include prompts (points/boxes) in dataset output.')
    
    # use point, box prompts or both at the same time: boxes are only used if at least 50 percent of the area enveloped by the box
    # belong to the class that is associated with the box
    parser.add_argument('--prompt_type', type=str, default='point',
                        choices=['point', 'box', 'hybrid'],
                        help='Type of prompts to generate.')
    
    # add some cropping to the target image -> make dataset variable
    parser.add_argument('--if_spatial', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply shared spatial transforms during training.')
    
    # delete masks in dataset before they reach the training dataset
    parser.add_argument('--delete_empty_masks', action=argparse.BooleanOptionalAction, default=True,
                        help='Filter out items with empty/irrelevant masks.')
    

# ----------------------------------------------------------------------------------------------
    # prompt and mask check utility (sam-slim-check)
# ----------------------------------------------------------------------------------------------

    parser.add_argument('--save_path', type=Path, default=None,
                    help='Path to save the prompt visualization (default: <img_name>_points_batches.png).')

    

# ---------------- training args for the sam finetuning ------------------------------------------------------------
    parser.add_argument('--finetune_type', type=str, default='peft',
                    choices=['vanilla', 'adapter', 'peft'],
                    help='Fine-tuning strategy')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--scheduler_type', type=str, default='poly', choices=['poly', 'cosine', 'step', 'exponential', 'plateau'], help='LR scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for scheduler')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Weight for Dice loss')
    parser.add_argument('--ce_weight', type=float, default=0.5, help='Weight for Cross Entropy loss')
    parser.add_argument('--use_focal', action='store_true', help='Use focal loss')
    parser.add_argument('--focal_weight', type=float, default=0.2, help='Weight for Focal loss')

    # train iou - the metric for mask quality estimation
    parser.add_argument('--train_iou', action=argparse.BooleanOptionalAction, default=True,
    help='Train IoU prediction head.')


    parser.add_argument('--iou_weight', type=float, default=0.1, help='Weight for IoU loss')
   #parser.add_argument('--train_list', type=str, required=True, help='Path to training data list')
   #parser.add_argument('--val_list', type=str, required=True, help='Path to validation data list')
    parser.add_argument('--out_size', type=int, default=256, help='Output mask size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--dir_checkpoint', type=str, default=r"C:\Users\Jan Karl Forstner\experiment\sam",
                    help='Directory for saving checkpoints')

    # If not present or False: Freezes encoder weights, used in the non - peft sam creation, decoder is trainable
    parser.add_argument('--if_update_encoder', action='store_true', help='Whether to update encoder')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--layer_decay', type=float, default=0.9, help='Layer-wise LR decay for TinyViT')

    # SemanticSAM specific, set it to True to use it always. If set to False, it will be used only if num_cls>3 (2 foreground + background)
    parser.add_argument('--use_semantic', action=argparse.BooleanOptionalAction, default=False,help='Use SemanticSam variant.')



# ------------------inheriting from outer cfg.py if available (for dataset args)----

    def _inherit_first_env_defaults(args):

        # If all three are set, nothing to do
        if args.outer_cfg is None and args.ann_dir and args.labels_file and args.label_mapping:
            return args  

        def _try(module_path: Path):
            if not module_path or not module_path.exists():
                return None
            spec = importlib.util.spec_from_file_location("outer_cfg", module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # executes outer cfg.py but doesn't parse our argv
            outer = mod.get_parser()      # build its parser (add_help=False in their code)
            # Find defaults for the three options
            def _default(flag):
                for a in outer._actions:
                    if flag in a.option_strings:
                        return a.default
                return None
            return {
                "ann_dir":        _default("--ann_dir"),
                "labels_file":    _default("--labels_file"),
                "label_mapping":  _default("--label_map_file"),
            }

        # Try file-based import first if provided
        inherited = _try(args.outer_cfg) if args.outer_cfg else None

        # Fallback: if user didn’t pass --outer_cfg, try a sane default search (optional)
        if inherited is None:
            # Heuristic: look one level up for cfg.py
            parent_cfg = Path.cwd().parent / "cfg.py"
            inherited = _try(parent_cfg) if parent_cfg.exists() else None

        # Last-ditch defaults if nothing can be inherited
        if inherited is None:
            base = Path.cwd() / "annotations"
            inherited = {
                "ann_dir": base,
                "labels_file": base / "labels.txt",
                "label_mapping": base / "label_mapping.pkl",
        }

        # Fill missing fields only
        if args.ann_dir is None:
            args.ann_dir = inherited["ann_dir"]
        if args.labels_file is None:
            args.labels_file = inherited["labels_file"] or (args.ann_dir and args.ann_dir / "labels.txt")
        if args.label_mapping is None:
            args.label_mapping = inherited["label_mapping"] or (args.ann_dir and args.ann_dir / "label_mapping.pkl")
        if getattr(args, "img_list", None) in (None, "", "auto"):
            if args.ann_dir is None:
                raise ValueError("--img_list not provided and --ann_dir unavailable; cannot infer <ann_dir>/img_list.csv")
            args.img_list = args.ann_dir / "img_list.csv"
        return args
    
    
    args = parser.parse_args()
    args = _inherit_first_env_defaults(args)

    # allows sam-ckpt (where the sam weights downloaded from internet are stored to be set to None
    # auto creates a new .cache where they are stored
    val = None if args.sam_ckpt is None else str(args.sam_ckpt).strip().lower()
    if val in (None, "", "auto", "none"):
        args.sam_ckpt = None

    # Post-processing for list-like fields - [0,1] to [cuda:0,cuda:1] etc.
    devs = _parse_list(args.devices, cast=str, allow_none=False)
    # Normalize to torch.device-friendly strings: allow 'cuda:0', '0', 'cpu'
    norm = []
    for d in devs:
        sdev = str(d).strip().lower()
        if sdev.isdigit():
            norm.append(f'cuda:{sdev}')
        elif sdev.startswith('cuda'):
            norm.append(sdev)
        elif sdev == 'cpu':
            norm.append('cpu')
        else:
            norm.append(sdev)
    args.devices = norm
    args.gpu_fractions = _parse_list(args.gpu_fractions, cast=float, allow_none=False)

    # targets can either be multi_all, combine_all or a indeterminate 
    if isinstance(args.targets, str) and args.targets not in ("multi_all", "combine_all"):
         args.targets = _parse_list(args.targets, cast=str, allow_none=False)

    args.encoder_adapter_depths = _parse_list(args.encoder_adapter_depths, cast=int, allow_none=False)
    # Optional lists
    args.encoder_layers = _parse_list(args.encoder_layers, cast=int, allow_none=True)

    args.decoder_layers = _parse_list(args.decoder_layers, cast=int, allow_none=True)
    if args.adapter_modules is not None:
        args.adapter_modules = _parse_list(args.adapter_modules, cast=str, allow_none=True)

    # added a function that converts "decoder","encoder" like series of strings into a list
    args.module_patterns= _parse_list(args.module_patterns, cast=str, allow_none=True)

    

    # call the function that replaces the number of classes if it should be wrong: NUMBER OF CLASSES IS ALWAYS THE NUMBER
    # OF THE MANUALLY CREATED CLASSES PLUS THE BACKGROUND: IF YOUR GROUND TRUTH MASK HAS A BACKGROUND ALWAYS SET IT TO 0!!
    # set the num_cls to 2 for the binary cases of "combine_all" and "cls" > 0 (class + background)
    # set the num_cls to n (number of created classes) + 1 (background)
    # set the num_cls to k (slice of n determined by the list you pass in args.targets) + 1 (background)
    inferred = _infer_num_cls(args)
    new_cls = None
    if args.targets == "combine_all" or getattr(args, "cls", 0) > 0:
        new_cls = 2
    elif args.targets == "multi_all":
        if inferred is not None:
            new_cls = inferred
    elif isinstance(args.targets, (list, tuple)):
        # we remap to contiguous ids → 1..K plus background
        # (targets already normalized to a list above)
        new_cls = len(dict.fromkeys(args.targets)) + 1  # unique count + bg
    # else: single label string → binary vs bg
    elif isinstance(args.targets, str):
        new_cls = 2

    # Preserve your original safety check:
    if new_cls is not None and new_cls != args.num_cls:
        print(f"[cfg] num_cls: {args.num_cls} -> {new_cls}")
        args.num_cls = new_cls

    ####checks if targets is a list (from cfg inheritance) or a str (from command line): adds it to list that makes up targets if not##########################
    if isinstance(args.targets, str) and args.targets not in ('labels_file', 'combine_all', 'multi_all'):
        args.targets = _parse_list(args.targets, cast=str, allow_none=False)

    return args
