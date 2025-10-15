# dataset.py — cleaned for point-only/boxes output and NumPy masks

import os
import random
import pickle
import numpy as np
from typing import Tuple, List

from pathlib import Path
import json


from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .funcs import get_first_prompt, get_top_boxes
from .. import cfg


class Public_dataset(Dataset):
    """
    Public dataset for SAM fine-tuning.

    Key behavior changes:
    - prepare_output() no longer modifies the mask; it returns the original mask (NumPy).
    - Prompts are unified: output['points'] contains points; labels are not returned.
    - Boxes (if requested) are returned in output['boxes'].
    - Image is torch.Tensor; mask is np.ndarray.
    - normalize_type, region_type, channel_num removed.
    """

    def __init__(
        self,
        args,
        # img_folder: str,
        # mask_folder: str,
        img_list: str,
        phase: str = 'train',
        sample_num: int = 50,                      # used for get_first_prompt(prompt_num=...)
        crop: bool = False,
        crop_size: int = 1024,
        targets: List[str] = ("multi_all",), 
        neg_prompt_ratio: float = 0.0,            # supports 'combine_all', 'multi_all', or specific class flow
        # part_list: List[str] = ('all',),
        cls: int = -1,
        if_prompt: bool = True,
        prompt_type: str = 'point',                # 'point' | 'box'
        label_mapping: str = None,
        if_spatial: bool = True,
        delete_empty_masks: bool = True,
    ):
        super().__init__()
        self.args = args
        #elf.img_folder = img_folder
        #elf.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase

        #targets can be 'combine_all', 'multi_all', or specific class names - for these targets a ground truth mask is created
        # either a binary mask of all foreground classes and a background class (combine all)
        # a multiclass mask of all available classes in the original mask (multi all)
        # a multiclass mask of a list of available labels 
        # a binary mask of a certain cls index (number) and background
        # DO NOT MIX WHILE TRAINING! 
        self.targets = self._finalize_targets(targets)

        # self.part_list = part_list
        self.cls = cls
        self.delete_empty_masks = delete_empty_masks
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.sample_num = sample_num
        self.neg_prompt_ratio = neg_prompt_ratio
        self.label_dic = {}
        self.data_list: List[str] = []  
        self.label_mapping = label_mapping
        self.if_spatial = if_spatial
        

        self.load_label_mapping()
        self.load_data_list(img_list)
        self.setup_transformations()

    # ----------------------------
    # init helpers
    # ----------------------------


    
    def _finalize_targets(self, targets_in):
        """
        Accept:
        - list/tuple[str]  → return as list
        - 'combine_all'/'multi_all' → return unchanged (dataset handles this)
        - any other str            → single-label list [str]
        """
        # Already a list/tuple
        if isinstance(targets_in, (list, tuple)):
            return list(targets_in)

        # Special flows: leave exactly as-is for downstream logic
        if isinstance(targets_in, str) and targets_in in {"combine_all", "multi_all"}:
            return targets_in
        
        return targets_in
       
    
#--------------------------------------------------------------------------------------------
    
    def load_label_mapping(self):
        if self.label_mapping:
            with open(self.label_mapping, 'rb') as handle:
                mapping = pickle.load(handle)
            # Accept either dict{name:id} or list/iterable of (name,id)
            if isinstance(mapping, dict):
                self.segment_names_to_labels = mapping
            else:
                self.segment_names_to_labels = {k: v for k, v in mapping}
            self.label_dic = {v: k for k, v in self.segment_names_to_labels.items()}
            self.label_name_list = list(self.segment_names_to_labels.keys())
            print(self.label_dic)
        else:
            self.segment_names_to_labels = {}
            self.label_dic = {value: 'all' for value in range(1, 256)}


    def load_data_list(self, img_list):
        with open(img_list, 'r') as file:
            lines = file.read().strip().split('\n')
        for line in lines:
            img_path, mask_path = line.split(',')
            mask_path = mask_path.strip()
            if mask_path.startswith('/'):
                mask_path = mask_path[1:]
            msk = Image.open(Path(mask_path))
            if self.should_keep(msk, mask_path):
                self.data_list.append(line)

        print(f'Filtered data list to {len(self.data_list)} entries.')

    def should_keep(self, msk, mask_path):
        if not self.delete_empty_masks: 
            return True
        arr = np.array(msk if msk.mode in ('P','L','I') else msk.convert('P'), dtype=int)

        if 'combine_all' in self.targets or 'multi_all' in self.targets:
            return np.any(arr > 0)
        elif any(t in self.segment_names_to_labels for t in self.targets):
            ids = [self.segment_names_to_labels[t] for t in self.targets if t in self.segment_names_to_labels]
            return any((arr == i).any() for i in ids)
        elif self.cls > 0:
            return (arr == self.cls).any()
        return False


    def setup_transformations(self):
        # Photometric transforms for the IMAGE only
        t_list = []
        if self.phase == 'train':
            t_list.extend([
                transforms.RandomEqualize(p=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ])
            if self.if_spatial:
                self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 1), interpolation=InterpolationMode.NEAREST),
                         transforms.RandomRotation(45, interpolation=InterpolationMode.NEAREST)])
                
        # Always: ToTensor + "SAM" normalization
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transform_img = transforms.Compose(t_list)

    # ----------------------------
    # Dataset API
    # ----------------------------
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        img_path, mask_path = data.split(',')
        if mask_path.startswith('/'):
            mask_path = mask_path[1:]

        # Load PIL
        img = Image.open(Path(img_path.strip())).convert('RGB')
        msk = Image.open(Path(mask_path.strip()))

        # Uniform resize first (PIL, NEAREST for mask)
        img = transforms.Resize((self.args.image_size, self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size, self.args.image_size),
                                InterpolationMode.NEAREST)(msk)

        # Apply crop/aug; returns (torch image, numpy mask)
        img, msk = self.apply_transformations(img, msk)

        # Optional target mapping (kept as before, but mask stays NumPy)
        if 'combine_all' in self.targets:          # binary foreground
            msk = (np.array(msk, dtype=int) > 0).astype(int)
        elif 'multi_all' in self.targets:
            msk = np.array(msk, dtype=int)
        # refactors thefound classes in the targets list -> what was once a list of [class2,class4,class1] gets refactored to [class1,class2,class3]
        # creation of a brand new multiclass mask with new class values. Num_classes then are the len of the refactored target list + 1 for background
        elif isinstance(self.targets, (list, tuple)) and any(t in self.segment_names_to_labels for t in self.targets):
            arr = np.array(msk, dtype=int)
            seen, selected_ids = set(), []
            for t in self.targets:
                if t in self.segment_names_to_labels:
                    cid = int(self.segment_names_to_labels[t])
                    if cid not in seen:
                        seen.add(cid)
                        selected_ids.append(cid)

            # build LUT: 0 stays background, selected map to 1..K
            lut_size = max(int(arr.max()), max(selected_ids, default=0))
            lut = np.zeros(lut_size + 1, dtype=np.int64)
            for k, cid in enumerate(selected_ids, start=1):
                lut[cid] = k

            msk = lut[arr]
            msk = (np.array(msk, dtype=int) == self.cls).astype(int)
        else:
            msk = np.array(msk, dtype=int)

        return self.prepare_output(img, msk, img_path, mask_path)
    
    def apply_transformations(self, img, msk):
    # If cropping is enabled, keep using your existing apply_crop
        if self.crop:
            img, msk = self.apply_crop(img, msk)

        # Image transforms stay exactly as configured in setup_transformations
        img = self.transform_img(img)

        # Mask to torch once (long); no NumPy ping-pong
        msk_t = torch.tensor(np.array(msk, dtype=int), dtype=torch.long)

        # Shared spatial transforms during training (unchanged list/callable)
        if self.phase == 'train' and self.if_spatial:
            # expand mask to 3ch in torch to appease spatial transforms
            mask3 = msk_t.unsqueeze(0).to(dtype=img.dtype).expand(3, -1, -1)  # (3,H,W)

            # stack image + mask3 along a pseudo-batch dim → (2,3,H,W)
            both = torch.stack((img, mask3), dim=0)

            # keep behavior whether self.transform_spatial is a Compose or a list
            ts = self.transform_spatial
            if callable(ts):
                out = ts(both)
            else:
                tmp = both
                for t in ts:
                    tmp = t(tmp)
                out = tmp

            # split back
            img = out[0]                       # torch.Tensor (3,H,W)
            msk_t = out[1][0].to(torch.long)   # torch.Tensor (H,W)

        # final: mask as NumPy int, image stays torch
        msk_np = msk_t.detach().cpu().numpy().astype(int)
    
        return img, msk_np
    
    def apply_crop(self, img_pil: Image.Image, msk_pil: Image.Image):
        t, l, h, w = transforms.RandomCrop.get_params(img_pil, (self.crop_size, self.crop_size))
        img_pil = TF.crop(img_pil, t, l, h, w)
        msk_pil = TF.crop(msk_pil, t, l, h, w)
        return img_pil, msk_pil
    
    def prepare_output(self, img, msk, img_path, mask_path):
        msk = torch.tensor(msk, dtype=torch.long)
        if len(msk.shape)==2:
            msk = msk.unsqueeze(0)  # (1,H,W)
        output = {'image': img, 'mask': msk, 'img_name': img_path}
        if self.if_prompt:
            # Assuming get_first_prompt and get_top_boxes functions are defined and handle prompt creation
            if self.prompt_type == 'point':
                points = get_first_prompt(msk.numpy()[0],sample_num=self.sample_num, neg_prompt_ratio=self.neg_prompt_ratio)
                output.update({'points': points})
            elif self.prompt_type == 'box':
                boxes = get_top_boxes(msk.numpy()[0])
                output.update({'boxes': boxes})
            elif self.prompt_type == 'hybrid':
                points = get_first_prompt(msk.numpy()[0],sample_num=self.sample_num)
                boxes = get_top_boxes(msk.numpy()[0])
                output.update({'points': points, 'boxes': boxes})
        return output
                
                


