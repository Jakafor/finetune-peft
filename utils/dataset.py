# dataset.py — cleaned for point-only/boxes output and NumPy masks

import os
import random
import pickle
import numpy as np
from typing import Tuple, List

from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils.funcs import get_first_prompt, get_top_boxes


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
        img_folder: str,
        mask_folder: str,
        img_list: str,
        phase: str = 'train',
        sample_num: int = 50,                      # used for get_first_prompt(prompt_num=...)
        crop: bool = False,
        crop_size: int = 1024,
        targets: List[str] = ('femur', 'hip'),     # supports 'combine_all', 'multi_all', or specific class flow
        part_list: List[str] = ('all',),
        cls: int = -1,
        if_prompt: bool = True,
        prompt_type: str = 'point',                # 'point' | 'box'
        label_mapping: str = None,
        if_spatial: bool = True,
        delete_empty_masks: bool = True,
    ):
        super().__init__()
        self.args = args
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.targets = targets
        self.part_list = part_list
        self.cls = cls
        self.delete_empty_masks = delete_empty_masks
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.sample_num = sample_num
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
    def load_label_mapping(self):
        if self.label_mapping:
            with open(self.label_mapping, 'rb') as handle:
                self.segment_names_to_labels = pickle.load(handle)
            self.label_dic = {seg[1]: seg[0] for seg in self.segment_names_to_labels}
            self.label_name_list = [seg[0] for seg in self.segment_names_to_labels]
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
            msk = Image.open(os.path.join(self.mask_folder, mask_path)).convert('L')
            if self.should_keep(msk, mask_path):
                self.data_list.append(line)

        print(f'Filtered data list to {len(self.data_list)} entries.')

    def should_keep(self, msk, mask_path):
        if self.delete_empty_masks:
            mask_array = np.array(msk, dtype=int)
            if 'combine_all' in self.targets:
                return np.any(mask_array > 0)
            elif 'multi_all' in self.targets:
                return np.any(mask_array > 0)
            elif any(t in self.segment_names_to_labels for t in self.targets):
                target_classes = [self.segment_names_to_labels[t][1] for t in self.targets
                                  if t in self.segment_names_to_labels]
                return any(np.any(mask_array == cls) for cls in target_classes)
            elif self.cls > 0:
                return np.any(mask_array == self.cls)
            if self.part_list[0] != 'all':
                return any(part in mask_path for part in self.part_list)
            return False
        else:
            return True

    def setup_transformations(self):
        # Photometric transforms for the IMAGE only
        t_list = []
        if self.phase == 'train':
            t_list.extend([
                transforms.RandomEqualize(p=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ])
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
        img = Image.open(os.path.join(self.img_folder, img_path.strip())).convert('RGB')
        msk = Image.open(os.path.join(self.mask_folder, mask_path.strip())).convert('L')

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
        elif self.cls > 0:
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
        if len(msk.shape)==2:
            msk = torch.unsqueeze(torch.tensor(msk,dtype=torch.long),0)
        output = {'image': img, 'mask': msk, 'img_name': img_path}
        if self.if_prompt:
            # Assuming get_first_prompt and get_top_boxes functions are defined and handle prompt creation
            if self.prompt_type == 'point':
                points = get_first_prompt(msk.numpy()[0],self.sample_num)
                output.update({'points': points})
            elif self.prompt_type == 'box':
                boxes = get_top_boxes(msk.numpy()[0])
                output.update({'boxes': boxes})
            elif self.prompt_type == 'hybrid':
                points = get_first_prompt(msk.numpy()[0],self.sample_num)
                boxes = get_top_boxes(msk.numpy()[0])
                output.update({'points': points, 'boxes': boxes})
                
                


