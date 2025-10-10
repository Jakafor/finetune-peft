# @copyright ziqi-jin

import torch.nn as nn
import torch
from .sam import Sam
#from .utils import fix_params
from .mask_decoder import MaskDecoder
from typing import List, Tuple
from torch.nn import functional as F
from .mask_decoder_heads import SemSegHead
from .mask_decoder_neck import MaskDecoderNeck


class BaseMaskDecoderAdapter(MaskDecoder):
    '''
      multimask_output (bool): If true, the model will return three masks.
    For ambiguous input prompts (such as a single click), this will often
    produce better masks than a single prediction. If only a single
    mask is needed, the model's predicted quality score can be used
    to select the best mask. For non-ambiguous prompts, such as multiple
    input prompts, multimask_output=False can give better results.
    '''

    # is fix and load params
    def __init__(self, ori_sam: Sam):
        super(BaseMaskDecoderAdapter, self).__init__(transformer_dim=ori_sam.mask_decoder.transformer_dim,
                                                     transformer=ori_sam.mask_decoder.transformer)
        self.sam_mask_decoder = ori_sam.mask_decoder

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True):
        low_res_masks, iou_predictions = self.sam_mask_decoder(image_embeddings=image_embeddings,
                                                                   image_pe=prompt_adapter.sam_prompt_encoder.get_dense_pe(),
                                                                   sparse_prompt_embeddings=sparse_embeddings,
                                                                   dense_prompt_embeddings=dense_embeddings,
                                                                   multimask_output=multimask_output, )
        return low_res_masks, iou_predictions


class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):
    def __init__(self, ori_sam: Sam, class_num=20):
        super(SemMaskDecoderAdapter, self).__init__(ori_sam)
        self.decoder_neck = MaskDecoderNeck(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                            transformer=self.sam_mask_decoder.transformer,
                                            num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs)
        self.decoder_head = SemSegHead(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                       num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs,
                                       iou_head_depth=self.sam_mask_decoder.iou_head_depth,
                                       iou_head_hidden_dim=self.sam_mask_decoder.iou_head_hidden_dim,
                                       class_num=class_num)
        # pair the params between ori mask_decoder and new mask_decoder_adapter
        self.pair_params(self.decoder_neck)
        self.pair_params_smart(self.decoder_head, class_num)

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True,
                scale=1):
        src, iou_token_out, mask_tokens_out, src_shape = self.decoder_neck(image_embeddings=image_embeddings,
                                                                           image_pe=prompt_adapter.sam_prompt_encoder.get_dense_pe(),
                                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                                           dense_prompt_embeddings=dense_embeddings,
                                                                           multimask_output=multimask_output, )
        masks, iou_pred = self.decoder_head(src, iou_token_out, mask_tokens_out, src_shape, mask_scale=scale)
        return masks, iou_pred

    def pair_params(self, target_model: nn.Module):
        src_dict = self.sam_mask_decoder.state_dict()
        for name, value in target_model.named_parameters():
            if name in src_dict.keys():
                value.data.copy_(src_dict[name].data)

    ##### adaptation from original semantic segmentation head ####################

    def pair_params_smart(self, target_model: nn.Module, class_num: int):
        """
        Intelligently copy parameters from original SAM decoder:
        - For hypernetworks: reuse the first N pre-trained networks (where N = min(4, class_num))
        - For output_upscaling: copy all (architecture matches)
        - For IoU head: skip (different output dimension)
        """
        src_dict = self.sam_mask_decoder.state_dict()
        target_dict = target_model.state_dict()
        
        # Get the number of original mask tokens (typically 4: 1 + 3 multimask)
        num_original_masks = self.sam_mask_decoder.num_mask_tokens
        
        for target_name, target_param in target_model.named_parameters():
            # Handle output_upscaling layers (same architecture, copy all)
            if 'output_upscaling' in target_name:
                if target_name in src_dict:
                    target_param.data.copy_(src_dict[target_name].data)
                    print(f"Copied: {target_name}")
            
            # Handle hypernetwork MLPs - reuse pre-trained ones where possible
            elif 'output_hypernetworks_mlps' in target_name:
                # Extract the MLP index from the parameter name
                # Format: output_hypernetworks_mlps.INDEX.layers.LAYER.weight/bias
                parts = target_name.split('.')
                if len(parts) >= 2:
                    try:
                        mlp_idx = int(parts[1])
                        # If this index exists in original model, copy the weights
                        if mlp_idx < num_original_masks:
                            if target_name in src_dict:
                                target_param.data.copy_(src_dict[target_name].data)
                                print(f"Reused pre-trained hypernetwork {mlp_idx}: {target_name}")
                        else:
                            print(f"Randomly initialized hypernetwork {mlp_idx}: {target_name}")
                    except (ValueError, IndexError):
                        pass
            
            # Skip IoU prediction head (different output dimensions)
            elif 'iou_prediction_head' in target_name:
                print(f"Skipped (different dims): {target_name}")
                # These will remain randomly initialized
        
        # Print summary
        print(f"\nParameter initialization summary:")
        print(f"- Reused {min(num_original_masks, class_num)} pre-trained hypernetworks out of {class_num} total")
        print(f"- Randomly initialized {max(0, class_num - num_original_masks)} new hypernetworks")
        print(f"- Copied all output_upscaling layers")
        print(f"- Randomly initialized IoU head for {class_num} classes")


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
