# copyright ziqi-jin

import torch.nn as nn
from models.sam.modeling.sam import Sam



class BasePromptEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BasePromptEncodeAdapter, self).__init__()

        self.sam_prompt_encoder = ori_sam.prompt_encoder
        

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings
