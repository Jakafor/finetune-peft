import torch.nn as nn
from .sam import Sam



class BaseImgEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseImgEncodeAdapter, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x
