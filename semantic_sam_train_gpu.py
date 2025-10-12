"""
Semantic SAM Training Pipeline
Supports both mask token strategy (Version A) and output hypernetwork strategy (Version B)
with comprehensive logging, IoU training, and professional evaluation metrics
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import monai
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
import torch.nn.functional as F

# Import your modules
from finetune_sam_slim.models.sam import sam_model_registry
from finetune_sam_slim.utils.dataset import Public_dataset
from finetune_sam_slim.models.advanced_sam_peft import create_peft_sam, PEFTMethod
from finetune_sam_slim import cfg


#################################################################################################

# --- collate: single-sample passthrough, force DataLoader batch_size=1 ---
def sam_collate_single(batch):
    assert len(batch) == 1, "Use batch_size=1 with this collate."
    b0 = batch[0]
    out = dict(b0)
    # (3,H,W) -> (1,3,H,W)
    if out['image'].dim() == 3:
        out['image'] = out['image'].unsqueeze(0)
    # (1,H,W) -> (1,1,H,W)
    if 'mask' in out and out['mask'] is not None and out['mask'].dim() == 3:
        out['mask'] = out['mask'].unsqueeze(0)
    return out


##################################################################################################

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
def dice_coeff_multi_class(
    pred: torch.Tensor,
    target: torch.Tensor, 
    num_classes: int,
    from_logits: bool = True,
    include_background: bool = False,
    ignore_index: Optional[int] = None,
    class_weight: Optional[torch.Tensor] = None,
    average: str = "macro",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Dice coefficient for multi-class segmentation.
    
    Args:
        pred: (B,H,W) hard labels or (B,C,H,W) logits/probabilities
        target: (B,H,W) integer labels or (B,1,H,W) or (B,C,H,W) one-hot
        num_classes: Number of classes
        from_logits: If True, apply softmax to pred
        include_background: If False, exclude class 0 from computation
        ignore_index: Label to ignore in target
        class_weight: Per-class weights
        average: 'macro', 'micro', or 'none'
        eps: Small value to avoid division by zero
    
    Returns:
        Dice coefficient (scalar if average != 'none')
    """
    # Handle input shapes
    if pred.dim() == 3:  # (B,H,W) - hard predictions
        B, H, W = pred.shape
        pred_1h = F.one_hot(pred.long().clamp(0, num_classes-1), num_classes)
        pred_1h = pred_1h.permute(0, 3, 1, 2).float()  # (B,C,H,W)
        probs = pred_1h
    elif pred.dim() == 4:  # (B,C,H,W)
        B, C, H, W = pred.shape
        if from_logits:
            probs = F.softmax(pred, dim=1)
        else:
            probs = pred.float()
    else:
        raise ValueError(f"pred must be (B,H,W) or (B,C,H,W), got {pred.shape}")
    
    # Handle target shapes
    if target.dim() == 3:  # (B,H,W)
        target = target.long().clamp(0, num_classes-1)
        target_1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    elif target.dim() == 4:
        if target.shape[1] == 1:  # (B,1,H,W)
            target = target.squeeze(1).long().clamp(0, num_classes-1)
            target_1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        else:  # (B,C,H,W) - already one-hot
            target_1h = target.float()
    else:
        raise ValueError(f"target must be (B,H,W), (B,1,H,W) or (B,C,H,W), got {target.shape}")
    
    # Apply ignore_index mask if specified
    if ignore_index is not None:
        valid_mask = (target != ignore_index).float()
        if valid_mask.dim() == 3:  # (B,H,W)
            valid_mask = valid_mask.unsqueeze(1)  # (B,1,H,W)
        probs = probs * valid_mask
        target_1h = target_1h * valid_mask
    
    # Optionally exclude background
    start_idx = 0 if include_background else 1
    probs = probs[:, start_idx:, :, :]
    target_1h = target_1h[:, start_idx:, :, :]
    
    # Compute dice per class
    dims = (0, 2, 3)  # Batch and spatial dimensions
    intersection = (probs * target_1h).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_1h.sum(dim=dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    
    # Apply class weights if provided
    if class_weight is not None:
        weight = torch.as_tensor(class_weight, device=dice_per_class.device, dtype=dice_per_class.dtype)
        if not include_background and weight.numel() == num_classes:
            weight = weight[1:]  # Remove background weight
        dice_per_class = dice_per_class * weight
    
    # Handle absent classes to avoid bias
    pred_sum = probs.sum(dim=dims)
    tgt_sum = target_1h.sum(dim=dims)
    present = (pred_sum > 0) | (tgt_sum > 0)
    dice_per_class = torch.where(present, dice_per_class, torch.nan)

    # Return based on averaging method
    if average == 'none':
        return dice_per_class
    elif average == 'micro':
        # For micro, only count present classes
        valid_dice = dice_per_class[~torch.isnan(dice_per_class)]
        return valid_dice.sum() / valid_dice.numel() if valid_dice.numel() > 0 else torch.tensor(0.0)
    else:  # macro
        return torch.nanmean(dice_per_class)
    


class CombinedLoss(nn.Module):
    """Flexible loss combination with configurable weights"""
    
    def __init__(self, loss_configs: Dict[str, Dict], num_classes: int):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.num_classes = num_classes
        self.dice_config = None
        
        for name, config in loss_configs.items():
            weight = config.pop('weight', 1.0)
            self.weights[name] = weight
            
            if name == 'dice':
                # Store dice config for internal computation
                self.dice_config = config
                self.losses[name] = None  # Placeholder
            elif name == 'ce':
                self.losses[name] = nn.CrossEntropyLoss(**config)
            elif name == 'focal':
                self.losses[name] = FocalLoss(**config)
            elif name == 'tversky':
                self.losses[name] = TverskyLoss(**config)
            else:
                raise ValueError(f"Unknown loss type: {name}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns total loss and individual loss components"""
        total_loss = 0
        loss_dict = {}
        
        for name in self.losses.keys():
            if name == 'dice':
                # Use internal dice computation
                dice_score = dice_coeff_multi_class(
                    pred=pred,
                    target=target,
                    num_classes=self.num_classes,
                    **self.dice_config
                )
                loss_val = 1.0 - dice_score  # Convert Dice to loss
            elif name == 'ce':
                # CE expects long targets without channel dimension
                if target.dim() == 4 and target.shape[1] == 1:
                    target_ce = target.squeeze(1).long()
                else:
                    target_ce = target.long()
                loss_val = self.losses[name](pred, target_ce)
            elif name in ['focal', 'tversky']:
                # Similar to CE
                if target.dim() == 4 and target.shape[1] == 1:
                    target_focal = target.squeeze(1).long()
                else:
                    target_focal = target.long()
                loss_val = self.losses[name](pred, target_focal)
            
            total_loss += self.weights[name] * loss_val
            loss_dict[name] = loss_val
        
        return total_loss, loss_dict


class IoULoss(nn.Module):
    """Loss for training IoU prediction head"""
    
    def __init__(self, loss_type='mse'):
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown IoU loss type: {loss_type}")
    
    def compute_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """Compute actual IoU between predicted and ground truth masks"""
        # pred_mask: B, C, H, W (logits or probabilities)
        # gt_mask: B, 1, H, W or B, C, H, W
        
        if pred_mask.shape[1] > 1:  # Multi-class
            pred_binary = torch.argmax(pred_mask, dim=1, keepdim=True)
            if gt_mask.shape[1] == 1:
                # One-hot encode ground truth
                gt_onehot = F.one_hot(gt_mask.squeeze(1).long(), pred_mask.shape[1])
                gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()
            else:
                gt_onehot = gt_mask.float()
            
            # Compute IoU per class
            pred_onehot = F.one_hot(pred_binary.squeeze(1).long(), pred_mask.shape[1])
            pred_onehot = pred_onehot.permute(0, 3, 1, 2).float()
            
            intersection = (pred_onehot * gt_onehot).sum(dim=(2, 3))
            union = pred_onehot.sum(dim=(2, 3)) + gt_onehot.sum(dim=(2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            return iou  # B, C
        else:
            # Binary case
            pred_binary = (pred_mask > 0).float()
            intersection = (pred_binary * gt_mask.float()).sum(dim=(2, 3))
            union = pred_binary.sum(dim=(2, 3)) + gt_mask.float().sum(dim=(2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            return iou  # B, 1
    
    def forward(self, pred_iou: torch.Tensor, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_iou: Predicted IoU scores from model (B, C) or (B, 1, C)
            pred_mask: Predicted masks (B, C, H, W)
            gt_mask: Ground truth masks (B, 1, H, W)
        """
        actual_iou = self.compute_iou(pred_mask, gt_mask)
        
        # Reshape if needed
        if len(pred_iou.shape) == 3 and pred_iou.shape[1] == 1:
            pred_iou = pred_iou.squeeze(1)
        
        return self.loss_fn(pred_iou, actual_iou.detach())


# ============================================================================
# SCHEDULERS
# ============================================================================

class WarmupPolyLR(optim.lr_scheduler._LRScheduler):
    """Polynomial decay with warmup"""
    
    def __init__(self, optimizer, warmup_iters, total_iters, power=0.9, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            alpha = (self.total_iters - self.last_epoch) / (self.total_iters - self.warmup_iters)
            alpha = max(alpha, 0) ** self.power
            return [base_lr * alpha for base_lr in self.base_lrs]


class CosineAnnealingWarmup(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup"""
    
    def __init__(self, optimizer, warmup_iters, total_iters, eta_min=0, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            return [self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * progress)) / 2
                    for base_lr in self.base_lrs]


def create_scheduler(optimizer, scheduler_type, total_iters, warmup_iters=0, **kwargs):
    """Factory function for schedulers"""
    if scheduler_type == 'poly':
        return WarmupPolyLR(optimizer, warmup_iters, total_iters, **kwargs)
    elif scheduler_type == 'cosine':
        return CosineAnnealingWarmup(optimizer, warmup_iters, total_iters, **kwargs)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============================================================================
# OPTIMIZER WITH LAYER-WISE DECAY (for TinyViT)
# ============================================================================

def create_optimizer_with_layer_scales(
    model,
    optimizer_type: str = "adamw",
    base_lr: float = 1e-4,
    weight_decay: float = 0.1,
    non_encoder_lr_mult: float = 1.0,   # fallback for params without p.lr_scale (e.g., decoder)
    round_scale: int = 8,               # round lr_scale to merge float-equal groups
    no_decay_keywords = ("bias", "norm", "layernorm", "bn", "gn"),
    **kwargs
):
    """
    Build param groups using TinyViT's per-parameter lr_scale (when present),
    and collapse to unique (lr_scale, decay_flag) groups.

    - Encoder params (TinyViT): use p.lr_scale set by set_layer_lr_decay(...)
    - Non-encoder params: use non_encoder_lr_mult (default 1.0)
    - decay_flag: True for regular weights; False for biases & norm params
    - Scheduler will later scale all these group LRs by the same per-step factor
    """
    groups = {}  # key: (rounded_lr_scale, decay_flag) -> list of params

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 1) lr_scale: from TinyViT encoder if available, otherwise fallback
        lr_mult = getattr(p, "lr_scale", None)
        if lr_mult is None:
            lr_mult = non_encoder_lr_mult

        # stabilize float keys
        lr_mult_key = round(float(lr_mult), round_scale)

        # 2) decay_flag: no weight decay for biases and normalization params
        lname = name.lower()
        decay_flag = not any(k in lname for k in no_decay_keywords)

        key = (lr_mult_key, decay_flag)
        groups.setdefault(key, []).append(p)

    # Assemble param groups
    param_groups = []
    # sort for reproducibility: by decay_flag (no-decay last), then lr_mult
    for (lr_mult_key, decay_flag), params in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        param_groups.append({
            "params": params,
            "lr": base_lr * lr_mult_key,
            "weight_decay": weight_decay if decay_flag else 0.0,
        })

    # Create optimizer
    opt_type = optimizer_type.lower()
    if opt_type == "adamw":
        optimizer = optim.AdamW(param_groups, **kwargs)
    elif opt_type == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        optimizer = optim.SGD(param_groups, momentum=momentum, **kwargs)
    elif opt_type == "adam":
        optimizer = optim.Adam(param_groups, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


# ============================================================================
# TRAINING METRICS AND LOGGING
# ============================================================================

class MetricsTracker:
    """Track and log training metrics"""
    
    def __init__(self, writer: SummaryWriter, num_classes: int):
        self.writer = writer
        self.num_classes = num_classes
        self.reset_epoch()
        
        # Track best metrics
        self.best_val_dice = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def reset_epoch(self):
        """Reset metrics for new epoch"""
        self.train_losses = []
        self.val_losses = []
        self.train_dice = []
        self.val_dice = []
        self.iou_losses = []
        self.learning_rates = []
    
    def update_train(self, loss: float, dice: float, iou_loss: float = None, lr: float = None):
        """Update training metrics"""
        self.train_losses.append(loss)
        self.train_dice.append(dice)
        if iou_loss is not None:
            self.iou_losses.append(iou_loss)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def update_val(self, loss: float, dice: float):
        """Update validation metrics"""
        self.val_losses.append(loss)
        self.val_dice.append(dice)
    
    def log_epoch(self, epoch: int, additional_metrics: Dict = None):
        """Log epoch metrics to tensorboard"""
        # Average metrics
        avg_iou_loss   = np.mean(self.iou_losses) if self.iou_losses else float('nan')
        avg_train_loss = np.mean(self.train_losses) if self.train_losses else 0
        avg_val_loss = np.mean(self.val_losses) if self.val_losses else 0
        avg_train_dice = np.mean(self.train_dice) if self.train_dice else 0
        avg_val_dice = np.mean(self.val_dice) if self.val_dice else 0

    # ------------------------------------------------------------------------------------------------------------------

        # Log per-class dice if available
        if hasattr(self, '_val_per_class_dice'):
            for c, dice_list in enumerate(self._val_per_class_dice):
                if dice_list:
                    class_avg = np.mean(dice_list)
                    self.writer.add_scalar(f'DicePerClass/val_class_{c+1}', class_avg, epoch)
                    if additional_metrics is None:
                        additional_metrics = {}
                    additional_metrics[f'dice_class_{c+1}'] = class_avg
            # Reset for next epoch
            self._val_per_class_dice = [[] for _ in range(len(self._val_per_class_dice))]

        # Persist metrics to CSV
        from pathlib import Path
        csv_path = Path(self.writer.log_dir).parent / "metrics.csv"
        headers = ["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "iou_loss", "lr", "best_dice"]
        values = [epoch, avg_train_loss, avg_val_loss, avg_train_dice, avg_val_dice, avg_iou_loss,
                self.learning_rates[-1] if self.learning_rates else 0, self.best_val_dice]

        # Add per-class dice to CSV if available
        if additional_metrics:
            for k, v in additional_metrics.items():
                if 'dice_class' in k:
                    headers.append(k)
                    values.append(v)

        if not csv_path.exists():
            with open(csv_path, 'w') as f:
                f.write(','.join(map(str, headers)) + '\n')
        with open(csv_path, 'a') as f:
            f.write(','.join(map(str, values)) + '\n')

    # ------------------------------------------------------------------------------------------------------------------

        avg_iou_loss = np.mean(self.iou_losses) if self.iou_losses else 0
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
        self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
        self.writer.add_scalar('Dice/train', avg_train_dice, epoch)
        self.writer.add_scalar('Dice/val', avg_val_dice, epoch)
        
        if self.iou_losses:
            self.writer.add_scalar('Loss/iou', avg_iou_loss, epoch)
        
        if self.learning_rates:
            self.writer.add_scalar('Learning_Rate', self.learning_rates[-1], epoch)
        
        # Log additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        # Check for improvement
        if avg_val_dice > self.best_val_dice:
            self.best_val_dice = avg_val_dice
            self.best_val_loss = avg_val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            improved = True
        else:
            self.epochs_without_improvement += 1
            improved = False
        
        return {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': avg_train_dice,
            'val_dice': avg_val_dice,
            'iou_loss': avg_iou_loss,
            'best_dice': self.best_val_dice,
            'improved': improved
        }


# ============================================================================
# MAIN TRAINING CLASS
# ============================================================================

class SemanticSAMTrainer:
    """Professional training pipeline for Semantic SAM"""
    
    def __init__(self, args):
        self.args = args

        # Handle multi-GPU setup
        if self.args.if_split_encoder_gpus and len(self.args.devices) > 1:
            self.device = torch.device(self.args.devices[0])
            self.devices = [torch.device(d) for d in self.args.devices]
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.devices = [self.device]
                
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This training requires GPU.")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.build_model()
        
        # Setup training components
        self.setup_training()
        
        # Initialize metrics tracking
        self.metrics = MetricsTracker(self.writer, args.num_cls)
        
        # Time tracking
        self.start_time = time.time()
    
    def setup_directories(self):
        """Setup checkpoint and logging directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(self.args.dir_checkpoint) / timestamp
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save arguments
        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(vars(self.args), f, indent=4, default=str)
        
        # Setup tensorboard
        self.writer = SummaryWriter(str(self.checkpoint_dir / 'tensorboard'))
    
    def build_model(self):

        """Build and configure the model"""
        print(f"Building model: {self.args.arch}")
        
        # Check if using semantic variant
        use_semantic = (
             getattr(self.args, 'use_semantic', False) or 
            'semantic' in self.args.arch or
             self.args.num_cls > 3  # Semantic needed for multi-class beyond SAM's 3 masks
            )
        
        if use_semantic:
            # Use semantic builders
            semantic_arch = f"semantic_{self.args.arch}"
            if semantic_arch in sam_model_registry:
                sam = sam_model_registry[semantic_arch](
                    self.args,
                    checkpoint=self.args.sam_ckpt,
                    num_classes=self.args.num_cls
                )
            else:
                raise ValueError(f"Semantic variant not found for {self.args.arch}")
        else:
            # Load base SAM model
            sam = sam_model_registry[self.args.arch](
                self.args,
                checkpoint=self.args.sam_ckpt,
                num_classes=self.args.num_cls
        )
        # Apply PEFT if specified
        if self.args.finetune_type == 'peft':
            print(f"Applying PEFT method: {self.args.peft_method}")
            sam = create_peft_sam(
                sam,
                method=self.args.peft_method,
                rank=self.args.peft_rank,
                alpha=self.args.peft_alpha,
                dropout=self.args.peft_dropout,
                encoder_layers=self.args.encoder_layers,
                decoder_layers=self.args.decoder_layers,
                enable_adapters=self.args.enable_adapters
                  )
            if self.args.module_patterns is not None:
                sam.set_trainable_modules(self.args.module_patterns)
         

        elif self.args.finetune_type == 'adapter':
            # Freeze non-adapter parameters
            for n, p in sam.named_parameters():
                if "Adapter" not in n:
                    p.requires_grad = False
        elif self.args.finetune_type == 'vanilla':
            if not self.args.if_update_encoder:
                enc = getattr(sam, "image_encoder", None)
                if enc is None and hasattr(sam, "img_adapter") and hasattr(sam.img_adapter, "sam_img_encoder"):
                    enc = sam.img_adapter.sam_img_encoder  # SemanticSam path
                if enc is None:
                    raise AttributeError("Could not locate encoder on this model for freezing.")
                for p in enc.parameters():
                    p.requires_grad = False
        
      # Handle multi-GPU setup
        if not self.args.if_split_encoder_gpus:
            sam = sam.to(self.device)
        # If split_encoder_gpus is True, the model handles its own device placement
        
        # Count parameters
        total_params = sum(p.numel() for p in sam.parameters())
        trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        return sam
    
    def setup_training(self):

        if not hasattr(self.args, 'ignore_index'):
             self.args.ignore_index = 255

        """Setup optimizer, scheduler, and loss functions"""
        # Don't estimate, will be set properly when dataloaders are created
        self.total_iters = None  # Will be set in train()

        # Setup optimizer with layer-wise decay for TinyViT

        self.optimizer = create_optimizer_with_layer_scales(
            self.model,
            optimizer_type=self.args.optimizer,
            base_lr=self.args.lr,
            weight_decay=self.args.weight_decay,
         )
        
        # Setup scheduler
        self.scheduler = None
        
        
        # Setup loss functions
        loss_configs = {
            'dice': {
                'weight': self.args.dice_weight,
                'from_logits': True,
                'include_background': False,  # Exclude background for better multi-class performance
                'ignore_index': getattr(self.args, 'ignore_index', None),
                'average': 'macro',
                'eps': 1e-6,
            },
            'ce': {'weight': self.args.ce_weight,
                   'ignore_index': getattr(self.args, 'ignore_index', None)
             }
        }

        # Add additional losses if specified
        if hasattr(self.args, 'use_focal') and self.args.use_focal:
            loss_configs['focal'] = {'weight': self.args.focal_weight, 'gamma': 2.0}

        self.criterion = CombinedLoss(loss_configs, num_classes=self.args.num_cls)
                
        self.iou_criterion = IoULoss(loss_type='mse')
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.args.use_amp else None
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for batch_idx, data in enumerate(pbar):
            # Move data to GPU
            imgs = data['image'].to(self.device)
           #masks = torchvision.transforms.Resize((self.args.out_size, self.args.out_size))(data['mask'])
           #masks = masks.to(self.device)

           #############################################################################new

            m = data['mask']                    # (1,1,H,W) from our collate
            m = F.interpolate(
                    m.float(), 
                    size=(self.args.out_size, self.args.out_size),
                    mode='nearest'
                ).long() 
            
            if hasattr(self.args, 'ignore_index') and self.args.ignore_index is not None:
                bad = ((m < 0) | ((m >= self.args.num_cls) & (m != self.args.ignore_index)))
                if bad.any():
                    # map truly bad ids to ignore; keeps training alive even if a dataset slips through
                    m[bad] = self.args.ignore_index

            masks = m.to(self.device)

           ###############################################################################new

            
            # Get prompts (handle both points and boxes)
            prompts = self.prepare_prompts(data)

            ############################################################################new

            if prompts['points'] is not None:
                P = prompts['points'][0].shape[0]  # (P, N, 2)
            elif prompts['boxes'] is not None and prompts['boxes'] is not None:
                P = prompts['boxes'].shape[0]      # (P, 4)
            else:
                P = 1

            # masks: (1,1,H,W) from collate; match it to (P,1,H,W)
            if masks.dim() == 3:      # very defensive; if someone bypassed collate
                masks = masks.unsqueeze(0)         # (1,1,H,W)
            if masks.size(0) != P:
                masks = masks.expand(P, -1, -1, -1)

            ############################################################################new
                        
            # Forward pass with mixed precision
            with autocast(enabled=self.args.use_amp):

                 # Check for PEFT wrapper first
                actual_model = self.model.sam if hasattr(self.model, 'sam') else self.model
    
            # Handle both regular SAM and SemanticSam
                if hasattr(actual_model, 'img_adapter'):  # SemanticSam
                    # Encode image
                    if self.args.if_update_encoder:
                        img_emb = actual_model.img_adapter(imgs)
                    else:
                        with torch.no_grad():
                            img_emb = actual_model.img_adapter(imgs)

                    # Get embeddings from prompts
                    sparse_emb, dense_emb = actual_model.prompt_adapter(**prompts)
                    
                    # Decode masks
                    pred_masks, pred_iou = actual_model.mask_adapter(
                    image_embeddings=img_emb,
                    prompt_adapter=actual_model.prompt_adapter,
                    sparse_embeddings=sparse_emb,
                    dense_embeddings=dense_emb,
                    multimask_output=(self.args.num_cls > 1)
                )
                    
                else:  # Regular SAM
                    if self.args.if_update_encoder:
                        img_emb = actual_model.image_encoder(imgs)
                    else:
                        with torch.no_grad():
                            img_emb = actual_model.image_encoder(imgs)

                    # Get embeddings from prompts
                    sparse_emb, dense_emb = actual_model.prompt_encoder(**prompts)

                    # Decode masks
                    pred_masks, pred_iou = actual_model.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=actual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=(self.args.num_cls > 1)
                    )
                
                # Compute losses
                mask_loss, loss_dict = self.criterion(pred_masks, masks)
                
                # IoU loss if training IoU head
                if self.args.train_iou:

            ####################################################################################new
                    gt_for_iou = masks
                    if hasattr(self.args, 'ignore_index') and self.args.ignore_index is not None:
                        gt_for_iou = gt_for_iou.clone()
                        gt_for_iou[gt_for_iou == self.args.ignore_index] = 0  # neutralize ignore for IoU

                    iou_loss = self.iou_criterion(pred_iou, pred_masks, gt_for_iou)

            ####################################################################################new

                   #iou_loss = self.iou_criterion(pred_iou, pred_masks, masks)
                    total_loss = mask_loss + self.args.iou_weight * iou_loss
                else:
                    iou_loss = torch.tensor(0.0, device=self.device)
                    total_loss = mask_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                
                # Update Delta-LoRA before optimizer step
                if hasattr(self.model, 'config') and self.model.config.method == PEFTMethod.DELTA_LORA:
                    self.model.update_delta_lora()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # Update Delta-LoRA before optimizer step
                if hasattr(self.model, 'config') and self.model.config.method == PEFTMethod.DELTA_LORA:
                    self.model.update_delta_lora()
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # Compute metrics
            with torch.no_grad():
                dice_score = dice_coeff_multi_class(
                    pred_masks.argmax(dim=1).cpu(),
                    masks.squeeze(1).cpu().long(),
                    self.args.num_cls
                )
            
            # Update metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics.update_train(
                mask_loss.item(),
                dice_score,
                iou_loss.item() if self.args.train_iou else None,
                current_lr
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{mask_loss.item():.4f}',
                'dice': f'{dice_score:.4f}',
                'iou_loss': f'{iou_loss.item():.4f}' if self.args.train_iou else 'N/A',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log to tensorboard every N iterations
            if batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Iteration/loss', mask_loss.item(), global_step)
                self.writer.add_scalar('Iteration/dice', dice_score, global_step)
                if self.args.train_iou:
                    self.writer.add_scalar('Iteration/iou_loss', iou_loss.item(), global_step)
    
    def validate_epoch(self, epoch: int):
        """Validate for one epoch"""
        self.model.eval()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
        
        with torch.no_grad():
            for data in pbar:
                # Move data to GPU
                imgs = data['image'].to(self.device)
                #masks = torchvision.transforms.Resize((self.args.out_size, self.args.out_size))(data['mask'])
                #masks = masks.to(self.device)

                ############################################################################new

                m = data['mask']                    # (1,1,H,W) from our collate
                m = F.interpolate(
                        m.float(), 
                        size=(self.args.out_size, self.args.out_size),
                        mode='nearest'
                    ).long()
                if hasattr(self.args, 'ignore_index') and self.args.ignore_index is not None:
                    bad = ((m < 0) | ((m >= self.args.num_cls) & (m != self.args.ignore_index)))
                    if bad.any():
                        # map truly bad ids to ignore; keeps training alive even if a dataset slips through
                        m[bad] = self.args.ignore_index

                masks = m.to(self.device)

                ############################################################################new
                
                # Get prompts
                prompts = self.prepare_prompts(data)

                ############################################################################new
           
                # Figure out prompt batch size P exactly like PromptEncoder._get_batch_size
                if prompts['points'] is not None:
                    P = prompts['points'][0].shape[0]  # (P, N, 2)
                elif prompts['boxes'] is not None and prompts['boxes'] is not None:
                    P = prompts['boxes'].shape[0]      # (P, 4)
                else:
                    P = 1

                # masks: (1,1,H,W) from collate; match it to (P,1,H,W)
                if masks.dim() == 3:      # very defensive; if someone bypassed collate
                    masks = masks.unsqueeze(0)         # (1,1,H,W)
                if masks.size(0) != P:
                    masks = masks.expand(P, -1, -1, -1)   # view, not copy

                ############################################################################new

                
                # Forward pass
                actual_model = self.model.sam if hasattr(self.model, 'sam') else self.model

                if hasattr(actual_model, 'img_adapter'):  # SemanticSam
                    img_emb = actual_model.img_adapter(imgs)
                    sparse_emb, dense_emb = actual_model.prompt_adapter(**prompts)

                    pred_masks, pred_iou = actual_model.mask_adapter(
                        image_embeddings=img_emb,
                        prompt_adapter=actual_model.prompt_adapter,
                        sparse_embeddings=sparse_emb,
                        dense_embeddings=dense_emb,
                        multimask_output=(self.args.num_cls > 1)
                    )
                else:  # Regular SAM
                    img_emb = actual_model.image_encoder(imgs)
                    sparse_emb, dense_emb = actual_model.prompt_encoder(**prompts)
                    
                    pred_masks, pred_iou = actual_model.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=actual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=(self.args.num_cls > 1)
                    )
            
                    

                # Compute loss
                val_loss, _ = self.criterion(pred_masks, masks)
                
                # Compute metrics
                dice_score = dice_coeff_multi_class(
                    pred_masks.argmax(dim=1).cpu(),
                    masks.squeeze(1).cpu().long(),
                    self.args.num_cls
                )

               # Compute and log per-class dice
                per_cls_dice = dice_coeff_multi_class(
                    pred_masks.argmax(dim=1).cpu(),
                    masks.squeeze(1).cpu().long(),
                    self.args.num_cls,
                    from_logits=False,
                    include_background=False,  # Skip background class
                    average='none'
                )
                # Store for epoch aggregation (create this list at method start)
                if not hasattr(self, '_val_per_class_dice'):
                    self.metrics._val_per_class_dice = [[] for _ in range(self.args.num_cls - 1)]  # Exclude background
                for c, v in enumerate(per_cls_dice.tolist()):
                    if not torch.isnan(torch.tensor(v)):
                        self.metrics._val_per_class_dice[c].append(v)

                 # Update metrics
                self.metrics.update_val(val_loss.item(), dice_score)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{val_loss.item():.4f}',
                    'dice': f'{dice_score:.4f}'
                })
    
    def prepare_prompts(self, data: Dict) -> Dict:
        """Prepare prompts for SAM encoder"""
        prompts = {}
        
        if 'points' in data and data['points'] is not None:
            points, labels = data['points']
            prompts['points'] = (
                points.to(self.device),
                labels.to(self.device)
            )
        else:
            prompts['points'] = None
        
        if 'boxes' in data and data['boxes'] is not None:
            prompts['boxes'] = data['boxes'].to(self.device)
        else:
            prompts['boxes'] = None
        
        prompts['masks'] = None
        
        return prompts
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_dice': self.metrics.best_val_dice,
            'args': vars(self.args)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with dice: {self.metrics.best_val_dice:.4f}")
        
        # Remove old checkpoints to save space (keep only last 3 + best)
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
                

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Handle PEFT model state dict
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            # Try loading to the underlying SAM model if PEFT wrapped
            if hasattr(self.model, 'sam'):
                # Extract SAM-only weights
                sam_state = {k: v for k, v in checkpoint['model_state_dict'].items() 
                            if not k.startswith('peft_')}
                self.model.sam.load_state_dict(sam_state, strict=False)
            else:
                raise e


        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.args.epochs} epochs")
        print(f"Training on device: {self.device}")
        
        # Create data loaders ---- for now we use img_list as both train and val lists
        train_dataset = Public_dataset(
            self.args,
            img_list=self.args.img_list,
            phase='train',
            sample_num=self.args.sample_num,
            prompt_type=self.args.prompt_type,
            targets=self.args.targets,
            cls=self.args.cls if hasattr(self.args, 'cls') else -1,
            label_mapping=self.args.label_mapping if hasattr(self.args, 'label_mapping') else None
        )
        
        val_dataset = Public_dataset(
            self.args,
            img_list=self.args.img_list,
            phase='val',
            sample_num=self.args.sample_num,
            prompt_type=self.args.prompt_type,
            targets=self.args.targets,
            cls=self.args.cls if hasattr(self.args, 'cls') else -1,
            label_mapping=self.args.label_mapping if hasattr(self.args, 'label_mapping') else None
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=sam_collate_single
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=sam_collate_single
        )

        # After creating train_loader
        self.total_iters = self.args.epochs * len(self.train_loader)

        # Now create scheduler with correct total_iters
        warmup_iters = int(self.args.warmup_ratio * self.total_iters) if hasattr(self.args, 'warmup_ratio') else 0
        self.scheduler = create_scheduler(
            self.optimizer,
            self.args.scheduler_type,
            self.total_iters,
            warmup_iters=warmup_iters
        )

        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            # Calculate time remaining
            elapsed = time.time() - self.start_time
            epochs_done = epoch - 1
            if epochs_done > 0:
                time_per_epoch = elapsed / epochs_done
                remaining_epochs = self.args.epochs - epochs_done
                eta = timedelta(seconds=int(time_per_epoch * remaining_epochs))
                print(f"\nEstimated time remaining: {eta}")
            
            # Train and validate
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            
            # Log metrics and check for improvement
            epoch_metrics = self.metrics.log_epoch(epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}, Train Dice: {epoch_metrics['train_dice']:.4f}")
            print(f"  Val Loss: {epoch_metrics['val_loss']:.4f}, Val Dice: {epoch_metrics['val_dice']:.4f}")
            print(f"  Best Val Dice: {epoch_metrics['best_dice']:.4f} (Epoch {self.metrics.best_epoch})")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=epoch_metrics['improved'])

            # Early stopping
            if hasattr(self.args, 'early_stopping_patience'):
                if self.metrics.epochs_without_improvement >= self.args.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {self.args.early_stopping_patience} epochs without improvement")
                    break
            
            # Reset epoch metrics
            self.metrics.reset_epoch()
        
        # Training complete
        total_time = timedelta(seconds=int(time.time() - self.start_time))
        print(f"\nTraining complete! Total time: {total_time}")
        print(f"Best validation dice: {self.metrics.best_val_dice:.4f} at epoch {self.metrics.best_epoch}")
        
        # Close tensorboard writer
        self.writer.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for training"""
    #import argparse
    
    # Create argument parser
    args = cfg.parse_args()
    
    # Model arguments
    # parser.add_argument('--arch', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h', 'vit_t'],
    #                     help='Model architecture')
    # parser.add_argument('--sam_ckpt', type=str, required=True,
    #                     help='Path to SAM checkpoint')
    # parser.add_argument('--num_cls', type=int, default=2,
    #                     help='Number of classes for segmentation')
    # parser.add_argument('--finetune_type', type=str, default='peft',
    #                     choices=['vanilla', 'adapter', 'peft'],
    #                     help='Fine-tuning strategy')
    
    # # PEFT arguments
    # parser.add_argument('--peft_method', type=str, default='lora',
    #                     choices=['lora', 'lora_fa', 'vera', 'delta_lora'],
    #                     help='PEFT method to use')
    # parser.add_argument('--peft_rank', type=int, default=4,
    #                     help='Rank for PEFT methods')
    # parser.add_argument('--peft_alpha', type=float, default=1.0,
    #                     help='Alpha scaling factor for PEFT')
    # parser.add_argument('--peft_dropout', type=float, default=0.0,
    #                     help='Dropout for PEFT layers')
    # parser.add_argument('--encoder_layers', type=int, nargs='*', default=None,
    #                     help='Which encoder layers to apply PEFT')
    # parser.add_argument('--decoder_layers', type=int, nargs='*', default=None,
    #                     help='Which decoder layers to apply PEFT')
    # parser.add_argument('--enable_adapters', action='store_true',
    #                     help='Enable adapter layers if present')
    
    # # Training arguments
    # parser.add_argument('--epochs', type=int, default=100,
    #                     help='Number of training epochs')
    # parser.add_argument('--batch_size', type=int, default=4,
    #                     help='Batch size')
    # parser.add_argument('--lr', type=float, default=1e-4,
    #                     help='Learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0.01,
    #                     help='Weight decay')
    # parser.add_argument('--optimizer', type=str, default='adamw',
    #                     choices=['adamw', 'sgd', 'adam'],
    #                     help='Optimizer type')
    # parser.add_argument('--scheduler_type', type=str, default='poly',
    #                     choices=['poly', 'cosine', 'step', 'exponential', 'plateau'],
    #                     help='Learning rate scheduler type')
    # parser.add_argument('--warmup_ratio', type=float, default=0.1,
    #                     help='Warmup ratio for scheduler')
    
    # # Loss arguments
    # parser.add_argument('--dice_weight', type=float, default=0.5,
    #                     help='Weight for Dice loss')
    # parser.add_argument('--ce_weight', type=float, default=0.5,
    #                     help='Weight for Cross Entropy loss')
    # parser.add_argument('--use_focal', action='store_true',
    #                     help='Use focal loss')
    # parser.add_argument('--focal_weight', type=float, default=0.2,
    #                     help='Weight for Focal loss')
    # parser.add_argument('--train_iou', action='store_true',
    #                     help='Train IoU prediction head')
    # parser.add_argument('--iou_weight', type=float, default=0.1,
    #                     help='Weight for IoU loss')
    
    # # Data arguments
    # parser.add_argument('--train_list', type=str, required=True,
    #                     help='Path to training data list')
    # parser.add_argument('--val_list', type=str, required=True,
    #                     help='Path to validation data list')
    # parser.add_argument('--image_size', type=int, default=1024,
    #                     help='Input image size')
    # parser.add_argument('--out_size', type=int, default=256,
    #                     help='Output mask size')
    # parser.add_argument('--num_workers', type=int, default=4,
    #                     help='Number of data loading workers')
    # parser.add_argument('--sample_num', type=int, default=50,
    #                     help='Number of prompt samples')
    # parser.add_argument('--prompt_type', type=str, default='point',
    #                     choices=['point', 'box', 'hybrid'],
    #                     help='Type of prompts to use')
    # parser.add_argument('--targets', type=str, default='multi_all',
    #                     help='Target classes')
    # parser.add_argument('--cls', type=int, default=-1,
    #                     help='Specific class to train on')
    # parser.add_argument('--label_mapping', type=str, default=None,
    #                     help='Path to label mapping file')
    
    # # Other arguments
    # parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints',
    #                     help='Directory for saving checkpoints')
    # parser.add_argument('--if_update_encoder', action='store_true',
    #                     help='Whether to update encoder')
    # parser.add_argument('--use_amp', action='store_true',
    #                     help='Use automatic mixed precision')
    # parser.add_argument('--early_stopping_patience', type=int, default=20,
    #                     help='Early stopping patience')
    # parser.add_argument('--layer_decay', type=float, default=0.9,
    #                     help='Layer-wise learning rate decay for TinyViT')
    # parser.add_argument('--if_encoder_adapter', action='store_true',
    #                     help='Use encoder adapters')
    # parser.add_argument('--encoder_adapter_depths', type=int, nargs='*', default=[],
    #                     help='Depths for encoder adapters')
    # parser.add_argument('--if_mask_decoder_adapter', action='store_true',
    #                     help='Use mask decoder adapters')
    # parser.add_argument('--decoder_adapt_depth', type=int, default=2,
    #                     help='Decoder adapter depth')
    # parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
    #                     help='Devices to use')
    # parser.add_argument('--if_split_encoder_gpus', action='store_true',
    #                     help='Split encoder across GPUs')
    # parser.add_argument('--gpu_fractions', type=float, nargs='+', default=[0.5, 0.5],
    #                     help='GPU memory fractions')
    # parser.add_argument('--use_semantic', action='store_true',
    #                     help='Use SemanticSam variant')

        
    # Parse arguments
    # args = parser.parse_args()
    
    # Create trainer and start training
    # Set seed for reproducibility
    def set_seed(seed: int):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not hasattr(args, 'seed'):
        args.seed = 1337
    set_seed(args.seed)
    print(f"Using seed: {args.seed}")

    trainer = SemanticSAMTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

