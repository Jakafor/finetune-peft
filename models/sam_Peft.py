"""
Enhanced SAM PEFT Module with TinyViT Support
Supports LoRA, LoRA-FA, VeRA, and Delta-LoRA for both standard SAM and Mobile-SAM (TinyViT)

Reading Guide:
1. Start with PEFTMethod and PEFTConfig classes - these define configuration
2. Review the base PEFT layer classes (BasePEFTLayer, LoRALayer, etc.) - these implement different PEFT methods
3. Look at MultiQueryPEFT class - handles attention QKV projections
4. Main class is EnhancedSAMPEFT - orchestrates everything
5. The key TinyViT handling is in _apply_encoder_peft() method
6. Use create_peft_sam() function as the main entry point
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
#from segment_anything.modeling import Sam
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class PEFTMethod(Enum):
    """Enum for different PEFT methods"""
    LORA = "lora"
    LORA_FA = "lora_fa"  # Frozen A matrix, trainable B
    VERA = "vera"  # Frozen random A,B with trainable scaling vectors
    DELTA_LORA = "delta_lora"  # Incremental LoRA updates to base weights


@dataclass
class PEFTConfig:
    """Configuration for PEFT methods"""
    method: PEFTMethod = PEFTMethod.LORA
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    encoder_layers: List[int] = None  # None means all layers
    decoder_layers: List[int] = None  # None means all layers
    enable_adapters: bool = False  # Enable existing adapter modules
    adapter_modules: List[str] = None  # Specific adapter patterns to enable


# ============================================================================
# WEIGHT CONTROL
# ============================================================================

class WeightController:
    """Controls which weights are trainable in the model"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_requires_grad = {}
        self._save_original_state()
    
    def _save_original_state(self):
        """Save the original requires_grad state of all parameters"""
        for name, param in self.model.named_parameters():
            self.original_requires_grad[name] = param.requires_grad
    
    def freeze_all(self):
        """Freeze all parameters in the model"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters in the model"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def set_trainable(self, module_patterns: List[str], trainable: bool = True):
        """Set specific modules as trainable/frozen based on name patterns"""
        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in module_patterns):
                param.requires_grad = trainable
    
    def enable_adapters(self, adapter_patterns: List[str] = None):
        """Enable adapter layers if they exist in the model"""
        if adapter_patterns is None:
            # Default adapter patterns from your TinyViT code
            adapter_patterns = ['Adapter', 'adapter', 'MLP_Adapter', 
                              'Space_Adapter', 'Depth_Adapter']
        
        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in adapter_patterns):
                param.requires_grad = True
                print(f"Enabled adapter: {name}")


# ============================================================================
# BASE PEFT LAYERS
# ============================================================================

class BasePEFTLayer(nn.Module):
    """Base class for all PEFT layer implementations"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, 
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()


class LoRALayer(BasePEFTLayer):
    """Standard LoRA layer: trainable A and B matrices"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, **kwargs):
        super().__init__(in_features, out_features, rank, **kwargs)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize A with Kaiming uniform, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard LoRA forward: x -> dropout -> A -> B -> scale
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAFALayer(BasePEFTLayer):
    """LoRA-FA: Frozen A matrix, trainable B matrix"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, **kwargs):
        super().__init__(in_features, out_features, rank, **kwargs)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()
        # The key difference: freeze A matrix
        self.lora_A.weight.requires_grad = False
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class VeRALayer(BasePEFTLayer):
    """VeRA: Frozen random A,B matrices with trainable scaling vectors"""
    
    # Shared frozen matrices across all VeRA layers (memory efficient)
    _shared_A: Dict[Tuple[int, int], torch.Tensor] = {}
    _shared_B: Dict[Tuple[int, int], torch.Tensor] = {}
    
    def __init__(self, in_features: int, out_features: int, rank: int, **kwargs):
        super().__init__(in_features, out_features, rank, **kwargs)
        
        # Create or reuse shared frozen matrices
        key_A = (in_features, rank)
        key_B = (rank, out_features)
        
        if key_A not in VeRALayer._shared_A:
            VeRALayer._shared_A[key_A] = nn.Parameter(
                torch.randn(in_features, rank) / math.sqrt(rank),
                requires_grad=False
            )
        if key_B not in VeRALayer._shared_B:
            VeRALayer._shared_B[key_B] = nn.Parameter(
                torch.randn(rank, out_features) / math.sqrt(rank),
                requires_grad=False
            )
        
        self.register_buffer('vera_A', VeRALayer._shared_A[key_A])
        self.register_buffer('vera_B', VeRALayer._shared_B[key_B])
        
        # Trainable scaling vectors (the only trainable parameters)
        self.scale_a = nn.Parameter(torch.ones(rank))
        self.scale_b = nn.Parameter(torch.ones(rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.linear(x, self.vera_A.T)  # Project to rank dimension
        x = x * self.scale_a * self.scale_b  # Apply trainable scaling
        x = F.linear(x, self.vera_B.T)  # Project to output dimension
        return x * self.scaling


class DeltaLoRALayer(BasePEFTLayer):
    """Delta-LoRA: Directly updates base weights with LoRA deltas"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, 
                 base_weight: torch.Tensor = None, **kwargs):
        super().__init__(in_features, out_features, rank, **kwargs)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Store reference to base weight for incremental updates
        self.base_weight = base_weight
        self.prev_delta = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def compute_delta(self) -> torch.Tensor:
        """Compute the current LoRA delta (B @ A)"""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling
    
    def update_base_weight(self):
        """Update base weight with the difference from previous delta"""
        if self.base_weight is not None:
            current_delta = self.compute_delta()
            if self.prev_delta is not None:
                # Add the change in delta to base weights
                delta_diff = current_delta - self.prev_delta
                self.base_weight.data += delta_diff
            self.prev_delta = current_delta.detach().clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard LoRA forward (base weight updated separately)
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


# ============================================================================
# WRAPPER AND MULTI-QUERY CLASSES
# ============================================================================

class PEFTWrapper(nn.Module):
    """Wraps a linear layer with PEFT functionality"""
    
    def __init__(self, base_layer: nn.Module, peft_layer: BasePEFTLayer):
        super().__init__()
        self.base_layer = base_layer
        self.peft_layer = peft_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combine base layer output with PEFT delta
        base_out = self.base_layer(x)
        peft_out = self.peft_layer(x)
        return base_out + peft_out


class MultiQueryPEFT(nn.Module):
    """
    PEFT for multi-query attention (q, k, v)
    Handles the special case where QKV are computed in a single linear layer
    """
    
    def __init__(self, qkv_layer: nn.Module, dim: int, rank: int, 
                 method: PEFTMethod, **kwargs):
        super().__init__()
        self.qkv = qkv_layer
        self.dim = dim
        
        # Create appropriate PEFT layers based on method
        layer_class = self._get_layer_class(method)
        
        # We only add LoRA to Q and V (common practice, K is often skipped)
        self.q_peft = layer_class(dim, dim, rank, **kwargs)
        self.v_peft = layer_class(dim, dim, rank, **kwargs)
        
        if method == PEFTMethod.DELTA_LORA:
            # Pass reference to QKV weight slices for delta updates
            self.q_peft.base_weight = qkv_layer.weight[:dim, :]
            self.v_peft.base_weight = qkv_layer.weight[-dim:, :]
    
    def _get_layer_class(self, method: PEFTMethod):
        """Get the appropriate PEFT layer class for the method"""
        return {
            PEFTMethod.LORA: LoRALayer,
            PEFTMethod.LORA_FA: LoRAFALayer,
            PEFTMethod.VERA: VeRALayer,
            PEFTMethod.DELTA_LORA: DeltaLoRALayer
        }[method]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute base QKV
        qkv = self.qkv(x)
        
        # Add PEFT updates to Q and V
        new_q = self.q_peft(x)
        new_v = self.v_peft(x)
        
        # Handle different tensor shapes
        # Standard ViT shape: B, H, W, 3*C (image patches)
        # or B, N, 3*C (sequence format)
        if len(qkv.shape) == 4:  # B, H, W, 3*C
            qkv[:, :, :, :self.dim] += new_q
            qkv[:, :, :, -self.dim:] += new_v
        else:  # B, N, 3*C
            qkv[:, :, :self.dim] += new_q
            qkv[:, :, -self.dim:] += new_v
        
        return qkv


# ============================================================================
# MAIN PEFT CLASS
# ============================================================================

class EnhancedSAMPEFT(nn.Module):
    """
    Enhanced SAM with multiple PEFT methods
    Supports both standard SAM (ViT) and Mobile-SAM (TinyViT)
    """
    
    def __init__(self, sam_model, config: PEFTConfig):
        super().__init__()
        self.sam = sam_model
        self.config = config
        self.weight_controller = WeightController(sam_model)
        
        # Freeze base model initially
        self.weight_controller.freeze_all()
        
        # Store PEFT layers for potential updates (e.g., Delta-LoRA)
        self.peft_layers = []
        
        # Detect architecture type
        self.is_tiny_vit = self._detect_tiny_vit()
        
        # Apply PEFT to encoder
        if config.encoder_layers is not None or config.encoder_layers != []:
            self._apply_encoder_peft()
        
        # Apply PEFT to decoder (if it exists and is requested)
        if hasattr(sam_model, 'mask_decoder'):
            if config.decoder_layers is not None or len(config.decoder_layers) > 0:
                self._apply_decoder_peft()
        
        # Enable adapters if requested
        if config.enable_adapters:
            self.weight_controller.enable_adapters(config.adapter_modules)
    
    def _detect_tiny_vit(self) -> bool:
        """
        Detect if the model is using TinyViT architecture
        TinyViT has a different structure with layers containing blocks
        """
        try:
            # Check if the encoder has 'layers' attribute (TinyViT structure)
            if hasattr(self.sam.image_encoder, 'layers'):
                # Further check: TinyViT layers have blocks inside
                if len(self.sam.image_encoder.layers) > 0:
                    first_layer = self.sam.image_encoder.layers[0]
                    # ConvLayer doesn't have attention, skip to next layers
                    if len(self.sam.image_encoder.layers) > 1:
                        second_layer = self.sam.image_encoder.layers[1]
                        if hasattr(second_layer, 'blocks'):
                            return True
            return False
        except:
            return False
    
    def _apply_encoder_peft(self):
        """Apply PEFT to encoder blocks (handles both ViT and TinyViT)"""
        
        if self.is_tiny_vit:
            print("Detected TinyViT architecture (Mobile-SAM)")
            self._apply_tiny_vit_peft()
        else:
            print("Detected standard ViT architecture")
            self._apply_standard_vit_peft()
    
    def _apply_standard_vit_peft(self):
        """Apply PEFT to standard ViT encoder"""
        
        # Standard SAM has blocks directly in image_encoder
        blocks = self.sam.image_encoder.blocks
        
        for idx, block in enumerate(blocks):
            # Check if this layer should have PEFT
            if self.config.encoder_layers is None or idx in self.config.encoder_layers:
                # Replace attention QKV with PEFT version
                old_qkv = block.attn.qkv
                block.attn.qkv = MultiQueryPEFT(
                    old_qkv,
                    old_qkv.in_features,
                    self.config.rank,
                    self.config.method,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout
                )
                self.peft_layers.append(block.attn.qkv)
                print(f"Applied PEFT to ViT encoder block {idx}")
    
    def _apply_tiny_vit_peft(self):
        """Apply PEFT to TinyViT encoder (Mobile-SAM)"""
        
        # First, make the conv layers trainable (as in original code)
        for n, value in self.sam.image_encoder.layers[0].named_parameters():
            value.requires_grad = True
            print(f"Unfroze conv layer parameter: {n}")
        
        # Apply PEFT to attention blocks in layers[1:] 
        # (layer[0] is ConvLayer, doesn't have attention)
        for layer_idx, layer in enumerate(self.sam.image_encoder.layers[1:]):
            # Check if this layer should have PEFT
            if self.config.encoder_layers is not None and layer_idx not in self.config.encoder_layers:
                continue
            
            # TinyViT has blocks inside each layer
            if hasattr(layer, 'blocks'):
                for block_idx, block in enumerate(layer.blocks):
                    # TinyViT block attention structure
                    if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                        old_qkv = block.attn.qkv
                        
                        # Replace with PEFT version
                        block.attn.qkv = MultiQueryPEFT(
                            old_qkv,
                            old_qkv.in_features,
                            self.config.rank,
                            self.config.method,
                            alpha=self.config.alpha,
                            dropout=self.config.dropout
                        )
                        self.peft_layers.append(block.attn.qkv)
                        print(f"Applied PEFT to TinyViT layer {layer_idx+1}, block {block_idx}")
    
    def _apply_decoder_peft(self):
        """Apply PEFT to decoder transformer"""
        
        layer_class = self._get_layer_class()
        
        # Apply to transformer layers
        if hasattr(self.sam.mask_decoder, 'transformer'):
            transformer = self.sam.mask_decoder.transformer
            
            # Apply to each transformer layer
            for idx, layer in enumerate(transformer.layers):
                if self.config.decoder_layers is not None and idx not in self.config.decoder_layers:
                    continue
                
                # Self attention
                self._wrap_attention_projections(layer.self_attn, layer_class, f"decoder_layer_{idx}_self")
                
                # Cross attention token to image
                if hasattr(layer, 'cross_attn_token_to_image'):
                    self._wrap_attention_projections(
                        layer.cross_attn_token_to_image, 
                        layer_class, 
                        f"decoder_layer_{idx}_token_to_image"
                    )
                
                # Cross attention image to token
                if hasattr(layer, 'cross_attn_image_to_token'):
                    self._wrap_attention_projections(
                        layer.cross_attn_image_to_token, 
                        layer_class, 
                        f"decoder_layer_{idx}_image_to_token"
                    )
            
            # Final attention
            if hasattr(transformer, 'final_attn_token_to_image'):
                self._wrap_attention_projections(
                    transformer.final_attn_token_to_image,
                    layer_class,
                    "decoder_final_attn"
                )
    
    def _wrap_attention_projections(self, attn_module: nn.Module, layer_class, name_prefix: str):
        """Wrap Q and V projections with PEFT layers"""
        
        for proj_name in ['q_proj', 'v_proj']:
            if hasattr(attn_module, proj_name):
                base_layer = getattr(attn_module, proj_name)
                
                # Create PEFT layer
                peft_layer = layer_class(
                    base_layer.in_features,
                    base_layer.out_features,
                    self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout
                )
                
                if self.config.method == PEFTMethod.DELTA_LORA:
                    peft_layer.base_weight = base_layer.weight
                
                # Wrap the projection
                wrapped = PEFTWrapper(base_layer, peft_layer)
                setattr(attn_module, proj_name, wrapped)
                self.peft_layers.append(wrapped)
                print(f"Applied PEFT to {name_prefix}_{proj_name}")
    
    def _get_layer_class(self):
        """Get the appropriate PEFT layer class for the configured method"""
        return {
            PEFTMethod.LORA: LoRALayer,
            PEFTMethod.LORA_FA: LoRAFALayer,
            PEFTMethod.VERA: VeRALayer,
            PEFTMethod.DELTA_LORA: DeltaLoRALayer
        }[self.config.method]
    
    def update_delta_lora(self):
        """
        Update base weights for Delta-LoRA
        Call this after each training step if using Delta-LoRA
        """
        if self.config.method == PEFTMethod.DELTA_LORA:
            for layer in self.peft_layers:
                if isinstance(layer, MultiQueryPEFT):
                    if hasattr(layer.q_peft, 'update_base_weight'):
                        layer.q_peft.update_base_weight()
                        layer.v_peft.update_base_weight()
                elif isinstance(layer, PEFTWrapper):
                    if hasattr(layer.peft_layer, 'update_base_weight'):
                        layer.peft_layer.update_base_weight()
    
    def set_trainable_modules(self, module_patterns: List[str], trainable: bool = True):
        """Dynamically set which modules are trainable"""
        self.weight_controller.set_trainable(module_patterns, trainable)
    
    def forward(self, batched_input, multimask_output, image_size):
        """Forward pass through the PEFT-enhanced SAM model"""
        return self.sam(batched_input, multimask_output, image_size)


# ============================================================================
# CONVENIENCE FUNCTION (MAIN ENTRY POINT)
# ============================================================================

def create_peft_sam(
    sam_model,
    method: str = "lora",
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    encoder_layers: List[int] = None,
    decoder_layers: List[int] = None,
    enable_adapters: bool = False,
    adapter_modules: List[str] = None
) -> EnhancedSAMPEFT:
    """
    Main entry point: Create a PEFT-enhanced SAM model
    
    Args:
        sam_model: Base SAM model (standard or Mobile-SAM with TinyViT)
        method: PEFT method - one of:
            - 'lora': Standard LoRA
            - 'lora_fa': LoRA with frozen A matrix
            - 'vera': VeRA with shared frozen matrices
            - 'delta_lora': Delta-LoRA with weight updates
        rank: Rank for low-rank approximation (typically 4-16)
        alpha: Scaling factor (typically same as rank)
        dropout: Dropout rate for PEFT layers (0.0-0.5)
        encoder_layers: Which encoder layers to apply PEFT (None = all)
        decoder_layers: Which decoder layers to apply PEFT (None = all)
        enable_adapters: Enable existing adapter layers in the model
        adapter_modules: Specific adapter module patterns to enable
    
    Returns:
        EnhancedSAMPEFT model ready for training
    
    Example usage:
        >>> sam = sam_model_registry['vit_b']()  # or mobile_sam
        >>> peft_model = create_peft_sam(
        ...     sam,
        ...     method='lora',
        ...     rank=8,
        ...     alpha=8,
        ...     encoder_layers=[5, 6, 7, 8, 9, 10, 11],  # Last 7 layers
        ...     enable_adapters=True
        ... )
        >>> # Now train peft_model with your data
    """
    
    # Create configuration
    config = PEFTConfig(
        method=PEFTMethod(method),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        enable_adapters=enable_adapters,
        adapter_modules=adapter_modules
    )
    
    # Create and return PEFT model
    peft_model = EnhancedSAMPEFT(sam_model, config)
    
    # Print summary
    total_params = sum(p.numel() for p in sam_model.parameters())
    trainable_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    print(f"\nPEFT Model Summary:")
    print(f"Method: {method}")
    print(f"Rank: {rank}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    return peft_model


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage (commented out, for reference)
    """
    # For standard SAM
    from segment_anything import sam_model_registry
    sam = sam_model_registry['vit_b'](checkpoint='path/to/checkpoint.pth')
    
    # For Mobile-SAM (TinyViT)
    from mobile_sam import sam_model_registry
    sam = sam_model_registry['vit_t'](checkpoint='path/to/mobile_sam.pt')
    
    # Create PEFT model
    peft_model = create_peft_sam(
        sam,
        method='lora',
        rank=8,
        alpha=8,
        encoder_layers=None,  # Apply to all layers
        decoder_layers=None,  # Apply to all layers
        enable_adapters=True  # Enable TinyViT adapters if present
    )
    
    # Training loop
    for batch in dataloader:
        output = peft_model(batch['input'], multimask_output=True, image_size=1024)
        loss = compute_loss(output, batch['target'])
        loss.backward()
        
        # If using Delta-LoRA, update base weights after optimizer step
        if peft_model.config.method == PEFTMethod.DELTA_LORA:
            optimizer.step()
            peft_model.update_delta_lora()
        else:
            optimizer.step()
    """
