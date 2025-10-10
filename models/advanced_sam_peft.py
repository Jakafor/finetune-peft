"""
Enhanced SAM PEFT Module with TinyViT and SemanticSam Support
Supports LoRA, LoRA-FA, VeRA, and Delta-LoRA for standard SAM, Mobile-SAM (TinyViT), and SemanticSam

Reading Guide:
1. Start with PEFTMethod and PEFTConfig classes - these define configuration
2. Review the base PEFT layer classes (BasePEFTLayer, LoRALayer, etc.) - these implement different PEFT methods
3. Look at MultiQueryPEFT class - handles attention QKV projections
4. Main class is EnhancedSAMPEFT - orchestrates everything
5. The key TinyViT handling is in _apply_encoder_peft() method
6. NEW: SemanticSam handling detects adapter structure and applies PEFT appropriately
7. Use create_peft_sam() function as the main entry point
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
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
        # Module patterns can be substrings of parameter names or special keywords:
        # SemanticSam: 'decoder_head', 'decoder_neck', 'prompt_encoder', 'image_encoder', 'decoder' (both), 'all'
        # Regular SAM: 'encoder', 'decoder', 'image_encoder', 'prompt_encoder', 'all'
    
        model = self.model

    # Helper: navigate dotted attributes safely
        def _get_attr(obj, path):
            cur = obj
            for part in path.split('.'):
                if cur is None or not hasattr(cur, part):
                    return None
                cur = getattr(cur, part)
            return cur

    # Detect SemanticSam by adapter presence
        is_semantic = hasattr(model, "img_adapter") and hasattr(model, "prompt_adapter") and hasattr(model, "mask_adapter")

        groups = {}

        if is_semantic:
            # Adapters expose the wrapped SAM submodules under these names
            img_enc = _get_attr(model, "img_adapter.sam_img_encoder") or _get_attr(model, "image_encoder")
            prm_enc = _get_attr(model, "prompt_adapter.sam_prompt_encoder") or _get_attr(model, "prompt_encoder")
            dec_neck = _get_attr(model, "mask_adapter.decoder_neck")
            dec_head = _get_attr(model, "mask_adapter.decoder_head")
            dec_full = []
            if dec_neck is not None: dec_full.append(dec_neck)
            if dec_head is not None: dec_full.append(dec_head)
            # Fallback to the wrapped original decoder if present
            wrapped_dec = _get_attr(model, "mask_adapter.sam_mask_decoder")
            if not dec_full and wrapped_dec is not None:
                dec_full = [wrapped_dec]

            groups.update({
                "image_encoder": img_enc,
                "prompt_encoder": prm_enc,
                "decoder_neck": dec_neck,
                "decoder_head": dec_head,
                "decoder": dec_full or None,  # both neck+head, or wrapped decoder
            })
        else:
            # Standard SAM / Mobile-SAM
            img_enc = _get_attr(model, "image_encoder")
            prm_enc = _get_attr(model, "prompt_encoder")
            dec    = _get_attr(model, "mask_decoder")

            groups.update({
                "encoder": img_enc,
                "image_encoder": img_enc,     # alias
                "prompt_encoder": prm_enc,    # alias
                "decoder": dec,
            })

        # Collect the exact parameter objects we want to toggle
        target_param_ids = set()
        special_keys = set(groups.keys())

        for pat in module_patterns:
            if pat == "all":
                for _, p in model.named_parameters():
                    target_param_ids.add(id(p))
                continue

            if pat in special_keys:
                mods = groups.get(pat)
                if mods is None:
                    continue
                if isinstance(mods, (list, tuple)):
                    for m in mods:
                        if m is None: 
                            continue
                        for _, p in m.named_parameters():
                            target_param_ids.add(id(p))
                else:
                    for _, p in mods.named_parameters():
                        target_param_ids.add(id(p))
            else:
                # Fallback: substring match on full parameter names
                for name, p in model.named_parameters():
                    if pat in name:
                        target_param_ids.add(id(p))

        # Apply the toggle
        for name, p in model.named_parameters():
            if id(p) in target_param_ids:
                p.requires_grad = trainable

        # Print trainable parameters
        _total = sum(p.numel() for p in model.parameters())
        _train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable: {_train:,} / {_total:,}  ({_train/_total:.2%})")
    
    def enable_adapters(self, adapter_patterns: List[str] = None):
        """Enable adapter layers if they exist in the model"""
        if adapter_patterns is None:
            # Default adapter patterns from your TinyViT code
            adapter_patterns = ['Adapter', 'MLP_Adapter', 
                              'Space_Adapter']
        
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
        
        self._base_module: Optional[nn.Module] = None   # e.g., qkv Linear or q_proj
        self._row_slice: Optional[slice] = None         # which rows to update in base
        # Keep prev_delta as a BUFFER so it moves with the module and saves cleanly.
        self.register_buffer("prev_delta", None, persistent=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def compute_delta(self) -> torch.Tensor:
        """Compute the current LoRA delta (B @ A)"""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling
    
    def set_base_ref(self, base_module: nn.Linear, row_slice: Optional[slice] = None):
        """
        base_module: module that owns the live parameter to update (e.g., qkv or q_proj)
        row_slice: rows in base_module.weight to be updated; None => all rows
        """
        if not isinstance(base_module, nn.Linear):
            raise TypeError(f"DeltaLoRALayer expects a Linear base module, got {type(base_module)}")
        self._base_module = base_module
        self._row_slice = row_slice if row_slice is not None else slice(None)
    
    def update_base_weight(self):
        """Update base weight with the difference from previous delta"""
        if self._base_module is None or self._row_slice is None:
            return  # not wired (should not happen if set_base_ref() is called)

        current_delta = self.compute_delta()  # [out_features, in_features]

        # Where to write: always address the live parameter
        W = self._base_module.weight  # Parameter on the correct device/dtype
        rows = self._row_slice

        # Compute diff from previous delta (zeros if first time)
        if self.prev_delta is None:
            delta_diff = current_delta
        else:
            delta_diff = current_delta - self.prev_delta.to(current_delta.device)

        # Update in-place with proper dtype/device, under no_grad
        with torch.no_grad():
            W[rows, :].add_(delta_diff.to(W.dtype).to(W.device))

        # Cache current delta for the next step (store as buffer so it moves with .to(...))
        self.prev_delta = current_delta.detach().to(W.device, dtype=W.dtype)

    
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
            self.q_peft.set_base_ref(self.qkv, slice(0, dim))
            self.v_peft.set_base_ref(self.qkv, slice(-dim, None))
    
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
    Supports standard SAM (ViT), Mobile-SAM (TinyViT), and SemanticSam
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
        
        # NEW: Detect model type
        self.is_semantic_sam = self._detect_semantic_sam()
        self.is_tiny_vit = False  # Will be set during detection
        
        # Get the actual encoder and decoder based on model type
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        
        # Apply PEFT to encoder
        # CHANGED: Fixed condition - should apply if encoder_layers is None OR has content
        if config.encoder_layers is None or (config.encoder_layers is not None and len(config.encoder_layers) > 0):
            self._apply_encoder_peft()
        
        # Apply PEFT to decoder (if it exists and is requested)
        if self.decoder is not None:
            # CHANGED: Fixed condition - should apply if decoder_layers is None OR has content
            if config.decoder_layers is None or (config.decoder_layers is not None and len(config.decoder_layers) > 0):
                self._apply_decoder_peft()
        
        # Enable adapters if requested
        if config.enable_adapters:
            self.weight_controller.enable_adapters(config.adapter_modules)
    
    # NEW: Method to detect SemanticSam
    def _detect_semantic_sam(self) -> bool:
        """
        Detect if the model is SemanticSam (has img_adapter, prompt_adapter, mask_adapter)
        """
        return (hasattr(self.sam, 'img_adapter') and 
                hasattr(self.sam, 'prompt_adapter') and 
                hasattr(self.sam, 'mask_adapter'))
    
    # NEW: Method to get encoder based on model type
    def _get_encoder(self):
        """Get the encoder based on model type (SemanticSam or standard)"""
        if self.is_semantic_sam:
            # For SemanticSam, the encoder is accessed through img_adapter
            return self.sam.img_adapter.sam_img_encoder
        else:
            # Standard SAM or Mobile-SAM
            return self.sam.image_encoder
    
    # NEW: Method to get decoder based on model type
    def _get_decoder(self):
        """Get the decoder based on model type (SemanticSam or standard)"""
        if self.is_semantic_sam:
            # For SemanticSam, we have mask_adapter with potentially different structure
            # Check if it has sam_mask_decoder (BaseMaskDecoderAdapter) or decoder_neck/head (SemMaskDecoderAdapter)
            if hasattr(self.sam.mask_adapter, 'sam_mask_decoder'):
                return self.sam.mask_adapter.sam_mask_decoder
            elif hasattr(self.sam.mask_adapter, 'decoder_neck'):
                # For semantic variant, return the neck which contains the transformer
                return self.sam.mask_adapter.decoder_neck
            else:
                return None
        else:
            # Standard SAM
            if hasattr(self.sam, 'mask_decoder'):
                return self.sam.mask_decoder
            else:
                return None
    
    # CHANGED: Updated to check encoder instead of self.sam.image_encoder
    def _detect_tiny_vit(self) -> bool:
        """
        Detect if the model is using TinyViT architecture
        TinyViT has a different structure with layers containing blocks
        """
        try:
            # Check if the encoder has 'layers' attribute (TinyViT structure)
            if hasattr(self.encoder, 'layers'):
                # Further check: TinyViT layers have blocks inside
                if len(self.encoder.layers) > 0:
                    first_layer = self.encoder.layers[0]
                    # ConvLayer doesn't have attention, skip to next layers
                    if len(self.encoder.layers) > 1:
                        second_layer = self.encoder.layers[1]
                        if hasattr(second_layer, 'blocks'):
                            return True
            return False
        except:
            return False
    
    def _apply_encoder_peft(self):
        """Apply PEFT to encoder blocks (handles both ViT and TinyViT)"""
        
        # CHANGED: Check TinyViT after getting encoder
        self.is_tiny_vit = self._detect_tiny_vit()
        
        if self.is_tiny_vit:
            model_type = "SemanticSam with TinyViT" if self.is_semantic_sam else "TinyViT (Mobile-SAM)"
            print(f"Detected {model_type} architecture")
            self._apply_tiny_vit_peft()
        else:
            model_type = "SemanticSam with ViT" if self.is_semantic_sam else "standard ViT"
            print(f"Detected {model_type} architecture")
            self._apply_standard_vit_peft()
    
    # CHANGED: Updated to use self.encoder instead of self.sam.image_encoder
    def _apply_standard_vit_peft(self):
        """Apply PEFT to standard ViT encoder"""
        
        # Standard SAM has blocks directly in encoder
        blocks = self.encoder.blocks
        
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
                model_type = "SemanticSam" if self.is_semantic_sam else "SAM"
                print(f"Applied PEFT to {model_type} ViT encoder block {idx}")
    
    # CHANGED: Updated to use self.encoder
    def _apply_tiny_vit_peft(self):
        """Apply PEFT to TinyViT encoder (Mobile-SAM or SemanticSam with TinyViT)"""
        
        # First, make the conv layers trainable (as in original code)
        for n, value in self.encoder.layers[0].named_parameters():
            value.requires_grad = True
            print(f"Unfroze conv layer parameter: {n}")
        
        # Apply PEFT to attention blocks in layers[1:] 
        # (layer[0] is ConvLayer, doesn't have attention)
        for layer_idx, layer in enumerate(self.encoder.layers[1:]):
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
                        model_type = "SemanticSam TinyViT" if self.is_semantic_sam else "TinyViT"
                        print(f"Applied PEFT to {model_type} layer {layer_idx+1}, block {block_idx}")
    
    # CHANGED: Updated to use self.decoder and handle SemanticSam structure
    def _apply_decoder_peft(self):
        """Apply PEFT to decoder transformer"""
        
        layer_class = self._get_layer_class()
        
        # Apply to transformer layers
        if hasattr(self.decoder, 'transformer'):
            transformer = self.decoder.transformer
            
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
            
            # NEW: Log model type for decoder
            model_type = "SemanticSam" if self.is_semantic_sam else "SAM"
            print(f"Applied PEFT to {model_type} decoder")
    
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
                    peft_layer.set_base_ref(base_layer)
                wrapped = PEFTWrapper(base_layer, peft_layer)
                
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
    
    # CHANGED: Updated forward to handle SemanticSam's different forward signature
    def forward(self, *args, **kwargs):
        """Forward pass through the PEFT-enhanced SAM model"""
        if self.is_semantic_sam:
            # SemanticSam's forward takes just img
            return self.sam(*args, **kwargs)
        else:
            # Standard SAM forward
            return self.sam(*args, **kwargs)
        
    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that handles both PEFT and non-PEFT checkpoints.
        """
        # Check if state dict has PEFT structure
        has_sam_prefix = any(k.startswith('sam.') for k in state_dict.keys())
        current_keys = set(self.state_dict().keys())
        
        if not has_sam_prefix:
            # Checkpoint doesn't have PEFT structure, add 'sam.' prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                if f'sam.{key}' in current_keys:
                    new_state_dict[f'sam.{key}'] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Call parent load_state_dict
        return super().load_state_dict(state_dict, strict=strict)


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
        sam_model: Base SAM model (standard, Mobile-SAM with TinyViT, or SemanticSam)
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
        >>> # For standard SAM
        >>> sam = sam_model_registry['vit_b']()
        >>> 
        >>> # For Mobile-SAM
        >>> mobile_sam = sam_model_registry['vit_t']()
        >>> 
        >>> # For SemanticSam
        >>> from models.sam import sam_model_registry
        >>> base_sam = sam_model_registry['vit_b'](args, checkpoint='sam_vit_b.pth')
        >>> semantic_sam = SemanticSam(ori_sam=base_sam, class_num=20)
        >>> 
        >>> # Create PEFT model for any of them
        >>> peft_model = create_peft_sam(
        ...     semantic_sam,  # or sam or mobile_sam
        ...     method='lora',
        ...     rank=8,
        ...     alpha=8,
        ...     encoder_layers=[5, 6, 7, 8, 9, 10, 11],  # Last 7 layers
        ...     enable_adapters=True
        ... )
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
    
    # For SemanticSam
    from models.sam.modeling.build_sam import sam_model_registry
    from models.sam.modeling.extend_sam import SemanticSam
    
    # Create base SAM
    base_sam = sam_model_registry['vit_b'](args, checkpoint='sam_vit_b.pth')
    # or use semantic builders directly:
    # semantic_sam = sam_model_registry['semantic_vit_b'](args, checkpoint='sam_vit_b.pth', num_classes=20)
    
    # Create SemanticSam
    semantic_sam = SemanticSam(ori_sam=base_sam, class_num=20, model_type='vit_b')
    
    # Create PEFT model for SemanticSam
    peft_model = create_peft_sam(
        semantic_sam,
        method='lora',
        rank=8,
        alpha=8,
        encoder_layers=None,  # Apply to all layers
        decoder_layers=None,  # Apply to all layers
        enable_adapters=True  # Enable existing adapters if present
    )
    
    # Training loop for SemanticSam
    for batch in dataloader:
        # SemanticSam forward only takes img
        output = peft_model(batch['image'])
        loss = compute_loss(output, batch['target'])
        loss.backward()
        
        # If using Delta-LoRA, update base weights after optimizer step
        if peft_model.config.method == PEFTMethod.DELTA_LORA:
            optimizer.step()
            peft_model.update_delta_lora()
        else:
            optimizer.step()
    """
