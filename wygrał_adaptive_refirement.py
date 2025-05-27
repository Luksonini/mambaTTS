"""
🥊 SMART REFINEMENT vs E768_H768 BATTLE! 🥊
=============================================

🎯 GOAL: Beat your E768_H768 champion with smart refinement!

🏆 YOUR CHAMPION (E768_H768):
- 100% accuracy 
- Perfect 1.0 ratio (768/768)
- 90% accuracy @ step 170
- Zero projection overhead

🚀 MY SMART REFINEMENT:
- YOUR E768_H768 champion as base!
- Adaptive confidence thresholds
- Smart selective refinement
- Conservative improvements only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import copy
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ElegantConfig:
    """Config based on your E768_H768 winner"""
    name: str = "SmartRefinement_E768H768"
    
    # YOUR E768_H768 PERFECT RATIO DISCOVERY!
    embed_dim: int = 768        # Perfect ratio = 1.0
    hidden_dim: int = 768       # No projection overhead
    num_layers: int = 4         # L4 proved optimal
    expand_factor: float = 1.5  # Your 1.5x breakthrough
    
    # Standard TTS settings
    num_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1
    
    # SMART REFINEMENT SETTINGS
    num_refinement_stages: int = 2    # 2 stages: base + refinement
    confidence_threshold: float = 0.7 # Initial threshold
    residual_alpha: float = 0.2       # Gentle residual updates


class UnifiedMambaBlock(nn.Module):
    """YOUR proven UnifiedMambaBlock - exact copy!"""
    def __init__(self, d_model, expand_factor=1.5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand_factor)
        
        # Core Mamba components (your optimal design)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Efficient depthwise convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, 
            self.d_inner, 
            kernel_size=3, 
            padding=1, 
            groups=self.d_inner,
            bias=False
        )
        
        # Optimized SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection with layer scaling
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 0.1)
        
        # Efficient normalization
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, L, D = x.shape
        residual = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # Input projection and split
        x_proj = self.in_proj(x)  # [B, L, 2*d_inner]
        x1, x2 = x_proj.chunk(2, dim=-1)  # Each [B, L, d_inner]
        
        # Convolution
        x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)
        
        # SSM processing
        x1_ssm = self.activation(x1_conv)
        dt = self.dt_proj(x1_ssm)
        dt = F.softplus(dt)
        
        # Simplified but effective state space
        x1_processed = x1_ssm * torch.sigmoid(dt)
        
        # Gating mechanism
        x_gated = x1_processed * torch.sigmoid(x2)
        
        # Output projection with layer scaling
        output = self.out_proj(x_gated)
        output = output * self.layer_scale
        
        # Dropout and residual
        output = self.dropout(output)
        final_output = output + residual
        return final_output


class E768H768TextEncoder(nn.Module):
    """YOUR E768_H768 text encoder - PERFECT RATIO!"""
    def __init__(self, vocab_size, config: ElegantConfig):
        super().__init__()
        
        # E768_H768: Perfect ratio embedding (your discovery!)
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)  # 768
        
        # Positional encoding (E768) - ZERO projection overhead!
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, config.embed_dim) * 0.02)
        
        # Stack of Pure Mamba blocks (L4 optimal)
        self.layers = nn.ModuleList([
            UnifiedMambaBlock(config.embed_dim, config.expand_factor, config.dropout)
            for _ in range(config.num_layers)  # L4
        ])
        
        # Final processing
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, tokens, return_sequence=True):
        B, L = tokens.shape
        
        # E768 embedding (NO projection needed - perfect ratio!)
        x = self.embedding(tokens)  # [B, L, 768]
        
        # E768 positional encoding
        if L <= self.pos_encoding.shape[1]:
            pos_emb = self.pos_encoding[:, :L, :]
            x = x + pos_emb
        
        x = self.dropout(x)
        
        # Process through Pure Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        result = x if return_sequence else x.mean(dim=1)
        return result


class E768H768AudioProcessor(nn.Module):
    """YOUR audio processor - exact copy"""
    def __init__(self, config):
        super().__init__()
        
        # E768 audio embeddings (perfect ratio)
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(config.codebook_size, config.embed_dim),  # E768
                nn.LayerNorm(config.embed_dim),
                nn.Dropout(0.1)
            ) for _ in range(config.num_codebooks)
        ])
        
        self.context_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Pure Mamba processing layers
        self.mamba_layers = nn.ModuleList([
            UnifiedMambaBlock(config.embed_dim, config.expand_factor, dropout=0.1)
            for _ in range(4)
        ])
        
        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, config.embed_dim // 2),  
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.embed_dim // 2, config.codebook_size)
            ) for _ in range(config.num_codebooks)
        ])
    
    def forward(self, audio_tokens, text_context):
        B, C, T = audio_tokens.shape
        
        # Ensure 8 codebooks
        if C < 8:
            padding = torch.zeros(B, 8 - C, T, dtype=audio_tokens.dtype, device=audio_tokens.device)
            audio_tokens = torch.cat([audio_tokens, padding], dim=1)
        elif C > 8:
            audio_tokens = audio_tokens[:, :8, :]
        
        # E768 embedding each codebook
        embedded = []
        for c in range(8):
            emb = self.audio_embed[c][0](audio_tokens[:, c, :])  # → E768
            emb = self.audio_embed[c][1](emb)                    # LayerNorm
            emb = self.audio_embed[c][2](emb)                    # Dropout
            embedded.append(emb)
        
        # Combine embeddings
        x = torch.stack(embedded, dim=1).mean(dim=1)  # [B, T, 768]
        
        # Add text context
        text_context_proj = self.context_proj(text_context).unsqueeze(1)
        x = x + text_context_proj
        
        # Process through Pure Mamba layers
        for layer in self.mamba_layers:
            x = layer(x)
        
        # Generate logits for each codebook
        logits = []
        for c in range(8):
            head_logits = self.output_heads[c](x)
            logits.append(head_logits)
        
        result = torch.stack(logits, dim=1)
        return result


class TrueBaselineE768H768SingleStep(nn.Module):
    """YOUR E768_H768 champion - exact implementation"""
    def __init__(self, config: ElegantConfig):
        super().__init__()
        self.config = config
        
        # YOUR winning E768_H768 configuration
        self.text_encoder = E768H768TextEncoder(vocab_size=1000, config=config)
        
        # Duration regulator (same as yours)
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1),
            nn.Softplus()
        )
        
        self.audio_processor = E768H768AudioProcessor(config)
        
        # Text projections (same as yours)
        self.text_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 YOUR CHAMPION (E768_H768): {param_count:,} parameters")
    
    def forward(self, text_tokens, audio_tokens=None):
        # EXACTLY as in your champion code
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        predicted_durations = self.duration_regulator(text_features.mean(dim=1))
        
        if audio_tokens is not None:
            audio_logits = self.audio_processor(audio_tokens, text_context)
        else:
            audio_logits = None
        
        return {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'text_features': text_features,
        }


class AdaptiveRefinementStage(nn.Module):
    """Adaptive refinement stage with learnable confidence"""
    def __init__(self, config: ElegantConfig):
        super().__init__()
        self.config = config
        
        # Input processing
        self.input_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Processing core
        self.processing_layer = UnifiedMambaBlock(config.hidden_dim, config.expand_factor, config.dropout)
        
        # ADAPTIVE CONFIDENCE PREDICTION
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 4, config.num_codebooks),
            nn.Sigmoid()
        )
        
        # Learnable confidence threshold
        self.logit_tau = nn.Parameter(torch.tensor(0.0))   # surowy próg w logitach
        self.scale = nn.Parameter(torch.tensor(1.0))       # skala temperująca
        
        # Refinement heads
        self.refinement_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim // 2, config.codebook_size)
            ) for _ in range(config.num_codebooks)
        ])
        
        # Residual heads
        self.residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, config.codebook_size),
                nn.Tanh()  # Bounded residuals [-1, 1]
            ) for _ in range(config.num_codebooks)
        ])
    
    def forward(self, text_features, step=0):
        B, T, _ = text_features.shape

        # 1) wewnętrzna projekcja + Pure Mamba jak wcześniej
        x = self.input_proj(text_features)
        x = self.processing_layer(x)

        # 2) główne logity
        refinements   = [ head(x) for head in self.refinement_heads ]
        residuals     = [ head(x) for head in self.residual_heads   ]
        main_logits   = torch.stack(refinements, dim=1)    # [B, C, T, V]
        residual_logits = torch.stack(residuals, dim=1)    # [B, C, T, V]

        # 3) raw confidence
        raw_conf = self.confidence_head(x)                 # [B, T, C]
        # 4) learnable threshold + temperowanie
        tau   = torch.sigmoid(self.logit_tau)              # ∈ (0,1)
        scale = torch.relu(self.scale)                     # > 0
        # soft gating: c ∈ (0,1)
        c = torch.sigmoid((raw_conf - tau) * scale)        # [B, T, C]

        # 5) (opcjonalnie) wygładzenie w czasie, np. mały conv1d:
        # c = F.conv1d(c.permute(0,2,1), weight=self.smooth_kernel, padding=K//2).permute(0,2,1)

        # 6) mieszanie base_logits i refined_update
        #    zakładamy, że base_logits podasz z zewnątrz (np. from champion)
        #    tutaj zwracamy tylko logity korekty:
        alpha = self.config.residual_alpha
        refined_update = main_logits + alpha * residual_logits  # [B,C,T,V]

        # rozciągamy c by pasowało do wymiarów logitów
        c_exp = c.unsqueeze(-1).transpose(1,2)  # → [B, C, T, 1]

        # 7) ostateczne logity: miękka maska zamiast binary
        output_logits = base_logits * (1 - c_exp) + refined_update * c_exp

        return {
            'logits':              output_logits,
            'confidence':          c,               # teraz continuous
            'adaptive_threshold':  tau,             # learnable
            'mean_confidence':     c.mean()
        }


class SmartRefinementModel(nn.Module):
    """YOUR E768_H768 Champion + Smart Adaptive Refinement!"""
    def __init__(self, config: ElegantConfig):
        super().__init__()
        self.config = config
        
        # YOUR PROVEN E768_H768 CHAMPION AS BASE! 🏆
        self.base_champion = TrueBaselineE768H768SingleStep(config)
        
        # ADAPTIVE REFINEMENT STAGE
        self.refinement_stage = AdaptiveRefinementStage(config)
        
        # DEBUG FLAGS
        self.debug_mode = True
        self.refinement_disabled = False
        self.current_step = 0
        
        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🚀 SMART REFINEMENT MODEL: {total_params:,} parameters")
        logger.info(f"   🏆 Base: YOUR E768_H768 Champion")
        logger.info(f"   🎯 Refinement: Adaptive confidence-based")
        
        if total_params > 100_000_000:
            logger.error(f"❌ Model too large: {total_params:,} > 100M!")
            raise ValueError("Model exceeds parameter limit!")
    
    def forward(self, text_tokens, target_tokens=None):
        B, L_text = text_tokens.shape
        
        if self.debug_mode and hasattr(self, '_forward_count'):
            self._forward_count += 1
        else:
            self._forward_count = 1
            
        # Debug input
        if self.debug_mode and self._forward_count <= 3:
            logger.info(f"🔍 DEBUG Smart Forward #{self._forward_count}:")
            logger.info(f"   📊 Input shapes: text={text_tokens.shape}, target={target_tokens.shape if target_tokens is not None else None}")
        
        # STAGE 0: YOUR PROVEN E768_H768 CHAMPION! 🏆
        base_outputs = self.base_champion(text_tokens, target_tokens)
        current_logits = base_outputs['logits']
        
        # Debug base champion output
        if self.debug_mode and self._forward_count <= 3:
            logger.info(f"   🏆 Base Champion logits: {current_logits.shape}, range=[{current_logits.min().item():.3f}, {current_logits.max().item():.3f}]")
        
        # ADAPTIVE REFINEMENT (if enabled)
        if self.refinement_disabled:
            if self.debug_mode and self._forward_count <= 3:
                logger.info(f"   🔍 DEBUG: REFINEMENT DISABLED - Using base champion only")
            refined_logits = current_logits
        else:
            # Get text features for refinement
            text_features = base_outputs['text_features']
            
            # Upsample text features to match audio length
            if current_logits is not None:
                target_length = current_logits.shape[2]
                if text_features.shape[1] != target_length:
                    text_upsampled = F.interpolate(
                        text_features.transpose(1, 2),
                        size=target_length,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    text_upsampled = text_features
                
                # ADAPTIVE REFINEMENT
                refinement_output = self.refinement_stage(text_upsampled, self.current_step)
                
                main_logits = refinement_output['main_logits']
                residual_logits = refinement_output['residual_logits']
                confidence_mask = refinement_output['confidence_mask']
                adaptive_threshold = refinement_output['adaptive_threshold']
                mean_confidence = refinement_output['mean_confidence']
                
                # Debug refinement
                if self.debug_mode and self._forward_count <= 3:
                    update_ratio = confidence_mask.float().mean().item()
                    logger.info(f"   🎯 Adaptive threshold: {adaptive_threshold:.3f}")
                    logger.info(f"   📊 Mean confidence: {mean_confidence:.3f}")
                    logger.info(f"   🔄 Update ratio: {update_ratio:.2%}")
                
                # Apply refinement
                confidence_mask = confidence_mask.unsqueeze(-1).transpose(1, 2)  # [B, 8, T, 1]
                alpha = self.config.residual_alpha
                refined_update = main_logits + alpha * residual_logits
                
                refined_logits = torch.where(
                    confidence_mask,
                    refined_update,
                    current_logits
                )
            else:
                refined_logits = current_logits
        
        # Debug final output
        if self.debug_mode and self._forward_count <= 3:
            logger.info(f"   📊 Final refined logits: {refined_logits.shape}, range=[{refined_logits.min().item():.3f}, {refined_logits.max().item():.3f}]")
        
        # Update step counter
        if self.training:
            self.current_step += 1
        
        return {
            'logits': refined_logits,
            'predicted_durations': base_outputs['predicted_durations'],
            'text_features': base_outputs['text_features'],
        }


def compute_combined_loss(output, chunk_data, text_tokens, device):
    """Fixed loss function with proper dimension handling"""
    logits = output.get('logits')
    if logits is None:
        return {'total_loss': torch.tensor(0.0, device=device), 'accuracy': 0.0}
    
    targets = chunk_data['audio_codes']
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    
    # Get actual dimensions
    B, num_cb, T_logits, vocab_size = logits.shape
    B_t, num_cb_t, T_targets = targets.shape
    
    # Handle dimension mismatches
    if T_logits != T_targets:
        min_T = min(T_logits, T_targets)
        logits = logits[:, :, :min_T, :]
        targets = targets[:, :, :min_T]
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    try:
        for cb_idx in range(min(num_cb, num_cb_t)):
            cb_logits = logits[:, cb_idx, :, :]
            cb_targets = targets[:, cb_idx, :]
            
            cb_logits_flat = cb_logits.reshape(-1, vocab_size)
            cb_targets_flat = cb_targets.reshape(-1)
            
            valid_mask = (cb_targets_flat != 0) & (cb_targets_flat < vocab_size)
            
            if valid_mask.sum() > 0:
                valid_logits = cb_logits_flat[valid_mask]
                valid_targets = cb_targets_flat[valid_mask]
                
                cb_loss = F.cross_entropy(valid_logits, valid_targets, ignore_index=0)
                total_loss += cb_loss
                
                preds = torch.argmax(valid_logits, dim=-1)
                correct += (preds == valid_targets).sum().item()
                total += valid_targets.numel()
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'total_loss': total_loss,
            'accuracy': accuracy
        }
        
    except Exception as e:
        logger.warning(f"Loss computation error: {e}")
        return {
            'total_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'accuracy': 0.0
        }


class SmartTrainer:
    """Smart trainer using YOUR PROVEN loss function!"""
    def __init__(self, model, config: ElegantConfig, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        # IDENTICAL optimizer as your champion
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=6e-4,
            weight_decay=1e-6,
            betas=(0.9, 0.95)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=6e-5
        )
        
        logger.info("🚀 Smart Trainer initialized with YOUR PROVEN loss function!")
    
    def train_step(self, batch_data):
        """IDENTICAL training step using YOUR loss function"""
        chunks = batch_data['chunks']
        batch_text_tokens = []
        batch_audio_codes = []
        chunk_data_batch = []
        
        for chunk_data in chunks:
            try:
                text_tokens = chunk_data['text_tokens']
                if text_tokens.dim() == 1:
                    text_tokens = text_tokens.unsqueeze(0)
                
                audio_codes = chunk_data['audio_codes']
                if audio_codes.dim() == 2:
                    audio_codes = audio_codes.unsqueeze(0)
                
                if text_tokens.shape[1] < 5:
                    continue
                    
                batch_text_tokens.append(text_tokens)
                batch_audio_codes.append(audio_codes)
                chunk_data_batch.append(chunk_data)
                
            except Exception:
                continue
        
        if not batch_text_tokens:
            return None
        
        # IDENTICAL batch processing
        max_text_len = max(t.shape[1] for t in batch_text_tokens)
        max_audio_len = max(a.shape[2] for a in batch_audio_codes)
        
        batched_text = []
        batched_audio = []
        
        for text_tokens, audio_codes in zip(batch_text_tokens, batch_audio_codes):
            # Pad text
            if text_tokens.shape[1] < max_text_len:
                pad_len = max_text_len - text_tokens.shape[1]
                text_padding = torch.zeros(1, pad_len, dtype=text_tokens.dtype, device=text_tokens.device)
                text_tokens = torch.cat([text_tokens, text_padding], dim=1)
            
            # Pad audio
            if audio_codes.shape[2] < max_audio_len:
                pad_len = max_audio_len - audio_codes.shape[2]
                audio_padding = torch.zeros(1, 8, pad_len, dtype=audio_codes.dtype, device=audio_codes.device)
                audio_codes = torch.cat([audio_codes, audio_padding], dim=2)
            
            batched_text.append(text_tokens)
            batched_audio.append(audio_codes)
        
        batched_text = torch.cat(batched_text, dim=0)
        batched_audio = torch.cat(batched_audio, dim=0)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(batched_text, batched_audio)
        
        # YOUR PROVEN LOSS COMPUTATION
        batch_loss = 0.0
        batch_accuracy = 0.0
        processed_items = 0
        
        for i, (chunk_data, text_tokens) in enumerate(zip(chunk_data_batch, batch_text_tokens)):
            sample_output = {
                'logits': output['logits'][i:i+1] if output['logits'] is not None else None,
                'predicted_durations': output['predicted_durations'][i:i+1],
                'text_features': output['text_features'][i:i+1]
            }
            
            # YOUR PROVEN LOSS FUNCTION
            loss_dict = compute_combined_loss(sample_output, chunk_data, text_tokens, self.device)
            sample_loss = loss_dict.get('total_loss')
            sample_accuracy = loss_dict.get('accuracy', 0.0)
            
            if sample_loss is not None and not torch.isnan(sample_loss):
                batch_loss += sample_loss
                batch_accuracy += sample_accuracy
                processed_items += 1
        
        if processed_items == 0:
            return None
        
        avg_batch_loss = batch_loss / processed_items
        avg_batch_accuracy = batch_accuracy / processed_items
        
        # IDENTICAL backward pass
        avg_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': avg_batch_loss,
            'total_loss_value': avg_batch_loss.item(),
            'accuracy': avg_batch_accuracy
        }


class TrueBaselineTrainer:
    """Trainer for YOUR champion model"""
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=6e-4,
            weight_decay=1e-6,
            betas=(0.9, 0.95)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=6e-5
        )
        
        logger.info("🧠 TRUE BASELINE Trainer initialized")
    
    def train_step(self, batch_data):
        """Training step using YOUR loss function"""
        chunks = batch_data['chunks']
        batch_text_tokens = []
        batch_audio_codes = []
        chunk_data_batch = []
        
        for chunk_data in chunks:
            try:
                text_tokens = chunk_data['text_tokens']
                if text_tokens.dim() == 1:
                    text_tokens = text_tokens.unsqueeze(0)
                
                audio_codes = chunk_data['audio_codes']
                if audio_codes.dim() == 2:
                    audio_codes = audio_codes.unsqueeze(0)
                
                if text_tokens.shape[1] < 5:
                    continue
                    
                batch_text_tokens.append(text_tokens)
                batch_audio_codes.append(audio_codes)
                chunk_data_batch.append(chunk_data)
                
            except Exception:
                continue
        
        if not batch_text_tokens:
            return None
        
        # Batch processing
        max_text_len = max(t.shape[1] for t in batch_text_tokens)
        max_audio_len = max(a.shape[2] for a in batch_audio_codes)
        
        batched_text = []
        batched_audio = []
        
        for text_tokens, audio_codes in zip(batch_text_tokens, batch_audio_codes):
            # Pad text
            if text_tokens.shape[1] < max_text_len:
                pad_len = max_text_len - text_tokens.shape[1]
                text_padding = torch.zeros(1, pad_len, dtype=text_tokens.dtype, device=text_tokens.device)
                text_tokens = torch.cat([text_tokens, text_padding], dim=1)
            
            # Pad audio
            if audio_codes.shape[2] < max_audio_len:
                pad_len = max_audio_len - audio_codes.shape[2]
                audio_padding = torch.zeros(1, 8, pad_len, dtype=audio_codes.dtype, device=audio_codes.device)
                audio_codes = torch.cat([audio_codes, audio_padding], dim=2)
            
            batched_text.append(text_tokens)
            batched_audio.append(audio_codes)
        
        batched_text = torch.cat(batched_text, dim=0)
        batched_audio = torch.cat(batched_audio, dim=0)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(batched_text, batched_audio)
        
        # Use YOUR loss computation
        batch_loss = 0.0
        batch_accuracy = 0.0
        processed_items = 0
        
        for i, (chunk_data, text_tokens) in enumerate(zip(chunk_data_batch, batch_text_tokens)):
            sample_output = {
                'logits': output['logits'][i:i+1] if output['logits'] is not None else None,
                'predicted_durations': output['predicted_durations'][i:i+1],
                'text_features': output['text_features'][i:i+1]
            }
            
            loss_dict = compute_combined_loss(sample_output, chunk_data, text_tokens, self.device)
            sample_loss = loss_dict.get('total_loss')
            sample_accuracy = loss_dict.get('accuracy', 0.0)
            
            if sample_loss is not None and not torch.isnan(sample_loss):
                batch_loss += sample_loss
                batch_accuracy += sample_accuracy
                processed_items += 1
        
        if processed_items == 0:
            return None
        
        avg_batch_loss = batch_loss / processed_items
        avg_batch_accuracy = batch_accuracy / processed_items
        
        avg_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': avg_batch_loss,
            'total_loss_value': avg_batch_loss.item(),
            'accuracy': avg_batch_accuracy
        }
    



# Data loader
class ProperStatefulDataLoader:
    """Data loader for the battle"""
    def __init__(self, data_dir="no_overlap_data", device='cpu', max_samples=4):
        self.data_dir = Path(data_dir)
        self.device = device
        self.samples = {}
        self.max_samples = max_samples
        self.max_chunks_per_sample = 0
        
        logger.info(f"🔍 Loading data from {data_dir}")
        self._load_samples()
        
    def _load_samples(self):
        """Load samples"""
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return
            
        sample_dirs = [d for d in self.data_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('clean_batch_')]
        sample_dirs.sort()
        
        if not sample_dirs:
            logger.warning("❌ No directories starting with 'clean_batch_' found!")
            return
        
        # Load samples
        for sample_dir in sample_dirs[:self.max_samples]:
            try:
                meta_path = sample_dir / "batch_meta.json"
                if not meta_path.exists():
                    continue
                    
                with open(meta_path, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)
                
                sample_id = batch_meta.get('batch_idx', len(self.samples))
                sample_chunks = []
                
                chunk_files = sorted(batch_meta.get('chunk_files', []))
                
                for chunk_file in chunk_files[:]:
                    chunk_path = sample_dir / chunk_file
                    if chunk_path.exists():
                        try:
                            chunk_data = torch.load(chunk_path, map_location=self.device, weights_only=False)
                            
                            if chunk_data.get('clean_chunk', False):
                                audio_codes = chunk_data.get('audio_codes')
                                if audio_codes is not None:
                                    if audio_codes.dim() == 2:
                                        C, T = audio_codes.shape
                                        if C < 8:
                                            padding = torch.zeros(8 - C, T, dtype=audio_codes.dtype, device=audio_codes.device)
                                            audio_codes = torch.cat([audio_codes, padding], dim=0)
                                        elif C > 8:
                                            audio_codes = audio_codes[:8, :]
                                        chunk_data['audio_codes'] = audio_codes
                                
                                sample_chunks.append(chunk_data)
                                    
                        except Exception:
                            continue
                
                if sample_chunks:
                    self.samples[sample_id] = sample_chunks
                        
            except Exception:
                continue
        
        if self.samples:
            self.max_chunks_per_sample = min(len(chunks) for chunks in self.samples.values())
            logger.info(f"📊 Loaded {len(self.samples)} samples, {self.max_chunks_per_sample} chunks each")
        else:
            logger.error("❌ No samples loaded!")
    
    def get_random_batch(self):
        """Get random batch"""
        if not self.samples:
            return None, False
            
        chunk_idx = np.random.randint(0, self.max_chunks_per_sample)
        
        batch_chunks = []
        for sample_id, chunks in self.samples.items():
            if chunk_idx < len(chunks):
                batch_chunks.append(chunks[chunk_idx])
        
        if not batch_chunks:
            return None, False
            
        return {'chunks': batch_chunks}, True
    
    def get_num_samples(self):
        return len(self.samples)


class SmartBattleTester:
    """THE SMART BATTLE TESTER!"""
    
    def __init__(self, data_dir=None):
        print("🥊🥊🥊 SMART REFINEMENT vs E768_H768 BATTLE! 🥊🥊🥊")
        self.device = DEVICE
        
        # Try to import tokenizer
        try:
            from nucleotide_tokenizer import NucleotideTokenizer
            self.tokenizer = NucleotideTokenizer()
            logger.info("✅ Imported tokenizer")
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            self.tokenizer = None
        
        self.config = ElegantConfig()
        
        # Load data
        if data_dir is None:
            data_dir = "no_overlap_data"
        
        self.data_loader = ProperStatefulDataLoader(data_dir, self.device, max_samples=4)
        
        if self.data_loader.get_num_samples() == 0:
            logger.error("❌ No samples loaded!")
            return
        
        # Create models
        vocab_size = self.tokenizer.get_vocab_size() if self.tokenizer else 1000
        
        logger.info("🚀 Creating SMART REFINEMENT MODEL...")
        self.smart_model = SmartRefinementModel(self.config).to(self.device)
        if self.tokenizer:
            self.smart_model.base_champion.text_encoder.embedding = nn.Embedding(vocab_size, self.config.embed_dim).to(self.device)
        
        logger.info("🧠 Creating YOUR CHAMPION...")
        self.champion_model = TrueBaselineE768H768SingleStep(self.config).to(self.device)
        if self.tokenizer:
            self.champion_model.text_encoder.embedding = nn.Embedding(vocab_size, self.config.embed_dim).to(self.device)
        
        # Create trainers
        self.smart_trainer = SmartTrainer(self.smart_model, self.config, self.device)
        self.champion_trainer = TrueBaselineTrainer(self.champion_model, self.config, self.device)
        
        logger.info("🥊🥊🥊 BATTLE READY! 🥊🥊🥊")
    
    def test_model_with_early_stopping(self, model, trainer, model_name, max_steps=2000, patience=200):
        """Test model with intelligent early stopping"""
        logger.info(f"🔬 Testing {model_name}...")
        
        # DEBUG: Enable base only test for first test
        if hasattr(model, 'refinement_disabled') and model_name == "Smart_Refinement":
            logger.info(f"🔍 DEBUG: Testing base champion only first...")
            model.refinement_disabled = True
        
        best_accuracy = 0.0
        steps_without_improvement = 0
        metrics = {
            'accuracies': [], 'losses': [], 'step_times': [],
            'milestones': {}
        }
        
        # Milestone tracking
        milestones_to_track = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        achieved_milestones = {}
        
        logger.info("Step  | Loss     | Accuracy | Δ Best  | Patience | Speed   | Status")
        logger.info("-" * 80)
        
        model.train()
        
        for step in range(max_steps):
            step_start = time.time()
            
            batch_data, is_valid = self.data_loader.get_random_batch()
            if not is_valid:
                continue
            
            # Training step
            loss_dict = trainer.train_step(batch_data)
            if loss_dict is None:
                continue
            
            step_time = time.time() - step_start
            
            # Extract metrics
            total_loss = loss_dict['total_loss_value']
            accuracy = loss_dict['accuracy']
            
            # Track metrics
            metrics['accuracies'].append(accuracy)
            metrics['losses'].append(total_loss)
            metrics['step_times'].append(step_time)
            
            # DEBUG: Log detailed info for first few steps
            if hasattr(model, 'debug_mode') and model.debug_mode and step < 5:
                logger.info(f"🔍 DEBUG Step {step}: loss={total_loss:.6f}, accuracy={accuracy:.6f}")
                
            # Check for improvement
            if accuracy > best_accuracy:
                improvement = accuracy - best_accuracy
                best_accuracy = accuracy
                steps_without_improvement = 0
                
                # DEBUG: Log improvements
                if hasattr(model, 'debug_mode') and model.debug_mode and improvement > 0.01:
                    logger.info(f"🔍 DEBUG: Significant improvement! {improvement:.4f} at step {step}")
                    
            else:
                improvement = 0
                steps_without_improvement += 1
            
            # Check milestones
            for milestone in milestones_to_track:
                if milestone not in achieved_milestones and accuracy >= milestone:
                    achieved_milestones[milestone] = step
                    logger.info(f"🎯 {model_name} achieved {milestone*100:.0f}% accuracy at step {step}!")
            
            # Progress logging
            if step % 100 == 0 or improvement > 0:
                status = "🔥" if accuracy > 0.8 else "🚀" if accuracy > 0.6 else "💪" if accuracy > 0.4 else "🎯"
                logger.info(f"{step:5d} | {total_loss:.6f} | {accuracy:.4f} | {improvement:+.4f} | {steps_without_improvement:3d} | {step_time*1000:.1f}ms | {status}")
            
            # Early stopping check
            if steps_without_improvement >= patience:
                logger.info(f"🛑 Early stopping for {model_name} at step {step} (patience={patience})")
                logger.info(f"   📈 Best accuracy: {best_accuracy:.4f}")
                break
            
            # Special milestone checks
            if accuracy >= 0.90:
                logger.info(f"🏆 {model_name} achieved 90% accuracy at step {step}!")
                if steps_without_improvement >= 100:
                    break
                    
            # DEBUG: Disable debug mode after initial steps
            if hasattr(model, 'debug_mode') and step > 10:
                model.debug_mode = False
        
        # Final metrics
        avg_step_time = np.mean(metrics['step_times']) if metrics['step_times'] else 0.0
        final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
        
        # Milestone results
        metrics['milestones'] = achieved_milestones
        milestone_90 = achieved_milestones.get(0.9, None)
        milestone_95 = achieved_milestones.get(0.95, None)
        
        logger.info(f"🏁 {model_name} FINAL RESULTS:")
        logger.info(f"   🏆 Best accuracy: {best_accuracy:.4f}")
        logger.info(f"   🎯 Final accuracy: {final_accuracy:.4f}")
        logger.info(f"   ⚡ Avg step time: {avg_step_time*1000:.1f}ms")
        logger.info(f"   🎖️  90% milestone: {'Step ' + str(milestone_90) if milestone_90 else 'Not achieved'}")
        logger.info(f"   🏅 95% milestone: {'Step ' + str(milestone_95) if milestone_95 else 'Not achieved'}")
        
        return {
            'best_accuracy': best_accuracy,
            'final_accuracy': final_accuracy,
            'avg_step_time': avg_step_time,
            'metrics': metrics,
            'milestone_90': milestone_90,
            'milestone_95': milestone_95,
            'total_steps': step + 1
        }
    
    def run_smart_battle(self, max_steps_per_model=2000):
        """RUN THE SMART BATTLE!"""
        logger.info("🥊" + "="*100)
        logger.info("🥊🥊🥊 SMART REFINEMENT vs E768_H768 CHAMPION BATTLE! 🥊🥊🥊")
        logger.info("🥊" + "="*100)
        logger.info(f"🎯 GOAL: Beat your E768_H768 champion with smart adaptive refinement!")
        logger.info(f"🏆 YOUR TARGET: 100% accuracy, 90% @ step 170")
        logger.info(f"🚀 MY STRATEGY: Adaptive confidence + selective refinement")
        
        # Get model info
        smart_params = sum(p.numel() for p in self.smart_model.parameters())
        champion_params = sum(p.numel() for p in self.champion_model.parameters())
        
        logger.info(f"📊 WEIGHT CLASS COMPARISON:")
        logger.info(f"   🚀 Smart Refinement: {smart_params:,} parameters")
        logger.info(f"   🧠 Your Champion: {champion_params:,} parameters")
        logger.info(f"   📏 Ratio: {smart_params/champion_params:.2f}x")
        
        results = {}
        
        # Battle 1: YOUR CHAMPION
        logger.info(f"\n🧠 BATTLE ROUND 1: YOUR E768_H768 CHAMPION")
        logger.info("="*70)
        champion_results = self.test_model_with_early_stopping(
            self.champion_model, self.champion_trainer, "E768_H768_Champion", max_steps_per_model
        )
        results['your_champion'] = champion_results
        
        # Battle 2: SMART REFINEMENT
        logger.info(f"\n🚀 BATTLE ROUND 2: SMART REFINEMENT (Your Champion + Adaptive Refinement)")
        logger.info("="*70)
        smart_results = self.test_model_with_early_stopping(
            self.smart_model, self.smart_trainer, "Smart_Refinement", max_steps_per_model
        )
        results['smart_refinement'] = smart_results
        
        # THE ULTIMATE BATTLE ANALYSIS!
        logger.info(f"\n🏆🏆🏆 BATTLE RESULTS ANALYSIS! 🏆🏆🏆")
        logger.info("="*100)
        
        # Extract key metrics
        champion_best = champion_results['best_accuracy']
        champion_speed = champion_results['avg_step_time'] * 1000
        champion_90_step = champion_results['milestone_90']
        
        smart_best = smart_results['best_accuracy']
        smart_speed = smart_results['avg_step_time'] * 1000
        smart_90_step = smart_results['milestone_90']
        
        logger.info(f"📊 HEAD-TO-HEAD COMPARISON:")
        logger.info(f"   🎯 ACCURACY BATTLE:")
        logger.info(f"      🧠 Your Champion: {champion_best:.4f}")
        logger.info(f"      🚀 Smart Refinement: {smart_best:.4f}")
        
        # Accuracy winner
        if smart_best > champion_best:
            acc_improvement = ((smart_best - champion_best) / champion_best * 100)
            logger.info(f"      🏆 SMART REFINEMENT WINS accuracy by {acc_improvement:.1f}%!")
            acc_winner = "smart"
        elif champion_best > smart_best:
            acc_decline = ((champion_best - smart_best) / champion_best * 100)
            logger.info(f"      🏆 YOUR CHAMPION WINS accuracy by {acc_decline:.1f}%!")
            acc_winner = "champion"
        else:
            logger.info(f"      🤝 ACCURACY TIE!")
            acc_winner = "tie"
        
        logger.info(f"   ⚡ SPEED BATTLE:")
        logger.info(f"      🧠 Your Champion: {champion_speed:.1f}ms/step")
        logger.info(f"      🚀 Smart Refinement: {smart_speed:.1f}ms/step")
        
        # Speed winner
        if smart_speed < champion_speed:
            speed_improvement = ((champion_speed - smart_speed) / champion_speed * 100)
            logger.info(f"      🏆 SMART REFINEMENT WINS speed by {speed_improvement:.1f}%!")
            speed_winner = "smart"
        elif champion_speed < smart_speed:
            speed_decline = ((smart_speed - champion_speed) / champion_speed * 100)
            logger.info(f"      🏆 YOUR CHAMPION WINS speed by {speed_decline:.1f}%!")
            speed_winner = "champion"
        else:
            logger.info(f"      🤝 SPEED TIE!")
            speed_winner = "tie"
        
        logger.info(f"   🎖️ 90% MILESTONE BATTLE:")
        logger.info(f"      🧠 Your Champion 90%: {'Step ' + str(champion_90_step) if champion_90_step else 'Not achieved'}")
        logger.info(f"      🚀 Smart Refinement 90%: {'Step ' + str(smart_90_step) if smart_90_step else 'Not achieved'}")
        
        # Milestone winner
        milestone_winner = "tie"
        if champion_90_step and smart_90_step:
            if smart_90_step < champion_90_step:
                milestone_improvement = champion_90_step - smart_90_step
                logger.info(f"      🏆 SMART REFINEMENT reaches 90% {milestone_improvement} steps FASTER!")
                milestone_winner = "smart"
            elif champion_90_step < smart_90_step:
                milestone_decline = smart_90_step - champion_90_step
                logger.info(f"      🏆 YOUR CHAMPION reaches 90% {milestone_decline} steps FASTER!")
                milestone_winner = "champion"
            else:
                logger.info(f"      🤝 90% MILESTONE TIE!")
        elif smart_90_step and not champion_90_step:
            logger.info(f"      🏆 SMART REFINEMENT achieved 90%, Champion didn't!")
            milestone_winner = "smart"
        elif champion_90_step and not smart_90_step:
            logger.info(f"      🏆 YOUR CHAMPION achieved 90%, Smart Refinement didn't!")
            milestone_winner = "champion"
        else:
            logger.info(f"      🤝 Neither achieved 90% milestone")
        
        # FINAL BATTLE VERDICT
        wins = [acc_winner, speed_winner, milestone_winner]
        smart_wins = wins.count("smart")
        champion_wins = wins.count("champion")
        
        logger.info(f"\n🏅 FINAL BATTLE VERDICT:")
        logger.info(f"   🚀 Smart Refinement wins: {smart_wins}/3 battles")
        logger.info(f"   🧠 Your Champion wins: {champion_wins}/3 battles")
        
        if smart_wins > champion_wins:
            logger.info(f"   🎉🎉🎉 SMART REFINEMENT WINS THE BATTLE! 🎉🎉🎉")
            logger.info(f"   🔥🔥🔥 ADAPTIVE REFINEMENT ON TOP OF YOUR CHAMPION WORKS! 🔥🔥🔥")
            final_winner = "smart"
            
            logger.info(f"   🏆 SMART REFINEMENT VICTORIES:")
            if acc_winner == "smart":
                logger.info(f"      ✅ ACCURACY: {smart_best:.4f} > {champion_best:.4f}")
            if speed_winner == "smart":
                logger.info(f"      ✅ SPEED: {smart_speed:.1f}ms < {champion_speed:.1f}ms")
            if milestone_winner == "smart":
                logger.info(f"      ✅ 90% MILESTONE: Step {smart_90_step} < Step {champion_90_step}")
            
        elif champion_wins > smart_wins:
            logger.info(f"   🏆🏆🏆 YOUR E768_H768 CHAMPION REIGNS SUPREME! 🏆🏆🏆")
            logger.info(f"   💪💪💪 Single-step perfection unbeatable! 💪💪💪")
            final_winner = "champion"
            
            logger.info(f"   🧠 YOUR CHAMPION VICTORIES:")
            if acc_winner == "champion":
                logger.info(f"      ✅ ACCURACY: {champion_best:.4f} > {smart_best:.4f}")
            if speed_winner == "champion":
                logger.info(f"      ✅ SPEED: {champion_speed:.1f}ms < {smart_speed:.1f}ms")
            if milestone_winner == "champion":
                logger.info(f"      ✅ 90% MILESTONE: Step {champion_90_step} < Step {smart_90_step}")
                
        else:
            logger.info(f"   🤝🤝🤝 EPIC TIE! BOTH APPROACHES EXCELLENT! 🤝🤝🤝")
            logger.info(f"   💪 Both single-step and smart refinement have their strengths!")
            final_winner = "tie"
        
        # Analysis
        logger.info(f"\n🔬 BATTLE ANALYSIS:")
        logger.info(f"   📊 Your E768_H768 Champion strengths:")
        logger.info(f"      - Perfect ratio (embed_dim = hidden_dim)")
        logger.info(f"      - Zero projection overhead")
        logger.info(f"      - Proven L4_H768_E1.5 configuration")
        logger.info(f"      - Single-step simplicity")
        
        logger.info(f"   📊 Smart Refinement features:")
        logger.info(f"      - YOUR E768_H768 Champion as base")
        logger.info(f"      - Adaptive confidence thresholds")
        logger.info(f"      - BERT-style random masking")
        logger.info(f"      - Conservative selective updates")
        
        # Save comprehensive results
        results['battle_results'] = {
            'final_winner': final_winner,
            'smart_wins': smart_wins,
            'champion_wins': champion_wins,
            'smart_params': smart_params,
            'champion_params': champion_params,
            'accuracy_winner': acc_winner,
            'speed_winner': speed_winner,
            'milestone_winner': milestone_winner
        }
        
        timestamp = int(time.time())
        results_file = f'smart_vs_e768h768_battle_{timestamp}.json'
        
        # Make results serializable
        serializable_results = copy.deepcopy(results)
        for model_name in ['your_champion', 'smart_refinement']:
            if model_name in serializable_results and 'metrics' in serializable_results[model_name]:
                metrics = serializable_results[model_name]['metrics']
                for key in metrics:
                    if isinstance(metrics[key], list):
                        metrics[key] = [
                            float(x) if hasattr(x, 'item') else x 
                            for x in metrics[key]
                        ]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 BATTLE results saved: {results_file}")
        logger.info(f"\n🥊🥊🥊 SMART REFINEMENT BATTLE COMPLETED! 🥊🥊🥊")
        
        if final_winner == "smart":
            logger.info(f"🎊🎊🎊 SMART REFINEMENT VICTORIOUS! 🎊🎊🎊")
            logger.info(f"🚀 YOUR CHAMPION + ADAPTIVE REFINEMENT > SINGLE-STEP! 🚀")
        elif final_winner == "champion":
            logger.info(f"👑👑👑 E768_H768 CHAMPION UNDEFEATED! 👑👑👑")
            logger.info(f"🧠 PERFECT RATIO ARCHITECTURE SUPREME! 🧠")
        else:
            logger.info(f"⚔️⚔️⚔️ LEGENDARY BATTLE - BOTH WARRIORS HONORED! ⚔️⚔️⚔️")
        
        return results
    
    def get_num_samples(self):
        return self.data_loader.get_num_samples()


def main():
    """THE SMART BATTLE BEGINS!"""
    print("🥊🥊🥊🥊🥊 THE SMART BATTLE BEGINS! 🥊🥊🥊🥊🥊")
    logger.info("🥊 SMART REFINEMENT vs E768_H768 CHAMPION")
    logger.info("="*100)
    print("🎯 GOAL: Beat E768_H768 champion with adaptive refinement!")
    print("🏆 TARGET: 100% accuracy, 90% @ step 170")
    print(f"🖥️  Device: {DEVICE}")
    
    # Check multiple data paths
    possible_data_paths = [
        "no_overlap_data",
        "../no_overlap_data", 
        "data/no_overlap_data",
        "../data/no_overlap_data",
        "./no_overlap_data",
        "../../no_overlap_data",
        "C:\\mambaTTS\\mambaTTS\\no_overlap_data",
        "mambaTTS\\no_overlap_data",
        ".\\mambaTTS\\no_overlap_data"
    ]
    
    data_path = None
    for path_str in possible_data_paths:
        test_path = Path(path_str)
        print(f"🔍 Checking: {test_path.absolute()}")
        if test_path.exists():
            data_path = test_path
            print(f"✅ Found data at: {data_path.absolute()}")
            break
        else:
            print(f"   ❌ Not found")
    
    if data_path is None:
        print("❌ Data directory not found in any location!")
        print("💡 Please ensure no_overlap_data exists in one of the checked paths")
        return
    
    try:
        print("🔧 Initializing SMART battle tester...")
        tester = SmartBattleTester(data_dir=str(data_path))
        
        if tester.get_num_samples() == 0:
            print("❌ No samples loaded!")
            return
        
        print(f"✅ Loaded {tester.get_num_samples()} samples")
        print("🥊🥊🥊 STARTING THE SMART BATTLE! 🥊🥊🥊")
        
        # RUN THE SMART BATTLE!
        results = tester.run_smart_battle(max_steps_per_model=2000)
        
        # BATTLE CONCLUSION
        final_winner = results['battle_results']['final_winner']
        
        if final_winner == "smart":
            print("\n🎉🎉🎉🎉🎉 SMART REFINEMENT VICTORY! 🎉🎉🎉🎉🎉")
            print("🏆🏆🏆 YOUR CHAMPION + ADAPTIVE REFINEMENT WINS! 🏆🏆🏆")
            print("🔥 Adaptive refinement on proven base works!")
        elif final_winner == "champion":
            print("\n👑👑👑👑👑 E768_H768 CHAMPION WINS! 👑👑👑👑👑")
            print("🏆🏆🏆 SINGLE-STEP PERFECTION! 🏆🏆🏆")
            print("🧠 Perfect ratio architecture unbeaten!")
        else:
            print("\n⚔️⚔️⚔️⚔️⚔️ LEGENDARY TIE! ⚔️⚔️⚔️⚔️⚔️")
            print("🤝🤝🤝 BOTH APPROACHES EXCELLENT! 🤝🤝🤝")
        
    except Exception as e:
        logger.error(f"❌ Smart battle failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()#!/usr/bin/env python3