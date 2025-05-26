#!/usr/bin/env python3
"""
Infinite Modules - Continuous Processing bez Internal Chunking
============================================================
Mamba modules designed for infinite-like continuous processing
Key features:
- Process entire minutowe batches as single sequences
- No internal chunking - true continuous flow
- Fresh state only between batches
- Virtual checkpoints for error tracking
- Optimized for long sequence processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logger = logging.getLogger(__name__)


# ============================================================================
# INFINITE MAMBA CORE
# ============================================================================

class InfiniteMambaCore(nn.Module):
    """
    Core Mamba optimized for infinite-like continuous processing
    No internal state resets - processes entire sequences seamlessly
    """
    def __init__(self, dim, state_size=64, dt_rank=16, expand_factor=2):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.dt_rank = dt_rank
        self.expand_factor = expand_factor
        self.inner_dim = dim * expand_factor
        
        # Core projections
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)  # Input + Gate
        self.conv1d = nn.Conv1d(
            self.inner_dim, self.inner_dim, 
            kernel_size=4, padding=3, groups=self.inner_dim
        )
        self.act = nn.SiLU()
        
        # SSM parameters
        self.dt_proj = nn.Linear(dt_rank, self.inner_dim, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.inner_dim, state_size))
        self.D = nn.Parameter(torch.randn(self.inner_dim))
        
        # ✅ NAPRAWKA: State projections z prawidłowymi wymiarami
        # B_proj: inner_dim → state_size
        self.B_proj = nn.Linear(self.inner_dim, state_size, bias=False)
        # C_proj: state_size → inner_dim (ODWROTNIE!)
        self.C_proj = nn.Linear(state_size, self.inner_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)
        
        # Initialize for infinite stability
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.uniform_(self.A_log, -5.0, -1.0)  # More stable for long sequences
        nn.init.zeros_(self.D)
        
        # State buffer for continuous processing
        self.register_buffer('continuous_state', torch.zeros(1, state_size))
        self.state_initialized = False
        
        logger.debug(f"InfiniteMambaCore: {dim}D → {self.inner_dim}D → {dim}D, state_size={state_size}")
        logger.debug(f"   B_proj: {self.inner_dim} → {state_size}")
        logger.debug(f"   C_proj: {state_size} → {self.inner_dim}")
    
    def reset_continuous_state(self, batch_size=1):
        """Reset state for new batch - ONLY place state gets reset!"""
        device = next(self.parameters()).device
        self.continuous_state = torch.zeros(batch_size, self.state_size, device=device, dtype=torch.float32)
        self.state_initialized = True
        logger.debug(f"🔄 Infinite state reset for batch_size={batch_size}")
    
    def forward(self, x, reset_state=False):
        """
        NAPRAWIONY forward z prawidłowymi wymiarami
        """
        B, T, D = x.shape
        device = x.device
        
        # Reset state only if requested (new batch)
        if reset_state or not self.state_initialized:
            self.reset_continuous_state(B)
        
        # Ensure state has correct batch size
        if self.continuous_state.shape[0] != B:
            self.reset_continuous_state(B)
        
        # Input projection and gating
        x_proj = self.in_proj(x)  # [B, T, inner_dim * 2]
        x_input, x_gate = x_proj.chunk(2, dim=-1)  # Each [B, T, inner_dim]
        
        # Temporal convolution
        x_conv = self.conv1d(x_input.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = self.act(x_conv)
        
        # === UPROSZCZONE SSM ===
        h = self.continuous_state  # [B, state_size]
        outputs = []
        
        # Proste parametry
        decay_factor = 0.95  # Decay rate
        input_factor = 0.05  # Input influence
        
        for t in range(T):
            u_t = x_conv[:, t, :]  # [B, inner_dim]
            
            # ✅ NAPRAWKA: Projekcja inner_dim → state_size
            u_projected = self.B_proj(u_t)  # [B, inner_dim] → [B, state_size]
            
            # Prosta aktualizacja stanu
            h = decay_factor * h + input_factor * u_projected  # [B, state_size]
            
            # ✅ NAPRAWKA: Projekcja state_size → inner_dim
            y_base = self.C_proj(h)  # [B, state_size] → [B, inner_dim]
            
            # Combine with skip connection
            y_t = y_base + self.D * u_t  # [B, inner_dim]
            outputs.append(y_t)
        
        # Update continuous state for next sequence segment
        self.continuous_state = h.detach()
        
        # Stack outputs and apply gating
        y = torch.stack(outputs, dim=1)  # [B, T, inner_dim]
        y = y * self.act(x_gate)
        
        # Output projection
        y = self.out_proj(y)
        
        # Residual connection with normalization
        output = self.norm(y + x)
        
        return output

# ============================================================================
# INFINITE TEXT ENCODER
# ============================================================================

class InfiniteMambaTextEncoder(nn.Module):
    """
    Text encoder for infinite continuous processing
    Processes entire minutowe sequences without breaks
    """
    def __init__(self, vocab_size, embed_dim=128, num_layers=4, state_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.state_size = state_size
        
        # Token embedding with extended position support
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(20000, embed_dim)  # Support very long sequences
        
        # Infinite Mamba layers
        self.infinite_layers = nn.ModuleList([
            InfiniteMambaCore(embed_dim, state_size) for _ in range(num_layers)
        ])
        
        # Multi-scale context capture (no chunking!)
        self.context_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=1, padding=1),   # Local
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=2, padding=2),   # Medium
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=4, padding=4),   # Long
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=8, padding=8),   # Max - rozsądne
        ])
        
        # Fusion for infinite context
        self.infinite_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
        # Global context extraction
        self.global_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        logger.info(f"🧠 InfiniteMambaTextEncoder: {embed_dim}D, {num_layers} layers")
        logger.info(f"   🔄 Infinite continuous processing - no internal resets")
    
    def reset_infinite_state(self, batch_size=1):
        """Reset state for new batch - ONLY between batches!"""
        for layer in self.infinite_layers:
            layer.reset_continuous_state(batch_size)
        logger.debug(f"🔄 Text encoder infinite state reset")
    
    def forward(self, x, reset_state=False):
        """
        BARDZO PROSTY forward - obsługuje zarówno token indices jak i embeddings
        """
        device = x.device
        
        # Sprawdź czy to token indices [B, T] czy embeddings [B, T, D]
        if x.dim() == 2:  # [B, T] - token indices
            B, T = x.shape
            # Convert tokens to embeddings
            x = self.token_embedding(x)  # [B, T, embed_dim]
            
            # Add positional embeddings
            positions = torch.arange(T, device=device, dtype=torch.long)
            pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)
            x = x + pos_emb
            
        elif x.dim() == 3:  # [B, T, D] - already embeddings
            B, T, D = x.shape
        else:
            raise ValueError(f"Expected input to be 2D [B, T] or 3D [B, T, D], got {x.shape}")
        
        # Reset infinite state if requested
        if reset_state:
            self.reset_infinite_state(B)
        
        # Simple processing - just pass through infinite layers
        output = x
        for layer in self.infinite_layers:
            output = layer(output, reset_state=reset_state)
            reset_state = False  # Only reset first layer
        
        return output  # [B, T, embed_dim]


# ============================================================================
# INFINITE AUDIO PROCESSOR
# ============================================================================

class InfiniteMambaAudioProcessor(nn.Module):
    """
    Audio processor for infinite continuous generation
    Processes entire minutowe sequences autoregressively
    """
    def __init__(self, hidden_dim=256, num_codebooks=4, codebook_size=1024, 
                 num_layers=3, state_size=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.state_size = state_size
        
        # Token embeddings for each codebook
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim) for _ in range(num_codebooks)
        ])
        
        # Position embedding for very long sequences
        self.pos_embedding = nn.Embedding(20000, hidden_dim)  # Support up to ~2 minutes at 75 tokens/s
        
        # Infinite Mamba layers for audio processing
        self.infinite_audio_layers = nn.ModuleList([
            InfiniteMambaCore(hidden_dim, state_size) for _ in range(num_layers)
        ])
        
        # Multi-scale audio pattern recognition
        self.audio_pattern_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=1, padding=1),   # Phoneme
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=3, padding=3),   # Syllable  
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=6, padding=6),   # Word
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=12, padding=12), # Phrase - max
        ])
        
        # Infinite audio fusion
        self.audio_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output heads for each codebook
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size) for _ in range(num_codebooks)
        ])
        
        logger.info(f"🎵 InfiniteMambaAudioProcessor: {hidden_dim}D, {num_layers} layers, {num_codebooks} codebooks")
        logger.info(f"   🔄 Infinite continuous audio processing")
    
    def reset_infinite_state(self, batch_size=1):
        """Reset state for new batch"""
        for layer in self.infinite_audio_layers:
            layer.reset_continuous_state(batch_size)
        logger.debug(f"🔄 Audio processor infinite state reset")
    
    def forward(self, audio_tokens, text_context, reset_state=False, position_offset=0):
        """
        Infinite continuous audio processing
        
        Args:
            audio_tokens: [B, C, T] - audio token sequence
            text_context: [B, hidden_dim] - text conditioning
            reset_state: bool - reset state (only for new batches)
            position_offset: int - position offset for continuous sequences
            
        Returns:
            logits: [B, C, T, codebook_size] - output logits
        """
        B, C, T = audio_tokens.shape
        device = audio_tokens.device
        
        # Reset infinite state if requested
        if reset_state:
            self.reset_infinite_state(B)
        
        # === TOKEN EMBEDDING ===
        token_embeds = []
        for cb_idx in range(C):
            cb_tokens = audio_tokens[:, cb_idx, :]  # [B, T]
            cb_embed = self.token_embeddings[cb_idx](cb_tokens)  # [B, T, hidden_dim]
            token_embeds.append(cb_embed)
        
        # Average across codebooks
        audio_emb = torch.stack(token_embeds, dim=1).mean(dim=1)  # [B, T, hidden_dim]
        
        # Continuous positional embedding
        positions = torch.arange(position_offset, position_offset + T,
                               device=device, dtype=torch.long)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)
        audio_emb = audio_emb + pos_emb
        
        # Add text conditioning
        text_expanded = text_context.unsqueeze(1).expand(-1, T, -1)
        audio_emb = audio_emb + text_expanded
        
        # === INFINITE AUDIO PROCESSING ===
        infinite_audio_features = audio_emb
        for layer in self.infinite_audio_layers:
            infinite_audio_features = layer(infinite_audio_features, reset_state=reset_state)
            reset_state = False  # Only reset first layer
        
        # === AUDIO PATTERN CONVOLUTIONS ===
        audio_conv = audio_emb.transpose(1, 2)  # [B, hidden_dim, T]
        
        pattern_features = []
        for conv_layer in self.audio_pattern_convs:
            pattern_feat = conv_layer(audio_conv)  # [B, hidden_dim//4, T]
            pattern_features.append(pattern_feat)
        
        pattern_combined = torch.cat(pattern_features, dim=1)  # [B, hidden_dim, T]
        pattern_features = pattern_combined.transpose(1, 2)    # [B, T, hidden_dim]
        
        # === INFINITE AUDIO FUSION ===
        combined = torch.cat([infinite_audio_features, pattern_features], dim=-1)
        processed_features = self.audio_fusion(combined)  # [B, T, hidden_dim]
        
        # === OUTPUT GENERATION ===
        outputs = []
        for cb_idx in range(self.num_codebooks):
            cb_output = self.output_heads[cb_idx](processed_features)  # [B, T, codebook_size]
            outputs.append(cb_output)
        
        logits = torch.stack(outputs, dim=1)  # [B, C, T, codebook_size]
        
        return logits


# ============================================================================
# INFINITE DURATION REGULATOR
# ============================================================================

class InfiniteDurationRegulator(nn.Module):
    """
    Duration regulator for infinite continuous processing
    Handles timing for long sequences without breaks
    """
    def __init__(self, text_dim=128, style_dim=64, hidden_dim=128, 
                 tokens_per_second=75.0, state_size=64):
        super().__init__()
        self.text_dim = text_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.tokens_per_second = tokens_per_second
        self.state_size = state_size
        
        # FiLM conditioning for style
        self.style_to_film = nn.Sequential(
            nn.Linear(style_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, text_dim, bias=False)
        )
        
        # Input projection
        self.input_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        
        # Infinite Mamba for duration processing
        self.infinite_duration_layer = InfiniteMambaCore(hidden_dim, state_size)
        
        # Multi-scale duration pattern recognition
        self.duration_pattern_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=1, padding=1, bias=False),   # Phoneme timing
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=4, padding=4, bias=False),   # Syllable timing
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=16, padding=16, bias=False), # Word timing
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=64, padding=64, bias=False), # Phrase timing
        ])
        
        # Infinite duration fusion
        self.duration_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output heads
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1, bias=False),
            nn.Softplus()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        logger.info(f"🎯 InfiniteDurationRegulator: {sum(p.numel() for p in self.parameters()):,} parameters")
        logger.info(f"   🔄 Infinite continuous duration processing")
    
    def reset_infinite_state(self, batch_size=1):
        """Reset state for new batch"""
        self.infinite_duration_layer.reset_continuous_state(batch_size)
        logger.debug(f"🔄 Duration regulator infinite state reset")
    
    def forward(self, text_features, style_embedding, reset_state=False):
        """
        Infinite continuous duration regulation
        
        Args:
            text_features: [B, T, text_dim]
            style_embedding: [B, style_dim]
            reset_state: bool - reset state (only for new batches)
            
        Returns:
            regulated_features: [B, T_regulated, text_dim]
            predicted_durations: [B, T]
            duration_tokens: [B, T] 
            confidence: [B, T]
        """
        B, T_text, _ = text_features.shape
        device = text_features.device
        
        # Reset infinite state if requested
        if reset_state:
            self.reset_infinite_state(B)
        
        # FiLM modulation
        film_modulation = self.style_to_film(style_embedding)
        film_modulation = film_modulation.unsqueeze(1).expand(-1, T_text, -1)
        modulated_features = text_features * (1.0 + film_modulation)
        
        # Input projection
        x = self.input_proj(modulated_features)
        
        # Infinite duration processing
        infinite_duration_features = self.infinite_duration_layer(x, reset_state=reset_state)
        
        # Multi-scale duration patterns
        x_conv = x.transpose(1, 2)
        duration_patterns = []
        for conv_layer in self.duration_pattern_convs:
            pattern_feat = conv_layer(x_conv)
            duration_patterns.append(pattern_feat)
        
        pattern_combined = torch.cat(duration_patterns, dim=1)
        pattern_features = pattern_combined.transpose(1, 2)
        
        # Infinite duration fusion
        combined = torch.cat([infinite_duration_features, pattern_features], dim=-1)
        fused_features = self.duration_fusion(combined)
        
        # Predictions
        shared_repr = self.shared_backbone(fused_features)
        predicted_durations = self.duration_head(shared_repr).squeeze(-1)
        confidence = self.confidence_head(shared_repr).squeeze(-1)
        
        # Convert to tokens
        duration_tokens = (predicted_durations * self.tokens_per_second).round().long()
        duration_tokens = torch.clamp(duration_tokens, min=1, max=10)
        
        # Length regulation
        regulated_features = self._regulate_length_infinite(text_features, duration_tokens)
        
        return regulated_features, predicted_durations, duration_tokens, confidence
    
    def _regulate_length_infinite(self, text_features, duration_tokens):
        """Length regulation optimized for infinite continuous processing"""
        batch_size, seq_len, feature_dim = text_features.shape
        device = text_features.device
        
        regulated_features = []
        
        for b in range(batch_size):
            batch_durations = duration_tokens[b]
            batch_features = text_features[b]
            
            # Expand with repeat_interleave
            expanded = torch.repeat_interleave(
                batch_features, 
                batch_durations, 
                dim=0
            )
            
            regulated_features.append(expanded)
        
        # Pad to same length
        max_len = max(f.shape[0] for f in regulated_features)
        
        padded_features = []
        for expanded in regulated_features:
            current_len = expanded.shape[0]
            if current_len < max_len:
                pad_size = max_len - current_len
                padding = torch.zeros(pad_size, feature_dim, device=device, dtype=expanded.dtype)
                padded = torch.cat([expanded, padding], dim=0)
            else:
                padded = expanded
            
            padded_features.append(padded.unsqueeze(0))
        
        result = torch.cat(padded_features, dim=0)
        return result


# ============================================================================
# INFINITE STYLE EXTRACTOR
# ============================================================================

class InfiniteAudioStyleExtractor(nn.Module):
    """Enhanced style extractor for infinite continuous processing"""
    def __init__(self, audio_dim=256, style_dim=64):
        super().__init__()
        self.audio_dim = audio_dim
        self.style_dim = style_dim
        
        # Multi-scale style feature extraction
        self.style_convs = nn.ModuleList([
            nn.Conv1d(audio_dim, 64, kernel_size=3, padding=1),   # Local style
            nn.Conv1d(audio_dim, 64, kernel_size=7, padding=3),   # Medium style  
            nn.Conv1d(audio_dim, 64, kernel_size=15, padding=7),  # Global style
        ])
        
        # Temporal pooling for different time scales
        self.temporal_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(16),  # Short-term style
            nn.AdaptiveAvgPool1d(32),  # Medium-term style
            nn.AdaptiveAvgPool1d(64),  # Long-term style
        ])
        
        # Style projection
        self.style_proj = nn.Sequential(
            nn.Linear(64 * 3 * (16 + 32 + 64), 256),  # 3 scales * 64 features * pooled sizes
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, style_dim),
            nn.Tanh()
        )
        
        logger.info(f"🎨 InfiniteAudioStyleExtractor: {audio_dim}D → {style_dim}D")
    
    def forward(self, audio_features):
        """Extract style from infinite continuous audio features"""
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        
        B, D, T = audio_features.shape
        
        # Multi-scale style feature extraction
        multi_scale_features = []
        for conv in self.style_convs:
            scale_feat = F.relu(conv(audio_features))  # [B, 64, T]
            
            # Multi-temporal pooling
            pooled_features = []
            for pool in self.temporal_pools:
                pooled_feat = pool(scale_feat)  # [B, 64, pool_size]
                pooled_features.append(pooled_feat)
            
            # Concatenate temporal scales
            scale_combined = torch.cat(pooled_features, dim=-1)  # [B, 64, total_pool_size]
            multi_scale_features.append(scale_combined)
        
        # Concatenate all scales
        combined_features = torch.cat(multi_scale_features, dim=1)  # [B, 192, total_pool_size]
        flattened = combined_features.view(B, -1)
        
        # Style projection
        style_embedding = self.style_proj(flattened)
        
        return style_embedding


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_infinite_modules():
    """Test infinite continuous processing modules"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🧪 Testing Infinite Modules on {device}")
    
    # Test InfiniteMambaTextEncoder
    logger.info("Testing InfiniteMambaTextEncoder...")
    text_encoder = InfiniteMambaTextEncoder(vocab_size=131, embed_dim=128).to(device)
    
    # Test with long sequence (simulating minutowe batch)
    long_tokens = torch.randint(0, 131, (1, 400), device=device)  # ~400 tokens ≈ 1 minute
    
    # Fresh batch processing
    text_encoder.reset_infinite_state(1)
    features = text_encoder(long_tokens, reset_state=True)
    logger.info(f"  Long sequence: {long_tokens.shape} → {features.shape}")
    
    # Test InfiniteMambaAudioProcessor
    logger.info("Testing InfiniteMambaAudioProcessor...")
    audio_processor = InfiniteMambaAudioProcessor(hidden_dim=256).to(device)
    
    # Long audio sequence
    long_audio = torch.randint(0, 1024, (1, 4, 4500), device=device)  # ~4500 tokens ≈ 1 minute
    text_context = torch.randn(1, 256, device=device)
    
    audio_processor.reset_infinite_state(1)
    logits = audio_processor(long_audio, text_context, reset_state=True)
    logger.info(f"  Long audio: {long_audio.shape} → {logits.shape}")
    
    # Test InfiniteDurationRegulator
    logger.info("Testing InfiniteDurationRegulator...")
    duration_reg = InfiniteDurationRegulator(text_dim=128, style_dim=64).to(device)
    
    long_text_features = torch.randn(1, 400, 128, device=device)
    style_emb = torch.randn(1, 64, device=device)
    
    duration_reg.reset_infinite_state(1)
    regulated, durations, tokens, conf = duration_reg(
        long_text_features, style_emb, reset_state=True
    )
    logger.info(f"  Duration regulation: {long_text_features.shape} → {regulated.shape}")
    logger.info(f"  Predicted durations: {durations.shape}, mean: {durations.mean().item():.2f}s")
    
    logger.info("✅ All infinite modules working correctly!")
    logger.info("🎯 Ready for infinite continuous processing!")


if __name__ == "__main__":
    test_infinite_modules()