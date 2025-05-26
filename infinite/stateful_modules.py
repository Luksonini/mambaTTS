#!/usr/bin/env python3
"""
Enhanced Stateful Modules - Complete State Management
====================================================
Full implementation of stateful Mamba with state persistence between chunks
Key features:
- StatefulMambaTextEncoder with complete state flow
- StatefulMambaAudioProcessor with sequential context
- StatefulDurationRegulator with temporal awareness
- Proper state serialization and transfer
- Clean integration with no-overlap training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logger = logging.getLogger(__name__)


# ============================================================================
# STATEFUL SSM CORE
# ============================================================================

class StatefulSSM(nn.Module):
    """
    Core Stateful State Space Model with proper state management
    """
    def __init__(self, dim, state_size=64, dt_rank=16):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.dt_rank = dt_rank
        
        # Core SSM parameters (simplified Mamba-style)
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)  # Input and gate
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        self.act = nn.SiLU()
        
        # State space parameters
        self.dt_proj = nn.Linear(dt_rank, dim, bias=True)
        self.A_log = nn.Parameter(torch.randn(dim, state_size))
        self.D = nn.Parameter(torch.randn(dim))
        
        # State projections
        self.B_proj = nn.Linear(dim, state_size, bias=False)
        self.C_proj = nn.Linear(dim, state_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(dim)
        
        # Initialize for stability
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        nn.init.zeros_(self.D)
        
        logger.debug(f"StatefulSSM: dim={dim}, state_size={state_size}")
    
    def forward(self, x, initial_state=None, return_state=False):
        """
        Stateful SSM forward pass
        
        Args:
            x: [B, T, dim] - input sequence
            initial_state: [B, state_size] - initial state from previous chunk
            return_state: bool - whether to return final state
            
        Returns:
            output: [B, T, dim] - processed sequence
            final_state: [B, state_size] - final state (if return_state=True)
        """
        B, T, D = x.shape
        device = x.device
        
        # Input projection
        x_and_res = self.in_proj(x)  # [B, T, 2*dim]
        x_input, x_res = x_and_res.chunk(2, dim=-1)  # Each [B, T, dim]
        
        # Convolution (temporal processing)
        x_conv = self.conv1d(x_input.transpose(1, 2))[:, :, :T].transpose(1, 2)  # [B, T, dim]
        x_conv = self.act(x_conv)
        
        # State space parameters
        A = -torch.exp(self.A_log.float())  # [dim, state_size]
        B = self.B_proj(x_conv)  # [B, T, state_size]
        C = self.C_proj(x_conv)  # [B, T, state_size]
        
        # Delta (time step)
        dt = F.softplus(self.dt_proj.weight).view(1, 1, -1)  # [1, 1, dim]
        
        # Initialize state
        if initial_state is not None:
            h = initial_state.clone()  # [B, state_size]
        else:
            h = torch.zeros(B, self.state_size, device=device, dtype=x.dtype)
        
        outputs = []
        
        # Process sequence step by step (for proper state evolution)
        for t in range(T):
            # Get current inputs
            u_t = x_conv[:, t, :]  # [B, dim]
            B_t = B[:, t, :]       # [B, state_size] 
            C_t = C[:, t, :]       # [B, state_size]
            
            # State evolution: h = A*h + B*u
            # Discretize: h_new = (I + dt*A)*h + dt*B*u
            A_discrete = torch.eye(self.state_size, device=device, dtype=x.dtype) + dt.squeeze(-1).unsqueeze(-1) * A.T  # [state_size, state_size]
            
            # Update state
            h = torch.matmul(h, A_discrete.T) + dt.squeeze() * (B_t * u_t.unsqueeze(-1)).sum(-1)  # [B, state_size]
            
            # Output: y = C*h + D*u
            y_t = torch.sum(C_t * h, dim=-1) + self.D * u_t  # [B, dim]
            outputs.append(y_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [B, T, dim]
        
        # Gating and residual
        y = y * self.act(x_res)
        y = self.out_proj(y)
        
        # Residual connection with normalization
        output = self.norm(y + x)
        
        if return_state:
            return output, h
        else:
            return output


# ============================================================================
# STATEFUL TEXT ENCODER
# ============================================================================

class StatefulMambaTextEncoder(nn.Module):
    """
    Stateful Text Encoder with complete state management
    """
    def __init__(self, vocab_size, embed_dim=128, state_size=64, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_size = state_size
        self.num_layers = num_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(2000, embed_dim)  # Support long sequences
        
        # Stateful Mamba layers
        self.stateful_layers = nn.ModuleList([
            StatefulSSM(embed_dim, state_size) for _ in range(num_layers)
        ])
        
        # Multi-scale convolutions for pattern recognition
        self.context_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=8, padding=8),
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
        # Global attention for context extraction
        self.global_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        logger.info(f"🧠 StatefulMambaTextEncoder: {embed_dim}D, {num_layers} layers, {state_size} state_size")
    
    def forward(self, tokens, initial_states=None, return_states=False, return_sequence=True, position_offset=0):
        """
        Forward pass with complete state management
        
        Args:
            tokens: [B, T] - input tokens
            initial_states: List of [B, state_size] - states from previous chunk
            return_states: bool - whether to return final states
            return_sequence: bool - return full sequence vs global context
            position_offset: int - position offset for continuous sequences
            
        Returns:
            features: [B, T, embed_dim] or [B, embed_dim] - text features
            final_states: List of [B, state_size] - final states (if return_states=True)
        """
        B, T = tokens.shape
        device = tokens.device
        
        # Token embedding
        x = self.token_embedding(tokens)  # [B, T, embed_dim]
        
        # Positional embedding (continuous across chunks)
        positions = torch.arange(position_offset, position_offset + T, device=device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        
        # === STATEFUL MAMBA PROCESSING ===
        if initial_states is None:
            initial_states = [None] * self.num_layers
        
        final_states = []
        stateful_features = x
        
        for i, layer in enumerate(self.stateful_layers):
            if return_states:
                stateful_features, final_state = layer(
                    stateful_features, 
                    initial_states[i], 
                    return_state=True
                )
                final_states.append(final_state)
            else:
                stateful_features = layer(stateful_features, initial_states[i])
        
        # === CONTEXT CONVOLUTIONS ===
        x_conv = x.transpose(1, 2)  # [B, embed_dim, T]
        
        context_features = []
        for conv_layer in self.context_convs:
            context_feat = conv_layer(x_conv)  # [B, embed_dim//4, T]
            context_features.append(context_feat)
        
        context_combined = torch.cat(context_features, dim=1)  # [B, embed_dim, T]
        context_features = context_combined.transpose(1, 2)    # [B, T, embed_dim]
        
        # === FUSION ===
        combined = torch.cat([stateful_features, context_features], dim=-1)  # [B, T, embed_dim*2]
        token_embeddings = self.fusion(combined)  # [B, T, embed_dim]
        
        if return_sequence:
            if return_states:
                return token_embeddings, final_states
            else:
                return token_embeddings
        else:
            # Global context via attention
            attention_scores = self.global_attention(token_embeddings).squeeze(-1)  # [B, T]
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(1)  # [B, 1, T]
            global_context = torch.bmm(attention_weights, token_embeddings).squeeze(1)  # [B, embed_dim]
            
            if return_states:
                return global_context, final_states
            else:
                return global_context


# ============================================================================
# STATEFUL AUDIO PROCESSOR
# ============================================================================

class StatefulMambaAudioProcessor(nn.Module):
    """
    Stateful Audio Processor with sequential context awareness
    """
    def __init__(self, hidden_dim=256, num_codebooks=4, codebook_size=1024, state_size=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.state_size = state_size
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim) for _ in range(num_codebooks)
        ])
        
        # Position embedding
        self.pos_embedding = nn.Embedding(3000, hidden_dim)  # Support very long sequences
        
        # Stateful processing layers
        self.stateful_layers = nn.ModuleList([
            StatefulSSM(hidden_dim, state_size) for _ in range(num_layers)
        ])
        
        # Context convolutions (audio-specific patterns)
        self.context_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=1, padding=1),   # Phoneme
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=3, padding=3),   # Syllable
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=9, padding=9),   # Word
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=27, padding=27), # Phrase
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size) for _ in range(num_codebooks)
        ])
        
        logger.info(f"🎵 StatefulMambaAudioProcessor: {hidden_dim}D, {num_layers} layers, {num_codebooks} codebooks")
    
    def forward(self, audio_tokens, text_context, initial_states=None, return_states=False, position_offset=0):
        """
        Forward pass with state management
        
        Args:
            audio_tokens: [B, C, T] - audio token sequence
            text_context: [B, hidden_dim] - text conditioning
            initial_states: List of [B, state_size] - states from previous chunk
            return_states: bool - whether to return final states
            position_offset: int - position offset for continuous sequences
            
        Returns:
            logits: [B, C, T, codebook_size] - output logits
            final_states: List of [B, state_size] - final states (if return_states=True)
        """
        B, C, T = audio_tokens.shape
        device = audio_tokens.device
        
        # === TOKEN EMBEDDING ===
        token_embeds = []
        for cb_idx in range(C):
            cb_tokens = audio_tokens[:, cb_idx, :]  # [B, T]
            cb_embed = self.token_embeddings[cb_idx](cb_tokens)  # [B, T, hidden_dim]
            token_embeds.append(cb_embed)
        
        # Average across codebooks
        audio_emb = torch.stack(token_embeds, dim=1).mean(dim=1)  # [B, T, hidden_dim]
        
        # Add positional embedding (continuous across chunks)
        positions = torch.arange(position_offset, position_offset + T, device=device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)
        audio_emb = audio_emb + pos_emb
        
        # Add text conditioning
        text_expanded = text_context.unsqueeze(1).expand(-1, T, -1)
        audio_emb = audio_emb + text_expanded
        
        # === STATEFUL PROCESSING ===
        if initial_states is None:
            initial_states = [None] * self.num_layers
        
        final_states = []
        stateful_features = audio_emb
        
        for i, layer in enumerate(self.stateful_layers):
            if return_states:
                stateful_features, final_state = layer(
                    stateful_features,
                    initial_states[i],
                    return_state=True
                )
                final_states.append(final_state)
            else:
                stateful_features = layer(stateful_features, initial_states[i])
        
        # === CONTEXT CONVOLUTIONS ===
        audio_conv = audio_emb.transpose(1, 2)  # [B, hidden_dim, T]
        
        context_features = []
        for conv_layer in self.context_convs:
            context_feat = conv_layer(audio_conv)  # [B, hidden_dim//4, T]
            context_features.append(context_feat)
        
        context_combined = torch.cat(context_features, dim=1)  # [B, hidden_dim, T]
        context_features = context_combined.transpose(1, 2)    # [B, T, hidden_dim]
        
        # === FUSION ===
        combined = torch.cat([stateful_features, context_features], dim=-1)  # [B, T, hidden_dim*2]
        processed_features = self.fusion(combined)  # [B, T, hidden_dim]
        
        # === OUTPUT GENERATION ===
        outputs = []
        for cb_idx in range(self.num_codebooks):
            cb_output = self.output_heads[cb_idx](processed_features)  # [B, T, codebook_size]
            outputs.append(cb_output)
        
        logits = torch.stack(outputs, dim=1)  # [B, C, T, codebook_size]
        
        if return_states:
            return logits, final_states
        else:
            return logits


# ============================================================================
# STATEFUL DURATION REGULATOR
# ============================================================================

class StatefulDurationRegulator(nn.Module):
    """
    Stateful Duration Regulator with temporal awareness
    """
    def __init__(self, text_dim=128, style_dim=64, hidden_dim=128, 
                 tokens_per_second=75.0, state_size=64):
        super().__init__()
        self.text_dim = text_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.tokens_per_second = tokens_per_second
        self.state_size = state_size
        
        # FiLM conditioning
        self.style_to_film = nn.Sequential(
            nn.Linear(style_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, text_dim, bias=False)
        )
        
        # Input projection
        self.input_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        
        # Stateful processing
        self.stateful_layer = StatefulSSM(hidden_dim, state_size)
        
        # Multi-scale convolutions for duration patterns
        self.duration_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=1, padding=1, bias=False),
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=4, padding=4, bias=False),
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=8, padding=8, bias=False),
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
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
        
        logger.info(f"🎯 StatefulDurationRegulator: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, text_features, style_embedding, initial_state=None, return_state=False):
        """
        Forward pass with state management
        
        Args:
            text_features: [B, T, text_dim]
            style_embedding: [B, style_dim]
            initial_state: [B, state_size] - state from previous chunk
            return_state: bool - whether to return final state
            
        Returns:
            regulated_features: [B, T_regulated, text_dim]
            duration_seconds: [B, T]
            duration_tokens: [B, T] 
            confidence: [B, T]
            final_state: [B, state_size] (if return_state=True)
        """
        B, T_text, _ = text_features.shape
        device = text_features.device
        
        # FiLM modulation
        film_modulation = self.style_to_film(style_embedding)
        film_modulation = film_modulation.unsqueeze(1).expand(-1, T_text, -1)
        modulated_features = text_features * (1.0 + film_modulation)
        
        # Input projection
        x = self.input_proj(modulated_features)
        
        # Stateful processing
        if return_state:
            stateful_features, final_state = self.stateful_layer(x, initial_state, return_state=True)
        else:
            stateful_features = self.stateful_layer(x, initial_state)
        
        # Multi-scale convolutions
        x_conv = x.transpose(1, 2)
        context_features = []
        for conv_layer in self.duration_convs:
            context_feat = conv_layer(x_conv)
            context_features.append(context_feat)
        
        context_combined = torch.cat(context_features, dim=1)
        context_features = context_combined.transpose(1, 2)
        
        # Fusion
        combined = torch.cat([stateful_features, context_features], dim=-1)
        fused_features = self.fusion(combined)
        
        # Predictions
        shared_repr = self.shared_backbone(fused_features)
        duration_seconds = self.duration_head(shared_repr).squeeze(-1)
        confidence = self.confidence_head(shared_repr).squeeze(-1)
        
        # Convert to tokens
        duration_tokens = (duration_seconds * self.tokens_per_second).round().long()
        duration_tokens = torch.clamp(duration_tokens, min=1, max=10)
        
        # Length regulation
        regulated_features = self._regulate_length(text_features, duration_tokens)
        
        if return_state:
            return regulated_features, duration_seconds, duration_tokens, confidence, final_state
        else:
            return regulated_features, duration_seconds, duration_tokens, confidence
    
    def _regulate_length(self, text_features, duration_tokens):
        """Length regulation with repeat_interleave"""
        batch_size, seq_len, feature_dim = text_features.shape
        device = text_features.device
        
        regulated_features = []
        
        for b in range(batch_size):
            batch_durations = duration_tokens[b]
            batch_features = text_features[b]
            
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
# ENHANCED AUDIO STYLE EXTRACTOR
# ============================================================================

class AudioStyleExtractor(nn.Module):
    """Enhanced style extractor with better audio understanding"""
    def __init__(self, audio_dim=256, style_dim=64):
        super().__init__()
        self.audio_dim = audio_dim
        self.style_dim = style_dim
        
        # Multi-scale feature extraction
        self.feature_convs = nn.ModuleList([
            nn.Conv1d(audio_dim, 64, kernel_size=3, padding=1),  # Local features
            nn.Conv1d(audio_dim, 64, kernel_size=5, padding=2),  # Medium features  
            nn.Conv1d(audio_dim, 64, kernel_size=7, padding=3),  # Global features
        ])
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(32)  # Fixed-size temporal features
        
        # Style projection
        self.style_proj = nn.Sequential(
            nn.Linear(64 * 3 * 32, 128),  # 3 scales * 64 features * 32 temporal
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, style_dim),
            nn.Tanh()
        )
        
        logger.info(f"🎨 AudioStyleExtractor: {audio_dim}D → {style_dim}D")
    
    def forward(self, audio_features):
        """Extract style from audio features"""
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        
        B, D, T = audio_features.shape
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.feature_convs:
            scale_feat = F.relu(conv(audio_features))  # [B, 64, T]
            pooled_feat = self.temporal_pool(scale_feat)  # [B, 64, 32]
            multi_scale_features.append(pooled_feat)
        
        # Concatenate all scales
        combined_features = torch.cat(multi_scale_features, dim=1)  # [B, 192, 32]
        flattened = combined_features.view(B, -1)  # [B, 192*32]
        
        # Style projection
        style_embedding = self.style_proj(flattened)
        
        return style_embedding


# ============================================================================
# STATE MANAGEMENT UTILITIES
# ============================================================================

def transfer_states(source_states: List[torch.Tensor], target_device: torch.device) -> List[torch.Tensor]:
    """Transfer states to target device"""
    if source_states is None:
        return None
    
    return [state.to(target_device) if state is not None else None for state in source_states]


def clone_states(states: List[torch.Tensor]) -> List[torch.Tensor]:
    """Deep clone states for safety"""
    if states is None:
        return None
    
    return [state.clone() if state is not None else None for state in states]


def states_to_dict(states: List[torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    """Convert state list to dictionary"""
    if states is None or not keys:
        return {}
    
    return {key: states[i] if i < len(states) and states[i] is not None else None 
            for i, key in enumerate(keys)}


def dict_to_states(state_dict: Dict[str, torch.Tensor], keys: List[str]) -> List[torch.Tensor]:
    """Convert state dictionary to list"""
    if not state_dict or not keys:
        return [None] * len(keys)
    
    return [state_dict.get(key) for key in keys]


# ============================================================================
# BATCH STATE MANAGER
# ============================================================================

class BatchStateManager:
    """Manages states across batches and chunks"""
    
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.state_cache = {}
        self.device_cache = {}
        
    def store_states(self, batch_ids: List[str], states: List[List[torch.Tensor]], 
                    component_names: List[str]):
        """Store states for multiple samples"""
        for i, batch_id in enumerate(batch_ids):
            if batch_id not in self.state_cache:
                self.state_cache[batch_id] = {}
            
            for j, component_name in enumerate(component_names):
                if j < len(states) and i < len(states[j]):
                    self.state_cache[batch_id][component_name] = states[j][i].clone()
    
    def retrieve_states(self, batch_ids: List[str], component_names: List[str], 
                       device: torch.device) -> Dict[str, List[torch.Tensor]]:
        """Retrieve states for batch processing"""
        batch_states = {name: [] for name in component_names}
        
        for batch_id in batch_ids:
            for component_name in component_names:
                if (batch_id in self.state_cache and 
                    component_name in self.state_cache[batch_id]):
                    state = self.state_cache[batch_id][component_name].to(device)
                    batch_states[component_name].append(state)
                else:
                    batch_states[component_name].append(None)
        
        return batch_states
    
    def clear_states(self, batch_ids: List[str] = None):
        """Clear states for specified batch IDs or all"""
        if batch_ids is None:
            self.state_cache.clear()
        else:
            for batch_id in batch_ids:
                if batch_id in self.state_cache:
                    del self.state_cache[batch_id]
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        total_tensors = 0
        total_elements = 0
        
        for batch_id, states in self.state_cache.items():
            for component_name, state in states.items():
                if state is not None:
                    total_tensors += 1
                    total_elements += state.numel()
        
        return {
            'total_cached_samples': len(self.state_cache),
            'total_tensors': total_tensors,
            'total_elements': total_elements,
            'estimated_memory_mb': total_elements * 4 / (1024 * 1024)  # Assuming float32
        }


# ============================================================================
# STREAMING INFERENCE MANAGER
# ============================================================================

class StreamingInferenceManager:
    """Manages streaming inference with state persistence"""
    
    def __init__(self, text_encoder, audio_processor, duration_regulator, 
                 style_extractor, chunk_size=50, overlap_size=5):
        self.text_encoder = text_encoder
        self.audio_processor = audio_processor
        self.duration_regulator = duration_regulator
        self.style_extractor = style_extractor
        
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # State management
        self.text_states = None
        self.audio_states = None
        self.duration_state = None
        
        # Position tracking
        self.text_position = 0
        self.audio_position = 0
        
        logger.info(f"🎬 StreamingInferenceManager initialized with chunk_size={chunk_size}")
    
    def reset_states(self):
        """Reset all states for new sequence"""
        self.text_states = None
        self.audio_states = None  
        self.duration_state = None
        self.text_position = 0
        self.audio_position = 0
        logger.debug("🔄 States reset for new sequence")
    
    def process_text_chunk(self, text_tokens, style_embedding):
        """Process a chunk of text tokens with state continuity"""
        device = text_tokens.device
        
        # Get text features with state management
        text_features, new_text_states = self.text_encoder(
            text_tokens,
            initial_states=self.text_states,
            return_states=True,
            position_offset=self.text_position
        )
        
        # Duration regulation with state
        if self.duration_state is not None:
            regulated_features, duration_secs, duration_tokens, confidence, new_duration_state = \
                self.duration_regulator(
                    text_features, 
                    style_embedding,
                    initial_state=self.duration_state,
                    return_state=True
                )
        else:
            regulated_features, duration_secs, duration_tokens, confidence, new_duration_state = \
                self.duration_regulator(
                    text_features,
                    style_embedding, 
                    return_state=True
                )
        
        # Update states
        self.text_states = new_text_states
        self.duration_state = new_duration_state
        self.text_position += text_tokens.shape[1]
        
        return {
            'text_features': text_features,
            'regulated_features': regulated_features,
            'duration_seconds': duration_secs,
            'duration_tokens': duration_tokens,
            'confidence': confidence
        }
    
    def process_audio_chunk(self, audio_tokens, text_context):
        """Process a chunk of audio tokens with state continuity"""
        device = audio_tokens.device
        
        # Process audio with state management
        audio_logits, new_audio_states = self.audio_processor(
            audio_tokens,
            text_context,
            initial_states=self.audio_states,
            return_states=True,
            position_offset=self.audio_position
        )
        
        # Update states
        self.audio_states = new_audio_states
        self.audio_position += audio_tokens.shape[2]
        
        return {
            'audio_logits': audio_logits,
            'predicted_tokens': torch.argmax(audio_logits, dim=-1)
        }
    
    def generate_streaming(self, text_tokens, reference_audio=None, max_audio_length=1000):
        """Generate audio with streaming processing"""
        device = text_tokens.device
        batch_size = text_tokens.shape[0]
        
        # Extract style from reference audio
        if reference_audio is not None:
            style_embedding = self.style_extractor(reference_audio)
        else:
            style_embedding = torch.randn(batch_size, 64, device=device)
        
        # Reset states for new generation
        self.reset_states()
        
        # Process text in chunks
        text_chunks = self._chunk_sequence(text_tokens, self.chunk_size, self.overlap_size)
        all_regulated_features = []
        
        logger.info(f"🎯 Processing {len(text_chunks)} text chunks")
        
        for i, text_chunk in enumerate(text_chunks):
            logger.debug(f"Processing text chunk {i+1}/{len(text_chunks)}")
            
            result = self.process_text_chunk(text_chunk, style_embedding)
            all_regulated_features.append(result['regulated_features'])
        
        # Concatenate all regulated features
        full_regulated_features = torch.cat(all_regulated_features, dim=1)
        logger.info(f"📝 Total regulated sequence length: {full_regulated_features.shape[1]}")
        
        # Generate audio autoregressively
        generated_tokens = []
        current_audio_tokens = torch.zeros(batch_size, 4, 1, device=device, dtype=torch.long)
        
        for step in range(max_audio_length):
            # Get text context for current position
            text_context_idx = min(step, full_regulated_features.shape[1] - 1)
            text_context = full_regulated_features[:, text_context_idx, :]
            
            # Process current audio chunk
            audio_result = self.process_audio_chunk(current_audio_tokens, text_context)
            next_tokens = audio_result['predicted_tokens'][:, :, -1:]  # Get last timestep
            
            generated_tokens.append(next_tokens)
            
            # Update current tokens for next iteration
            current_audio_tokens = next_tokens
            
            if step % 100 == 0:
                logger.debug(f"Generated {step} audio tokens")
        
        # Concatenate all generated tokens
        full_generated_tokens = torch.cat(generated_tokens, dim=2)
        
        logger.info(f"🎵 Generated audio sequence: {full_generated_tokens.shape}")
        return full_generated_tokens
    
    def _chunk_sequence(self, sequence, chunk_size, overlap_size):
        """Split sequence into overlapping chunks"""
        chunks = []
        seq_len = sequence.shape[1]
        start = 0
        
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            chunk = sequence[:, start:end]
            chunks.append(chunk)
            
            if end >= seq_len:
                break
                
            start = end - overlap_size
        
        return chunks


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_stateful_modules():
    """Test all stateful modules"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🧪 Testing stateful modules on {device}")
    
    # Test StatefulMambaTextEncoder
    text_encoder = StatefulMambaTextEncoder(vocab_size=131, embed_dim=128).to(device)
    tokens = torch.randint(0, 131, (2, 20), device=device)
    
    logger.info("Testing StatefulMambaTextEncoder...")
    features1, states1 = text_encoder(tokens, return_states=True)
    logger.info(f"  Features: {features1.shape}, States: {len(states1)}")
    
    # Test state continuation
    features2, states2 = text_encoder(tokens, initial_states=states1, return_states=True, 
                                     position_offset=20)
    logger.info(f"  Continued: {features2.shape}, New states: {len(states2)}")
    
    # Test StatefulMambaAudioProcessor  
    audio_processor = StatefulMambaAudioProcessor(hidden_dim=256).to(device)
    audio_tokens = torch.randint(0, 1024, (2, 4, 50), device=device)
    text_context = torch.randn(2, 256, device=device)
    
    logger.info("Testing StatefulMambaAudioProcessor...")
    logits1, audio_states1 = audio_processor(audio_tokens, text_context, return_states=True)
    logger.info(f"  Logits: {logits1.shape}, States: {len(audio_states1)}")
    
    # Test StatefulDurationRegulator
    duration_reg = StatefulDurationRegulator(text_dim=128, style_dim=64).to(device)
    text_features = torch.randn(2, 20, 128, device=device)
    style_emb = torch.randn(2, 64, device=device)
    
    logger.info("Testing StatefulDurationRegulator...")
    regulated, durations, tokens, conf, dur_state = duration_reg(
        text_features, style_emb, return_state=True)
    logger.info(f"  Regulated: {regulated.shape}, Durations: {durations.shape}")
    logger.info(f"  Duration state: {dur_state.shape if dur_state is not None else None}")
    
    logger.info("✅ All stateful modules working correctly!")


def test_streaming_inference():
    """Test streaming inference functionality"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🎬 Testing streaming inference on {device}")
    
    # Initialize components
    text_encoder = StatefulMambaTextEncoder(vocab_size=131, embed_dim=128).to(device)
    audio_processor = StatefulMambaAudioProcessor(hidden_dim=256).to(device)
    duration_reg = StatefulDurationRegulator(text_dim=128, style_dim=64).to(device)
    style_extractor = AudioStyleExtractor(audio_dim=256, style_dim=64).to(device)
    
    # Create streaming manager
    streaming_manager = StreamingInferenceManager(
        text_encoder, audio_processor, duration_reg, style_extractor,
        chunk_size=25, overlap_size=5
    )
    
    # Test data
    text_tokens = torch.randint(0, 131, (1, 100), device=device)  # Long sequence
    reference_audio = torch.randn(1, 256, 200, device=device)
    
    logger.info("Testing streaming generation...")
    generated_audio = streaming_manager.generate_streaming(
        text_tokens, reference_audio, max_audio_length=150
    )
    
    logger.info(f"✅ Streaming generation successful: {generated_audio.shape}")


def test_batch_state_manager():
    """Test batch state management"""
    logger.info("🗂️ Testing BatchStateManager...")
    
    state_manager = BatchStateManager(max_batch_size=4)
    
    # Create dummy states
    batch_ids = ['sample_1', 'sample_2', 'sample_3']
    component_names = ['text_encoder', 'audio_processor', 'duration_regulator']
    
    dummy_states = [
        [torch.randn(64), torch.randn(64), torch.randn(64)],  # text_encoder states
        [torch.randn(64), torch.randn(64), torch.randn(64)],  # audio_processor states  
        [torch.randn(64), torch.randn(64), torch.randn(64)]   # duration_regulator states
    ]
    
    # Store states
    state_manager.store_states(batch_ids, dummy_states, component_names)
    
    # Retrieve states
    retrieved_states = state_manager.retrieve_states(
        batch_ids, component_names, torch.device('cpu')
    )
    
    logger.info(f"  Stored and retrieved states for {len(batch_ids)} samples")
    
    # Check memory usage
    memory_info = state_manager.get_memory_usage()
    logger.info(f"  Memory usage: {memory_info}")
    
    # Clear states
    state_manager.clear_states(['sample_1'])
    memory_info_after = state_manager.get_memory_usage()
    logger.info(f"  Memory after cleanup: {memory_info_after}")
    
    logger.info("✅ BatchStateManager working correctly!")


def benchmark_stateful_vs_stateless():
    """Benchmark stateful vs stateless processing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"⚡ Benchmarking stateful vs stateless on {device}")
    
    # Setup
    text_encoder = StatefulMambaTextEncoder(vocab_size=131, embed_dim=128).to(device)
    long_sequence = torch.randint(0, 131, (1, 200), device=device)
    
    import time
    
    # Stateless processing (process entire sequence at once)
    start_time = time.time()
    with torch.no_grad():
        stateless_output = text_encoder(long_sequence)
    stateless_time = time.time() - start_time
    
    # Stateful processing (process in chunks)
    chunk_size = 50
    chunks = [long_sequence[:, i:i+chunk_size] for i in range(0, 200, chunk_size)]
    
    start_time = time.time()
    with torch.no_grad():
        states = None
        stateful_outputs = []
        
        for i, chunk in enumerate(chunks):
            if states is None:
                output, states = text_encoder(chunk, return_states=True, position_offset=i*chunk_size)
            else:
                output, states = text_encoder(chunk, initial_states=states, 
                                            return_states=True, position_offset=i*chunk_size)
            stateful_outputs.append(output)
        
        stateful_output = torch.cat(stateful_outputs, dim=1)
    
    stateful_time = time.time() - start_time
    
    logger.info(f"  Stateless time: {stateless_time:.4f}s")
    logger.info(f"  Stateful time: {stateful_time:.4f}s")
    logger.info(f"  Output shapes - Stateless: {stateless_output.shape}, Stateful: {stateful_output.shape}")
    
    # Check similarity (should be similar but not identical due to positional differences)
    similarity = torch.cosine_similarity(
        stateless_output.flatten(), 
        stateful_output.flatten(), 
        dim=0
    )
    logger.info(f"  Output similarity: {similarity:.4f}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("🚀 Starting Enhanced Stateful Modules Tests")
        
        # Run all tests
        test_stateful_modules()
        test_streaming_inference() 
        test_batch_state_manager()
        benchmark_stateful_vs_stateless()
        
        logger.info("🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise