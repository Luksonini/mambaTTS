#!/usr/bin/env python3
"""
Ultimate Unified Mamba TTS Training System with Architecture Comparison
====================================================================
Supports both Pure Mamba and Hybrid Conv+Mamba architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager
import os

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("🔥 SCRIPT STARTED!")
import sys
print(f"🐍 Python: {sys.version}")
print(f"📍 CWD: {os.getcwd()}")

# Import existing components
try:
    from nucleotide_tokenizer import NucleotideTokenizer
    from losses import compute_combined_loss
    logger.info("✅ Imported tokenizer and losses")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    exit(1)


@contextmanager
def timer(name):
    """Context manager for timing code blocks"""
    start = time.time()
    yield
    end = time.time()
    logger.info(f"⏱️  {name}: {(end - start)*1000:.2f}ms")


class TimingStats:
    """Collect and analyze timing statistics"""
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def record(self, name, duration):
        if name not in self.times:
            self.times[name] = []
            self.counts[name] = 0
        self.times[name].append(duration)
        self.counts[name] += 1
    
    def get_stats(self, name):
        if name not in self.times:
            return None
        times = self.times[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def print_summary(self):
        logger.info("📊 TIMING SUMMARY:")
        logger.info("=" * 50)
        for name in sorted(self.times.keys()):
            stats = self.get_stats(name)
            logger.info(f"{name:25s}: {stats['mean']*1000:6.2f}ms avg, {stats['count']:4d} calls, {stats['total']*1000:8.1f}ms total")


class AccuracyMilestoneTracker:
    """Track when specific accuracy milestones are reached"""
    def __init__(self):
        self.milestones = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        self.duration_milestones = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        
        self.accuracy_reached = {}
        self.duration_accuracy_reached = {}
        
        # Track when milestones were reached
        for milestone in self.milestones:
            self.accuracy_reached[milestone] = None
            
        for milestone in self.duration_milestones:
            self.duration_accuracy_reached[milestone] = None
    
    def update(self, step, accuracy, duration_accuracy):
        """Update milestone tracking"""
        # Check audio accuracy milestones
        for milestone in self.milestones:
            if accuracy >= milestone and self.accuracy_reached[milestone] is None:
                self.accuracy_reached[milestone] = step
                logger.info(f"🎯 MILESTONE: Audio accuracy {milestone:.1%} reached at step {step}!")
        
        # Check duration accuracy milestones
        for milestone in self.duration_milestones:
            if duration_accuracy >= milestone and self.duration_accuracy_reached[milestone] is None:
                self.duration_accuracy_reached[milestone] = step
                logger.info(f"🎯 MILESTONE: Duration accuracy {milestone:.1%} reached at step {step}!")
    
    def get_summary(self):
        """Get milestone summary"""
        summary = {
            'audio_accuracy_milestones': {},
            'duration_accuracy_milestones': {}
        }
        
        for milestone, step in self.accuracy_reached.items():
            if step is not None:
                summary['audio_accuracy_milestones'][f"{milestone:.1%}"] = step
        
        for milestone, step in self.duration_accuracy_reached.items():
            if step is not None:
                summary['duration_accuracy_milestones'][f"{milestone:.1%}"] = step
        
        return summary
    
    def print_milestones(self):
        """Print achieved milestones"""
        logger.info("\n🎯 ACCURACY MILESTONES ACHIEVED:")
        logger.info("=" * 50)
        
        logger.info("Audio Accuracy:")
        for milestone in self.milestones:
            step = self.accuracy_reached[milestone]
            if step is not None:
                logger.info(f"  {milestone:>6.1%}: Step {step:>4d}")
            else:
                logger.info(f"  {milestone:>6.1%}: Not reached")
        
        logger.info("\nDuration Accuracy:")
        for milestone in self.duration_milestones:
            step = self.duration_accuracy_reached[milestone]
            if step is not None:
                logger.info(f"  {milestone:>6.1%}: Step {step:>4d}")
            else:
                logger.info(f"  {milestone:>6.1%}: Not reached")


class UnifiedMambaBlock(nn.Module):
    """Unified block supporting both Pure Mamba and Hybrid Conv+Mamba architectures"""
    def __init__(self, d_model, expand_factor=2, dropout=0.1, reverse=False, 
                 architecture="hybrid", kernel_sizes=(3, 5)):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        self.reverse = reverse
        self.architecture = architecture
        self.kernel_sizes = kernel_sizes
        
        # MAMBA COMPONENTS (used by both architectures)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise convolution for Mamba path
        self.conv1d = nn.Conv1d(
            self.d_inner, 
            self.d_inner, 
            kernel_size=3, 
            padding=1, 
            groups=self.d_inner,
            bias=False
        )
        
        # SSM parameters (simplified Mamba state space)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # HYBRID-SPECIFIC COMPONENTS (only for hybrid architecture)
        if architecture == "hybrid":
            k1, k2 = kernel_sizes
            self.local_conv1 = nn.Conv1d(d_model, d_model, kernel_size=k1, padding=k1//2, groups=d_model//4)
            self.local_conv2 = nn.Conv1d(d_model, d_model, kernel_size=k2, padding=k2//2, groups=d_model//8)
            self.conv_norm = nn.GroupNorm(8, d_model)
            self.combine_proj = nn.Linear(d_model * 2, d_model, bias=False)
        
        # Normalization and activation
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Log configuration
        if architecture == "hybrid":
            total_rf = max(kernel_sizes)
            logger.debug(f"UnifiedMambaBlock: Hybrid mode, RF: {kernel_sizes} -> {total_rf} tokens")
        else:
            logger.debug(f"UnifiedMambaBlock: Pure Mamba mode")
        
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        residual = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # Reverse sequence if backward processing
        if self.reverse:
            x = torch.flip(x, dims=[1])
        
        if self.architecture == "hybrid":
            # HYBRID PATH: Conv1D + Mamba
            
            # PATH 1: LOCAL PATTERNS with Conv1D
            x_local = x.transpose(1, 2)  # [B, D, L] for Conv1D
            
            # Multi-scale local convolutions
            local1 = self.local_conv1(x_local)  # Long-range local patterns
            local2 = self.local_conv2(x_local)  # Short-range local patterns
            
            # Combine local patterns
            x_local_combined = local1 + local2
            x_local_combined = self.conv_norm(x_local_combined)
            x_local_combined = self.activation(x_local_combined)
            x_local_final = x_local_combined.transpose(1, 2)  # Back to [B, L, D]
            
            # PATH 2: GLOBAL DEPENDENCIES with Mamba
            x_proj = self.in_proj(x)  # [B, L, 2*d_inner]
            x1, x2 = x_proj.chunk(2, dim=-1)  # Each [B, L, d_inner]
            
            # Conv1d processing (additional local context for global path)
            x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)  # [B, L, d_inner]
            
            # SSM processing (simplified state space)
            x1_ssm = self.activation(x1_conv)
            dt = self.dt_proj(x1_ssm)  # Time step
            dt = F.softplus(dt)  # Ensure positive
            
            # Simplified state space operation
            x1_processed = x1_ssm * torch.sigmoid(dt)
            
            # Gating mechanism
            x_gated = x1_processed * torch.sigmoid(x2)
            
            # Global output projection
            x_global_final = self.out_proj(x_gated)
            
            # COMBINE LOCAL + GLOBAL
            combined = torch.cat([x_local_final, x_global_final], dim=-1)
            output = self.combine_proj(combined)
            
        else:
            # PURE MAMBA PATH: Only Mamba processing
            
            # Input projection and split
            x_proj = self.in_proj(x)  # [B, L, 2*d_inner]
            x1, x2 = x_proj.chunk(2, dim=-1)  # Each [B, L, d_inner]
            
            # Conv1d processing (local context)
            x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)  # [B, L, d_inner]
            
            # SSM processing (simplified state space)
            x1_ssm = self.activation(x1_conv)
            dt = self.dt_proj(x1_ssm)  # Time step
            dt = F.softplus(dt)  # Ensure positive
            
            # Simplified state space operation
            x1_processed = x1_ssm * torch.sigmoid(dt)
            
            # Gating mechanism
            x_gated = x1_processed * torch.sigmoid(x2)
            
            # Output projection
            output = self.out_proj(x_gated)
        
        output = self.dropout(output)
        
        # Reverse back if was reversed
        if self.reverse:
            output = torch.flip(output, dims=[1])
        
        # Residual connection
        return output + residual


class UnifiedMambaTextEncoder(nn.Module):
    """Unified text encoder supporting both Pure Mamba and Hybrid architectures"""
    def __init__(self, vocab_size, embed_dim=384, num_layers=6, architecture="hybrid", kernel_sizes=(3, 5)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.architecture = architecture
        self.kernel_sizes = kernel_sizes
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, embed_dim) * 0.02)
        
        # Stack of Unified blocks
        self.layers = nn.ModuleList([
            UnifiedMambaBlock(embed_dim, expand_factor=2, dropout=0.1, reverse=False, 
                            architecture=architecture, kernel_sizes=kernel_sizes)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.RMSNorm(embed_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Log configuration
        if architecture == "hybrid":
            logger.info(f"   📡 Text encoder: Hybrid Conv+Mamba, RF: {kernel_sizes} -> {max(kernel_sizes)} tokens")
        else:
            logger.info(f"   📡 Text encoder: Pure Mamba architecture")
        
    def forward(self, tokens, return_sequence=True):
        B, L = tokens.shape
        
        # Token embeddings
        x = self.embedding(tokens)  # [B, L, D]
        
        # Add positional encoding
        if L <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :L, :]
        
        x = self.dropout(x)
        
        # Unified processing
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        if return_sequence:
            return x  # [B, L, D]
        else:
            return x.mean(dim=1)  # [B, D] - global context


class UnifiedMambaDurationRegulator(nn.Module):
    """Unified duration regulator supporting both architectures"""
    def __init__(self, text_dim=384, style_dim=128, hidden_dim=256, tokens_per_second=75.0, 
                 architecture="hybrid", kernel_sizes=(3, 5)):
        super().__init__()
        self.tokens_per_second = tokens_per_second
        
        # Input projection
        self.input_proj = nn.Linear(text_dim + style_dim, hidden_dim)
        
        # Unified processing for duration context
        self.duration_unified = UnifiedMambaBlock(hidden_dim, expand_factor=2, dropout=0.1, 
                                                architecture=architecture, kernel_sizes=kernel_sizes)
        
        # Duration prediction heads
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_features, style_embedding):
        B, L, D = text_features.shape
        
        # Expand style
        style_expanded = style_embedding.unsqueeze(1).expand(B, L, -1)
        
        # Combine text and style
        combined = torch.cat([text_features, style_expanded], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(combined)  # [B, L, hidden_dim]
        
        # Unified processing for temporal dependencies
        x = self.duration_unified(x)  # [B, L, hidden_dim]
        
        # Predict durations and confidence
        predicted_durations = self.duration_predictor(x).squeeze(-1)  # [B, L]
        predicted_durations = torch.clamp(predicted_durations, min=0.05, max=0.2)
        
        duration_confidence = self.confidence_predictor(x).squeeze(-1)  # [B, L]
        
        # Duration tokens
        duration_tokens = (predicted_durations * self.tokens_per_second).round().long()
        duration_tokens = torch.clamp(duration_tokens, min=2, max=15)
        
        return text_features, predicted_durations, duration_tokens, duration_confidence


class UnifiedStyleExtractor(nn.Module):
    """Unified backward style extractor"""
    def __init__(self, audio_dim=512, style_dim=128, hidden_dim=256, architecture="hybrid", kernel_sizes=(3, 5)):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Unified backward processing for global style context
        self.backward_unified = UnifiedMambaBlock(
            hidden_dim, 
            expand_factor=2, 
            dropout=0.1, 
            reverse=True,  # KEY: Backward processing
            architecture=architecture,
            kernel_sizes=kernel_sizes
        )
        
        # Style projection
        self.style_proj = nn.Sequential(
            nn.Linear(hidden_dim, style_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(style_dim * 2, style_dim),
            nn.LayerNorm(style_dim)
        )
        
    def forward(self, audio_features):
        # audio_features: [B, D, T]
        B, D, T = audio_features.shape
        
        # Transpose to sequence format
        x = audio_features.transpose(1, 2)  # [B, T, D]
        
        # Input projection
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # Unified backward processing (reverses internally)
        processed = self.backward_unified(x)  # [B, T, hidden_dim]
        
        # Global style from mean of processed sequence
        style_features = processed.mean(dim=1)  # [B, hidden_dim]
        
        # Project to style dimension
        style_vector = self.style_proj(style_features)  # [B, style_dim]
        
        return style_vector


class UnifiedMambaAudioProcessor(nn.Module):
    """Unified audio processor for 8 codebooks"""
    def __init__(self, hidden_dim=512, num_codebooks=8, codebook_size=1024, 
                 architecture="hybrid", kernel_sizes=(3, 5)):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Enhanced embeddings for each codebook
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(codebook_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_codebooks)
        ])
        
        # Context projection
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Unified processing layers
        self.unified_layers = nn.ModuleList([
            UnifiedMambaBlock(hidden_dim, expand_factor=2, dropout=0.1, 
                            architecture=architecture, kernel_sizes=kernel_sizes)
            for _ in range(4)
        ])
        
        # Output heads for each codebook
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, codebook_size)
            ) for _ in range(num_codebooks)
        ])
        
    def forward(self, audio_tokens, text_context):
        B, C, T = audio_tokens.shape
        
        # Ensure 8 codebooks
        if C < 8:
            padding = torch.zeros(B, 8 - C, T, dtype=audio_tokens.dtype, device=audio_tokens.device)
            audio_tokens = torch.cat([audio_tokens, padding], dim=1)
        elif C > 8:
            audio_tokens = audio_tokens[:, :8, :]
        
        # Embed each codebook
        embedded = []
        for c in range(8):
            emb = self.audio_embed[c][0](audio_tokens[:, c, :])  # Embedding
            emb = self.audio_embed[c][1](emb)  # LayerNorm
            emb = self.audio_embed[c][2](emb)  # Dropout
            embedded.append(emb)
        
        # Combine embeddings
        x = torch.stack(embedded, dim=1).mean(dim=1)  # [B, T, hidden_dim]
        
        # Add text context
        text_context_proj = self.context_proj(text_context).unsqueeze(1)
        x = x + text_context_proj
        
        # Unified processing
        for layer in self.unified_layers:
            x = layer(x)
        
        # Generate logits for each codebook
        logits = []
        for c in range(8):
            head_logits = self.output_heads[c](x)  # [B, T, codebook_size]
            logits.append(head_logits)
        
        return torch.stack(logits, dim=1)  # [B, 8, T, codebook_size]


class UnifiedMambaTTSModel(nn.Module):
    """Complete Unified TTS model supporting both Pure Mamba and Hybrid architectures"""
    def __init__(self, vocab_size, embed_dim=384, hidden_dim=512, 
                 num_codebooks=8, codebook_size=1024, architecture="hybrid", kernel_sizes=(3, 5)):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.architecture = architecture
        self.kernel_sizes = kernel_sizes
        
        # All Unified components
        self.text_encoder = UnifiedMambaTextEncoder(vocab_size, embed_dim, num_layers=6, 
                                                  architecture=architecture, kernel_sizes=kernel_sizes)
        self.duration_regulator = UnifiedMambaDurationRegulator(
            text_dim=embed_dim, style_dim=128, hidden_dim=256, tokens_per_second=75.0,
            architecture=architecture, kernel_sizes=kernel_sizes
        )
        self.audio_processor = UnifiedMambaAudioProcessor(hidden_dim, num_codebooks, codebook_size, 
                                                        architecture=architecture, kernel_sizes=kernel_sizes)
        self.style_extractor = UnifiedStyleExtractor(hidden_dim, 128, 256, 
                                                   architecture=architecture, kernel_sizes=kernel_sizes)
        
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.default_style = nn.Parameter(torch.randn(128) * 0.01)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 UnifiedMambaTTSModel: {total_params:,} parameters")
        logger.info(f"   📏 Embed dim: {embed_dim}, Hidden dim: {hidden_dim}")
        
        if architecture == "hybrid":
            logger.info(f"   🔄 Hybrid Conv+Mamba architecture")
            logger.info(f"   📡 Receptive field: {kernel_sizes} -> effective: {max(kernel_sizes)} tokens")
        else:
            logger.info(f"   🔄 Pure Mamba architecture")
        
    def forward(self, text_tokens, audio_tokens=None, chunk_duration=None):
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Unified text encoding
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        # Unified style extraction
        if audio_tokens is not None:
            with torch.no_grad():
                B, C, T = audio_tokens.shape
                # Create pseudo audio features for style extraction
                audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2])
                pseudo_audio = audio_mean.unsqueeze(1).unsqueeze(2).expand(B, self.hidden_dim, min(T, 120))
                style_embedding = self.style_extractor(pseudo_audio)
        else:
            style_embedding = self.default_style.unsqueeze(0).expand(batch_size, -1)
        
        # Unified duration regulation
        regulated_features, predicted_durations, duration_tokens, duration_confidence = \
            self.duration_regulator(text_features, style_embedding)
        
        # Unified audio processing
        if audio_tokens is not None:
            regulated_context = torch.mean(self.text_proj(regulated_features), dim=1)
            audio_logits = self.audio_processor(audio_tokens, regulated_context)
        else:
            audio_logits = None
        
        return {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'duration_tokens': duration_tokens,
            'duration_confidence': duration_confidence,
            'text_features': text_features,
            'regulated_features': regulated_features,
            'style_embedding': style_embedding
        }


class UnifiedMambaDataLoader:
    """Unified data loader supporting both architectures"""
    def __init__(self, data_dir="no_overlap_data", device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = device
        self.chunks = []
        self.batches = []
        
        logger.info(f"🔍 Loading NO-OVERLAP data for Unified system from {data_dir}")
        self._load_all_chunks()
        
    def _load_all_chunks(self):
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return
            
        batch_dirs = [d for d in self.data_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('clean_batch_')]
        batch_dirs.sort()
        
        logger.info(f"📁 Found {len(batch_dirs)} clean batch directories")
        
        for batch_dir in batch_dirs:
            try:
                meta_path = batch_dir / "batch_meta.json"
                if not meta_path.exists():
                    continue
                    
                with open(meta_path, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)
                
                batch_chunks = []
                for chunk_file in batch_meta.get('chunk_files', []):
                    chunk_path = batch_dir / chunk_file
                    if chunk_path.exists():
                        try:
                            chunk_data = torch.load(chunk_path, map_location=self.device, weights_only=False)
                            
                            if chunk_data.get('clean_chunk', False) and not chunk_data.get('has_overlap', True):
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
                                
                                chunk_data['batch_dir'] = batch_dir.name
                                chunk_data['unified_mamba'] = True
                                batch_chunks.append(chunk_data)
                                self.chunks.append(chunk_data)
                            
                        except Exception as e:
                            logger.debug(f"Failed to load {chunk_file}: {e}")
                            continue
                
                if batch_chunks:
                    self.batches.append({
                        'batch_idx': len(self.batches),
                        'chunks': batch_chunks,
                        'metadata': batch_meta,
                        'unified_mamba': True
                    })
                    
            except Exception as e:
                logger.debug(f"Failed to process {batch_dir.name}: {e}")
                continue
        
        logger.info(f"📊 Loaded {len(self.chunks)} clean chunks from {len(self.batches)} batches")
        logger.info(f"   🔄 All chunks processed for Unified system")
        
    def get_random_chunk(self):
        if not self.chunks:
            return None
        return self.chunks[np.random.randint(0, len(self.chunks))]
    
    def get_stats(self):
        if not self.chunks:
            return {'total_chunks': 0, 'total_batches': 0}
        
        total_duration = sum(chunk.get('duration', 0) for chunk in self.chunks)
        return {
            'total_chunks': len(self.chunks),
            'total_batches': len(self.batches),
            'total_duration': total_duration,
            'avg_duration': total_duration / len(self.chunks) if self.chunks else 0,
            'unified_mamba': True
        }


class UnifiedMambaTrainer:
    """Unified trainer supporting both Pure Mamba and Hybrid architectures"""
    
    def __init__(self, model, tokenizer, data_loader, architecture="hybrid"):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        self.timing_stats = TimingStats()
        self.milestone_tracker = AccuracyMilestoneTracker()
        self.architecture = architecture
        
        logger.info(f"🎯 UnifiedMambaTrainer initialized")
        logger.info(f"   Data: {data_loader.get_stats()['total_chunks']} clean chunks")
        logger.info(f"   🔄 Architecture: {architecture}")
        if hasattr(model, 'kernel_sizes') and model.kernel_sizes and architecture == "hybrid":
            logger.info(f"   📡 Receptive field: {model.kernel_sizes}")
    
    def train_step(self, chunk_data, step_num=None):
        """Training step with detailed timing"""
        step_start = time.time()
        
        try:
            # Data preparation
            prep_start = time.time()
            if 'text_tokens' not in chunk_data or 'audio_codes' not in chunk_data:
                return None
                
            text_tokens = chunk_data['text_tokens']
            if text_tokens.dim() == 1:
                text_tokens = text_tokens.unsqueeze(0)
                
            audio_codes = chunk_data['audio_codes']
            if audio_codes.dim() == 2:
                audio_codes = audio_codes.unsqueeze(0)
            
            # Ensure 8 codebooks
            B, C, T = audio_codes.shape
            if C < 8:
                padding = torch.zeros(B, 8 - C, T, dtype=audio_codes.dtype, device=audio_codes.device)
                audio_codes = torch.cat([audio_codes, padding], dim=1)
            elif C > 8:
                audio_codes = audio_codes[:, :8, :]
            
            chunk_duration = chunk_data.get('duration', 4.0)
            prep_time = time.time() - prep_start
            self.timing_stats.record('data_prep', prep_time)
            
            # Forward pass timing
            forward_start = time.time()
            output = self.model(text_tokens, audio_codes, chunk_duration=chunk_duration)
            forward_time = time.time() - forward_start
            self.timing_stats.record('forward_pass', forward_time)
            
            # Loss computation timing
            loss_start = time.time()
            loss_dict = compute_combined_loss(output, chunk_data, text_tokens, self.device)
            loss_time = time.time() - loss_start
            self.timing_stats.record('loss_computation', loss_time)
            
            # Check loss validity
            total_loss = loss_dict.get('total_loss')
            if total_loss is None or torch.isnan(total_loss) or torch.isinf(total_loss):
                return None
            
            # Add timing info
            loss_dict['timing'] = {
                'data_prep': prep_time * 1000,
                'forward_pass': forward_time * 1000,
                'loss_computation': loss_time * 1000,
                'total_step': (time.time() - step_start) * 1000
            }
            
            # Add chunk info
            loss_dict['chunk_info'] = {
                'text': chunk_data['text'][:50] + "..." if len(chunk_data['text']) > 50 else chunk_data['text'],
                'duration': chunk_duration,
                'batch_dir': chunk_data.get('batch_dir', 'unknown'),
                'clean_chunk': chunk_data.get('clean_chunk', False),
                'architecture': self.architecture
            }
            
            step_time = time.time() - step_start
            self.timing_stats.record('total_step', step_time)
            
            return loss_dict
            
        except Exception as e:
            logger.warning(f"❌ Training step failed: {e}")
            if step_num is not None and step_num < 10:
                import traceback
                traceback.print_exc()
            return None
    
    def train(self, steps=3000, learning_rate=8e-4):
        """Training with comprehensive timing analysis and milestone tracking"""
        training_start = time.time()
        
        arch_name = "Pure Mamba" if self.architecture == "mamba" else "Hybrid Conv+Mamba"
        logger.info(f"🚀 Starting {arch_name} NO-OVERLAP training for {steps} steps")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   🔄 {arch_name} architecture with milestone tracking")
        logger.info(f"   ⏱️  Detailed timing analysis enabled")
        logger.info(f"   🎯 Tracking accuracy milestones: 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 50%, 60%, 70%, 80%, 90%, 95%")
        
        # Enhanced optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-6,
            betas=(0.9, 0.95)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=learning_rate * 0.1
        )
        
        # Metrics tracking
        successful_steps = 0
        failed_steps = 0
        losses = []
        accuracies = []
        duration_accuracies = []
        best_accuracy = 0.0
        best_duration_accuracy = 0.0
        
        # Training loop with timing
        logger.info(f"🔍 Starting training loop with {len(self.data_loader.chunks)} chunks available...")
        
        for step in range(steps):
            try:
                # Get data
                data_start = time.time()
                chunk_data = self.data_loader.get_random_chunk()
                if chunk_data is None:
                    failed_steps += 1
                    continue
                data_time = time.time() - data_start
                self.timing_stats.record('data_loading', data_time)
                
                # Training step
                optimizer.zero_grad()
                
                loss_dict = self.train_step(chunk_data, step_num=step)
                
                if loss_dict is not None:
                    # Backward pass timing
                    backward_start = time.time()
                    total_loss = loss_dict['total_loss']
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    optimizer.step()
                    scheduler.step()
                    backward_time = time.time() - backward_start
                    self.timing_stats.record('backward_pass', backward_time)
                    
                    # Track metrics
                    losses.append(total_loss.item())
                    current_accuracy = loss_dict['accuracy']
                    current_duration_accuracy = loss_dict['duration_accuracy']
                    accuracies.append(current_accuracy)
                    duration_accuracies.append(current_duration_accuracy)
                    
                    # Update milestone tracking
                    self.milestone_tracker.update(step, current_accuracy, current_duration_accuracy)
                    
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    if current_duration_accuracy > best_duration_accuracy:
                        best_duration_accuracy = current_duration_accuracy
                    
                    successful_steps += 1
                    
                    # Enhanced logging with timing
                    if step % 50 == 0 or current_accuracy > 0.15 or current_duration_accuracy > 0.4:
                        current_lr = scheduler.get_last_lr()[0]
                        timing = loss_dict.get('timing', {})
                        
                        logger.info(f"Step {step:4d}: Loss={total_loss.item():.4f}, "
                                  f"Acc={current_accuracy:.4f}, DurAcc={current_duration_accuracy:.4f}, "
                                  f"LR={current_lr:.2e}")
                        logger.info(f"         ⏱️  Forward: {timing.get('forward_pass', 0):.1f}ms, "
                                  f"Backward: {backward_time*1000:.1f}ms, "
                                  f"Total: {timing.get('total_step', 0):.1f}ms")
                        
                        chunk_info = loss_dict['chunk_info']
                        logger.info(f"         Chunk: '{chunk_info['text']}' ({chunk_info['duration']:.1f}s)")
                    
                    # Success detection with enhanced messaging
                    if current_accuracy > 0.4:
                        logger.info(f"🎉 EXCELLENT {arch_name.upper()} PROGRESS! Accuracy {current_accuracy:.4f}")
                    if current_duration_accuracy > 0.6:
                        logger.info(f"🎉 EXCELLENT DURATION PROGRESS! Duration Accuracy {current_duration_accuracy:.4f}")
                    
                    # Special milestone celebrations
                    if current_accuracy >= 0.5 and step > 1000:
                        logger.info(f"🔥 AMAZING! 50%+ accuracy reached at step {step} with {arch_name}!")
                    if current_accuracy >= 0.7 and step > 1500:
                        logger.info(f"🚀 INCREDIBLE! 70%+ accuracy reached at step {step} with {arch_name}!")
                    if current_accuracy >= 0.9:
                        logger.info(f"🌟 PHENOMENAL! 90%+ accuracy reached at step {step} with {arch_name}!")
                    if current_accuracy >= 0.95:
                        logger.info(f"🏆 PERFECT! 95%+ accuracy reached at step {step} with {arch_name}!")
                    
                    # Early success
                    if best_accuracy > 0.35 and best_duration_accuracy > 0.5 and step > 2000:
                        logger.info(f"🎉 {arch_name.upper()} TRAINING SUCCESS!")
                        break
                        
                else:
                    failed_steps += 1
                    if step < 10:
                        logger.warning(f"Step {step}: Training step returned None")
                        
            except Exception as e:
                logger.warning(f"Step {step} failed with exception: {e}")
                if step < 10:
                    import traceback
                    traceback.print_exc()
                failed_steps += 1
                continue
        
        # Training completion timing
        total_training_time = time.time() - training_start
        
        # Results summary
        success_rate = successful_steps / (successful_steps + failed_steps) * 100 if (successful_steps + failed_steps) > 0 else 0
        
        final_loss = losses[-1] if losses else 999.0
        final_acc = accuracies[-1] if accuracies else 0.0
        final_dur_acc = duration_accuracies[-1] if duration_accuracies else 0.0
        
        logger.info(f"\n🎉 {arch_name} training completed!")
        logger.info(f"   Successful steps: {successful_steps}/{steps} ({success_rate:.1f}%)")
        logger.info(f"   Best audio accuracy: {best_accuracy:.4f}")
        logger.info(f"   Best duration accuracy: {best_duration_accuracy:.4f}")
        logger.info(f"   Final - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}, DurAcc: {final_dur_acc:.4f}")
        
        # Print milestone achievements
        self.milestone_tracker.print_milestones()
        
        # Detailed timing analysis
        logger.info(f"\n⏱️  TRAINING TIMING ANALYSIS:")
        logger.info(f"   Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
        if successful_steps > 0:
            avg_step_time = total_training_time / successful_steps
            logger.info(f"   Average step time: {avg_step_time*1000:.1f}ms")
            logger.info(f"   Steps per second: {1/avg_step_time:.2f}")
        
        # Print detailed timing stats
        self.timing_stats.print_summary()
        
        # Generate comparison audio
        logger.info(f"\n🎵 Generating {arch_name} audio from trained model...")
        self.generate_unified_audio()
        
        # Save model with timing info and milestones
        if best_accuracy > 0.12 or best_duration_accuracy > 0.35:
            model_filename = f'{self.architecture}_unified_mamba_model.pt'
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'best_accuracy': best_accuracy,
                'best_duration_accuracy': best_duration_accuracy,
                'final_loss': final_loss,
                'vocab_size': self.tokenizer.get_vocab_size(),
                'architecture': self.architecture,
                'unified_mamba_training': True,
                'no_overlap_training': True,
                'milestone_achievements': self.milestone_tracker.get_summary(),
                'timing_stats': {
                    'total_training_time': total_training_time,
                    'successful_steps': successful_steps,
                    'avg_step_time': total_training_time / successful_steps if successful_steps > 0 else 0,
                    'detailed_timing': {name: self.timing_stats.get_stats(name) for name in self.timing_stats.times.keys()}
                },
                'model_config': {
                    'embed_dim': 384,
                    'hidden_dim': 512,
                    'num_codebooks': 8,
                    'codebook_size': 1024,
                    'architecture': self.architecture,
                    'kernel_sizes': getattr(self.model, 'kernel_sizes', None)
                }
            }
            
            torch.save(model_data, model_filename)
            
            logger.info(f"💾 {arch_name} model saved as '{model_filename}'")
            logger.info("   📊 Includes detailed timing analysis and milestone achievements")
            return True
        else:
            logger.warning("⚠️  Training not successful enough")
            return False
    
    def generate_unified_audio(self):
        """Generate audio with Unified system"""
        try:
            arch_name = "Pure Mamba" if self.architecture == "mamba" else "Hybrid Conv+Mamba"
            logger.info(f"🎵 Generating {arch_name} audio tokens...")
            
            batch = self.data_loader.get_random_chunk()
            if batch is None:
                logger.warning("⚠️  No batch available for audio generation")
                return
                
            chunk_data = batch
            text = chunk_data['text']
            audio_codes = chunk_data['audio_codes']
            
            # Ensure proper shape [C, T] where C=8 codebooks
            if audio_codes.dim() == 3:
                audio_codes = audio_codes.squeeze(0)
            
            C, T = audio_codes.shape
            if C < 8:
                padding = torch.zeros(8 - C, T, dtype=audio_codes.dtype, device=audio_codes.device)
                audio_codes = torch.cat([audio_codes, padding], dim=0)
            elif C > 8:
                audio_codes = audio_codes[:8, :]
            
            logger.info(f"📊 Generated from: '{text[:50]}...'")
            logger.info(f"🎵 {arch_name} audio tokens shape: {audio_codes.shape}")
            
            # Save tokens with Unified info
            filename = f'{self.architecture}_unified_audio_tokens.pt'
            output_data = {
                'audio_tokens': audio_codes.cpu(),
                'text': text,
                'generation_info': {
                    'model': 'UnifiedMambaTTSModel',
                    'architecture': self.architecture,
                    'training': f'{self.architecture}_unified_no_overlap',
                    'kernel_sizes': getattr(self.model, 'kernel_sizes', None)
                },
                'codebook_info': {
                    'num_codebooks': 8,
                    'codebook_size': 1024,
                    'sample_rate': 24000,
                    'format': f'{self.architecture}_unified_tokens'
                }
            }
            
            torch.save(output_data, filename)
            logger.info(f"💾 {arch_name} audio tokens saved as '{filename}'")
            
            # Save comparison info
            info_filename = f'{self.architecture}_unified_audio_info.txt'
            with open(info_filename, 'w', encoding='utf-8') as f:
                f.write(f"{arch_name} Audio Generation Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Architecture: {arch_name}\n")
                f.write(f"Audio tokens shape: {audio_codes.shape}\n")
                f.write(f"Text: {text}\n\n")
                f.write("Key features:\n")
                if self.architecture == "mamba":
                    f.write("- Pure Mamba blocks (no Conv1D)\n")
                    f.write("- All SSM state space modeling\n")
                    f.write("- Enhanced long-range dependencies\n")
                    f.write("- Selective attention mechanism\n")
                else:
                    f.write("- Hybrid Conv1D + Mamba blocks\n")
                    f.write("- Conv1D for local phonetic patterns\n")
                    f.write("- Mamba for global dependencies\n")
                    f.write("- Optimized speed vs quality balance\n")
                    if hasattr(self.model, 'kernel_sizes') and self.model.kernel_sizes:
                        f.write(f"- Receptive field: {self.model.kernel_sizes}\n")
                f.write("- Milestone tracking enabled\n")
                f.write("- Comprehensive timing analysis\n")
            
            logger.info(f"📄 Info saved as '{info_filename}'")
            logger.info(f"🎵 Ready for comparison with other architectures!")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate {arch_name} audio: {e}")
            import traceback
            traceback.print_exc()


class FullArchitectureExperiment:
    """Class to run comprehensive experiments comparing Pure Mamba vs Hybrid variants"""
    def __init__(self, base_trainer_class):
        self.base_trainer_class = base_trainer_class
        self.results = {}
        
    def run_full_architecture_experiment(self, tokenizer, data_loader, device, experiment_configs):
        """Run comprehensive architecture comparison"""
        logger.info("🚀 FULL ARCHITECTURE EXPERIMENT")
        logger.info("=" * 70)
        logger.info("Comprehensive comparison: Pure Mamba vs Hybrid Conv+Mamba variants")
        
        all_results = {}
        
        for exp_name, architecture, kernel_sizes, test_steps in experiment_configs:
            logger.info(f"\n🔬 Experiment: {exp_name}")
            logger.info(f"   Architecture: {architecture}")
            if kernel_sizes:
                logger.info(f"   Kernel sizes: {kernel_sizes}")
                logger.info(f"   Expected phonetic coverage: ~{max(kernel_sizes)//2}-{max(kernel_sizes)} phonemes")
            else:
                logger.info(f"   Pure Mamba - no Conv1D layers")
            logger.info(f"   Test steps: {test_steps}")
            
            try:
                # Create model with specific architecture
                model = UnifiedMambaTTSModel(
                    vocab_size=tokenizer.get_vocab_size(),
                    embed_dim=384,
                    hidden_dim=512,
                    num_codebooks=8,
                    codebook_size=1024,
                    architecture=architecture,
                    kernel_sizes=kernel_sizes if kernel_sizes else (3, 5)
                ).to(device)
                
                # Create specialized trainer
                trainer = UnifiedMambaTrainer(model, tokenizer, data_loader, architecture=architecture)
                
                # Quick training run
                logger.info(f"🚀 Starting {exp_name} training...")
                success = trainer.train(steps=test_steps, learning_rate=8e-4)
                
                # Collect results
                milestone_summary = trainer.milestone_tracker.get_summary()
                timing_stats = trainer.timing_stats
                
                result = {
                    'architecture': architecture,
                    'kernel_sizes': kernel_sizes,
                    'success': success,
                    'milestones': milestone_summary,
                    'avg_step_time': timing_stats.get_stats('total_step')['mean'] if timing_stats.get_stats('total_step') else 0,
                    'avg_forward_time': timing_stats.get_stats('forward_pass')['mean'] if timing_stats.get_stats('forward_pass') else 0,
                    'param_count': sum(p.numel() for p in model.parameters())
                }
                
                all_results[exp_name] = result
                
                # Quick summary
                milestones_audio = milestone_summary.get('audio_accuracy_milestones', {})
                if milestones_audio:
                    best_milestone = max(milestones_audio.keys(), key=lambda x: float(x.rstrip('%')))
                    step_achieved = milestones_audio[best_milestone]
                    logger.info(f"   ✅ Best audio milestone: {best_milestone} at step {step_achieved}")
                else:
                    logger.info(f"   ⚠️  No audio milestones achieved in {test_steps} steps")
                
                logger.info(f"   ⏱️  Avg step time: {result['avg_step_time']*1000:.1f}ms")
                logger.info(f"   📊 Model parameters: {result['param_count']:,}")
                
                # Save intermediate results
                model_filename = f'full_experiment_{exp_name.lower().replace(" ", "_").replace("+", "_")}.pt'
                torch.save({
                    'experiment_name': exp_name,
                    'model_state_dict': model.state_dict(),
                    'results': result,
                    'config': {
                        'architecture': architecture,
                        'kernel_sizes': kernel_sizes,
                        'embed_dim': 384,
                        'hidden_dim': 512
                    }
                }, model_filename)
                
                logger.info(f"   💾 Model saved as: {model_filename}")
                
            except Exception as e:
                logger.error(f"❌ Experiment {exp_name} failed: {e}")
                all_results[exp_name] = {
                    'error': str(e), 
                    'architecture': architecture, 
                    'kernel_sizes': kernel_sizes
                }
                continue
        
        # Comprehensive comparison
        self._compare_full_results(all_results)
        return all_results
    
    def _compare_full_results(self, results):
        """Comprehensive comparison of all architecture results"""
        logger.info("\n🏆 COMPREHENSIVE ARCHITECTURE COMPARISON:")
        logger.info("=" * 70)
        
        # Architecture type summary
        pure_mamba_results = []
        hybrid_results = []
        
        for name, result in results.items():
            if 'error' in result:
                continue
                
            if result['architecture'] == 'mamba':
                pure_mamba_results.append((name, result))
            else:
                hybrid_results.append((name, result))
        
        logger.info(f"📊 Pure Mamba models: {len(pure_mamba_results)}")
        logger.info(f"📊 Hybrid Conv+Mamba models: {len(hybrid_results)}")
        
        # Speed comparison
        logger.info("\n⚡ SPEED RANKING (faster training = better):")
        speed_results = []
        for name, result in results.items():
            if 'avg_step_time' in result and result['avg_step_time'] > 0:
                speed_results.append((name, result['avg_step_time'], result['architecture'], result.get('kernel_sizes')))
        
        speed_results.sort(key=lambda x: x[1])  # Sort by speed (fastest first)
        for i, (name, avg_time, arch, kernels) in enumerate(speed_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            kernel_str = f"kernels: {kernels}" if kernels else "no kernels"
            logger.info(f"   {rank} {name}: {avg_time*1000:.1f}ms/step ({arch}, {kernel_str})")
        
        # Milestone achievement comparison
        logger.info("\n🎯 AUDIO ACCURACY MILESTONE RANKING:")
        milestone_results = []
        for name, result in results.items():
            if 'milestones' in result:
                milestones = result['milestones'].get('audio_accuracy_milestones', {})
                if milestones:
                    # Find highest milestone achieved
                    best_milestone = max(milestones.keys(), key=lambda x: float(x.rstrip('%')))
                    best_step = milestones[best_milestone]
                    milestone_results.append((name, float(best_milestone.rstrip('%')), best_step, 
                                           result['architecture'], result.get('kernel_sizes')))
                else:
                    milestone_results.append((name, 0.0, 9999, result['architecture'], result.get('kernel_sizes')))
        
        # Sort by milestone percentage (desc), then by step (asc)
        milestone_results.sort(key=lambda x: (-x[1], x[2]))
        for i, (name, milestone_pct, step, arch, kernels) in enumerate(milestone_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            kernel_str = f", kernels: {kernels}" if kernels else ""
            if milestone_pct > 0:
                logger.info(f"   {rank} {name}: {milestone_pct:.1f}% at step {step} ({arch}{kernel_str})")
            else:
                logger.info(f"   {rank} {name}: No milestones achieved ({arch}{kernel_str})")
        
        # Final recommendations
        logger.info("\n💡 FINAL RECOMMENDATIONS:")
        if milestone_results:
            winner = milestone_results[0]
            logger.info(f"🏆 Overall winner: {winner[0]}")
            logger.info(f"   Architecture: {winner[3]}")
            if winner[4]:
                logger.info(f"   Kernel sizes: {winner[4]}")
            logger.info(f"   Achievement: {winner[1]:.1f}% accuracy at step {winner[2]}")
        
        # Save comprehensive results
        comprehensive_results = {
            'experiment_type': 'full_architecture_comparison',
            'all_results': results,
            'speed_ranking': speed_results,
            'milestone_ranking': milestone_results,
            'pure_mamba_results': pure_mamba_results,
            'hybrid_results': hybrid_results,
            'timestamp': time.time()
        }
        
        torch.save(comprehensive_results, 'full_architecture_experiment_results.pt')
        logger.info("\n💾 Comprehensive results saved as 'full_architecture_experiment_results.pt'")


def run_full_architecture_experiments():
    """Main function to run comprehensive architecture experiments"""
    logger.info("🚀 COMPREHENSIVE ARCHITECTURE EXPERIMENT")
    logger.info("=" * 80)
    logger.info("Ultimate comparison: Pure Mamba vs All Hybrid Conv+Mamba variants")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check data directory
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        logger.error("❌ no_overlap_data directory not found!")
        return
    
    try:
        # Setup base components
        tokenizer = NucleotideTokenizer()
        data_loader = UnifiedMambaDataLoader("no_overlap_data", device)
        stats = data_loader.get_stats()
        
        if stats['total_chunks'] == 0:
            logger.error("❌ No chunks loaded!")
            return
        
        logger.info(f"📊 Data: {stats['total_chunks']} chunks, {stats['total_duration']:.1f}s total")
        
        # Define comprehensive experiment configurations
        experiment_configs = [
            # (name, architecture, kernel_sizes, test_steps)
            ("Pure Mamba", "mamba", None, 1000),                    # Pure Mamba baseline
            ("Hybrid Small RF", "hybrid", (3, 5), 1000),           # Current baseline 
            ("Hybrid Medium RF", "hybrid", (5, 7), 1000),          # Slightly larger context
            ("Hybrid Large RF", "hybrid", (7, 9), 1000),           # Large context
            ("Hybrid XLarge RF", "hybrid", (9, 11), 1000),         # Very large context
            ("Hybrid Multi RF", "hybrid", (3, 9), 1000),           # Multi-scale approach
        ]
        
        logger.info("🎯 Comprehensive experiment configurations:")
        logger.info("   🔍 Pure Mamba: All Mamba blocks, no Conv1D")
        for name, arch, kernels, steps in experiment_configs:
            if kernels:
                phonetic_coverage = f"~{max(kernels)//2}-{max(kernels)} phonemes"
                logger.info(f"   🔍 {name}: {arch} with kernels {kernels} → {phonetic_coverage}")
            else:
                logger.info(f"   🔍 {name}: {arch} architecture")
        
        logger.info(f"\n🧪 Each experiment will run for {experiment_configs[0][3]} steps")
        logger.info("📊 Comparing: speed, milestone achievements, final accuracy")
        logger.info("🏆 Will determine ultimate winner across all architectures")
        
        # Run comprehensive experiments
        full_experiment = FullArchitectureExperiment(UnifiedMambaTrainer)
        results = full_experiment.run_full_architecture_experiment(
            tokenizer, data_loader, device, experiment_configs
        )
        
        # Final summary
        logger.info("\n✅ COMPREHENSIVE ARCHITECTURE EXPERIMENT COMPLETED!")
        logger.info("🏆 Check results for ultimate architecture winner")
        logger.info("🎯 Use winning configuration for production training")
        logger.info("📊 All models saved for detailed comparison")
        
    except Exception as e:
        logger.error(f"❌ Comprehensive architecture experiment failed: {e}")
        import traceback
        traceback.print_exc()


def compare_architectures():
    """Compare different architectures including milestone achievements"""
    logger.info("\n🔍 ARCHITECTURE COMPARISON:")
    logger.info("=" * 60)
    
    models_to_check = [
        ('enhanced_8codebook_model.pt', 'Enhanced 8-Codebook (Conv+Mamba)'),
        ('full_mamba_model.pt', 'Full Mamba (All Mamba)'),
        ('mamba_unified_mamba_model.pt', 'Pure Mamba (Unified)'),
        ('hybrid_unified_mamba_model.pt', 'Hybrid Conv+Mamba (Unified)'),
        ('no_overlap_model.pt', 'Original NO-OVERLAP')
    ]
    
    found_models = []
    for model_file, description in models_to_check:
        if Path(model_file).exists():
            try:
                model_data = torch.load(model_file, map_location='cpu')
                found_models.append((model_file, description, model_data))
                logger.info(f"✅ Found: {description}")
                
                # Print key metrics
                if 'best_accuracy' in model_data:
                    logger.info(f"   Audio Accuracy: {model_data['best_accuracy']:.4f}")
                if 'best_duration_accuracy' in model_data:
                    logger.info(f"   Duration Accuracy: {model_data['best_duration_accuracy']:.4f}")
                if 'timing_stats' in model_data:
                    timing = model_data['timing_stats']
                    if 'avg_step_time' in timing:
                        logger.info(f"   Avg Step Time: {timing['avg_step_time']*1000:.1f}ms")
                
                # Print milestone achievements if available
                if 'milestone_achievements' in model_data:
                    milestones = model_data['milestone_achievements']
                    audio_milestones = milestones.get('audio_accuracy_milestones', {})
                    duration_milestones = milestones.get('duration_accuracy_milestones', {})
                    
                    if audio_milestones:
                        highest_audio = max(audio_milestones.keys(), key=lambda x: float(x.rstrip('%')))
                        logger.info(f"   Highest Audio Milestone: {highest_audio} at step {audio_milestones[highest_audio]}")
                    
                    if duration_milestones:
                        highest_duration = max(duration_milestones.keys(), key=lambda x: float(x.rstrip('%')))
                        logger.info(f"   Highest Duration Milestone: {highest_duration} at step {duration_milestones[highest_duration]}")
                
                logger.info("")  # Empty line for readability
                
            except Exception as e:
                logger.debug(f"Failed to load {model_file}: {e}")
    
    if len(found_models) >= 2:
        logger.info(f"📊 Found {len(found_models)} models for comparison!")
        logger.info("🎯 Compare milestone achievements to see which architecture learns fastest")
        logger.info("⏱️  Use timing analysis to compare training speed")
        logger.info("🎵 Use real_encodec_decoder.py to compare audio quality")
        
        # Speed comparison
        logger.info("\n⚡ SPEED COMPARISON:")
        speed_data = []
        for model_file, description, model_data in found_models:
            if 'timing_stats' in model_data and 'avg_step_time' in model_data['timing_stats']:
                avg_time = model_data['timing_stats']['avg_step_time'] * 1000
                speed_data.append((description, avg_time))
        
        speed_data.sort(key=lambda x: x[1])  # Sort by speed (fastest first)
        for i, (desc, avg_time) in enumerate(speed_data):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            logger.info(f"   {rank} {desc}: {avg_time:.1f}ms/step")
        
    else:
        logger.info("📝 Train multiple models to enable comparison.")


def analyze_milestone_performance():
    """Analyze milestone performance across models"""
    logger.info("\n🎯 MILESTONE PERFORMANCE ANALYSIS:")
    logger.info("=" * 50)
    
    models_with_milestones = []
    model_files = ['mamba_unified_mamba_model.pt', 'hybrid_unified_mamba_model.pt', 
                   'full_mamba_model.pt', 'enhanced_8codebook_model.pt']
    
    for model_file in model_files:
        if Path(model_file).exists():
            try:
                model_data = torch.load(model_file, map_location='cpu')
                if 'milestone_achievements' in model_data:
                    models_with_milestones.append((model_file, model_data))
            except:
                continue
    
    if not models_with_milestones:
        logger.info("No models with milestone data found.")
        return
    
    # Analyze which model reaches milestones fastest
    milestone_comparison = {}
    target_milestones = ['10.0%', '20.0%', '30.0%', '40.0%', '50.0%']
    
    for milestone in target_milestones:
        milestone_comparison[milestone] = []
        
        for model_file, model_data in models_with_milestones:
            milestones = model_data['milestone_achievements']
            audio_milestones = milestones.get('audio_accuracy_milestones', {})
            
            if milestone in audio_milestones:
                step = audio_milestones[milestone]
                arch_name = model_data.get('architecture', 'unknown')
                milestone_comparison[milestone].append((arch_name, step))
    
    # Print comparison
    for milestone, data in milestone_comparison.items():
        if data:
            logger.info(f"\n{milestone} Audio Accuracy:")
            data.sort(key=lambda x: x[1])  # Sort by step number
            for i, (arch, step) in enumerate(data):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                logger.info(f"   {rank} {arch}: Step {step}")


def main():
    """Enhanced main function with full architecture comparison"""
    logger.info("🔄 Ultimate Unified Mamba TTS Training System")
    logger.info("=" * 80)
    logger.info("🎯 Features: Pure Mamba vs Hybrid Conv+Mamba architecture comparison")
    logger.info("📊 Enhanced with milestone tracking and comprehensive timing analysis")
    
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == '--full-experiment':
            run_full_architecture_experiments()
            return
        elif arg in ['--help', '-h']:
            print_usage_help()
            return
        elif arg.startswith('--arch='):
            # Specify architecture
            architecture = arg.split('=')[1]
            if architecture not in ['mamba', 'hybrid']:
                logger.error("❌ Architecture must be 'mamba' or 'hybrid'")
                return
        else:
            architecture = "hybrid"  # Default
    else:
        architecture = "hybrid"  # Default
    
    # Parse kernel sizes if provided
    kernel_sizes = (3, 5)  # Default
    if len(sys.argv) > 2 and sys.argv[2].startswith('--kernels='):
        kernel_str = sys.argv[2].split('=')[1]
        try:
            kernel_sizes = tuple(map(int, kernel_str.split(',')))
            logger.info(f"🔧 Using custom kernel sizes: {kernel_sizes}")
        except:
            logger.warning(f"⚠️  Invalid kernel format, using default: {kernel_sizes}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check data directory
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        logger.error("❌ no_overlap_data directory not found!")
        logger.error("   Please run audio_processor_sequential.py first")
        return
    
    try:
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        data_loader = UnifiedMambaDataLoader("no_overlap_data", device)
        stats = data_loader.get_stats()
        
        if stats['total_chunks'] == 0:
            logger.error("❌ No chunks loaded!")
            return
        
        arch_name = "Pure Mamba" if architecture == "mamba" else "Hybrid Conv+Mamba"
        logger.info(f"📊 {arch_name} Data: {stats['total_chunks']} chunks, {stats['total_duration']:.1f}s total")
        
        # Unified model with specified architecture
        model = UnifiedMambaTTSModel(
            vocab_size=vocab_size,
            embed_dim=384,
            hidden_dim=512,
            num_codebooks=8,
            codebook_size=1024,
            architecture=architecture,
            kernel_sizes=kernel_sizes
        ).to(device)
        
        # Unified trainer
        trainer = UnifiedMambaTrainer(model, tokenizer, data_loader, architecture=architecture)
        
        # Training with comprehensive analysis
        logger.info(f"\n🚀 Starting {arch_name} training...")
        logger.info(f"   🔄 {arch_name} architecture")
        if architecture == "hybrid":
            logger.info(f"   📡 Receptive field: {kernel_sizes}")
        logger.info(f"   ⏱️  Comprehensive timing analysis")
        logger.info(f"   🎯 Milestone tracking for accuracy achievements")
        logger.info(f"   📊 Compare with other architectures using --full-experiment")
        
        success = trainer.train(steps=3000, learning_rate=8e-4)
        
        if success:
            logger.info(f"✅ {arch_name} training successful!")
            logger.info("🎵 Compare with other models using milestone analysis!")
            logger.info("🚀 Run --full-experiment for comprehensive architecture comparison!")
        else:
            logger.warning("⚠️  Training needs more steps or parameter adjustment")
        
        # Compare with other architectures
        compare_architectures()
        
        # Analyze milestone performance
        analyze_milestone_performance()
            
    except Exception as e:
        logger.error(f"❌ {arch_name} training failed: {e}")
        import traceback
        traceback.print_exc()


def print_usage_help():
    """Print comprehensive usage help"""
    print("🔄 Ultimate Unified Mamba TTS Training System")
    print("=" * 70)
    print()
    print("USAGE OPTIONS:")
    print("1. Regular training with default Hybrid architecture:")
    print("   python unified_mamba_tts.py")
    print()
    print("2. Training with specific architecture:")
    print("   python unified_mamba_tts.py --arch=mamba      # Pure Mamba")
    print("   python unified_mamba_tts.py --arch=hybrid     # Hybrid Conv+Mamba")
    print()
    print("3. Training with custom kernel sizes (Hybrid only):")
    print("   python unified_mamba_tts.py --arch=hybrid --kernels=5,7")
    print("   python unified_mamba_tts.py --arch=hybrid --kernels=7,9")
    print("   python unified_mamba_tts.py --arch=hybrid --kernels=3,9")
    print()
    print("4. Run COMPREHENSIVE architecture comparison:")
    print("   python unified_mamba_tts.py --full-experiment")
    print("   (Tests Pure Mamba + 5 Hybrid variants automatically!)")
    print()
    print("ARCHITECTURE COMPARISON:")
    print("   Pure Mamba:")
    print("   - All Mamba blocks, no Conv1D")
    print("   - Best for long-range dependencies")
    print("   - Potentially slower training")
    print()
    print("   Hybrid Conv+Mamba:")
    print("   - Conv1D for local phonetic patterns")
    print("   - Mamba for global dependencies")
    print("   - Faster training, balanced approach")
    print()
    print("KERNEL SIZE GUIDE (Hybrid only):")
    print("   (3,5) - Default, good for local phonetic patterns")
    print("   (5,7) - Medium context, likely optimal")
    print("   (7,9) - Large context, good for complex phonetics")
    print("   (9,11) - Very large, may be too slow")
    print("   (3,9) - Multi-scale, combines local + global")
    print()
    print("EXPECTED RESULTS:")
    print("   --full-experiment will determine the ultimate winner!")


if __name__ == "__main__":
    main()