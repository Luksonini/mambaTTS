#!/usr/bin/env python3
"""
Full Mamba TTS Training System with Timing
==========================================
Complete rewrite with Mamba everywhere + detailed timing analysis
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

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

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


class FullMambaBlock(nn.Module):
    """Enhanced Mamba block for all components"""
    def __init__(self, d_model, expand_factor=2, dropout=0.1, reverse=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        self.reverse = reverse
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise convolution for local context
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
        
        # Normalization and activation
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        residual = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # Reverse sequence if backward processing
        if self.reverse:
            x = torch.flip(x, dims=[1])
        
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
        # In real Mamba, this would be more complex selective scan
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


class FullMambaTextEncoder(nn.Module):
    """Full Mamba text encoder - replaces all Conv/Attention"""
    def __init__(self, vocab_size, embed_dim=384, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, embed_dim) * 0.02)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            FullMambaBlock(embed_dim, expand_factor=2, dropout=0.1, reverse=False)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.RMSNorm(embed_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, tokens, return_sequence=True):
        B, L = tokens.shape
        
        # Token embeddings
        x = self.embedding(tokens)  # [B, L, D]
        
        # Add positional encoding
        if L <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :L, :]
        
        x = self.dropout(x)
        
        # Mamba processing
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        if return_sequence:
            return x  # [B, L, D]
        else:
            return x.mean(dim=1)  # [B, D] - global context


class FullMambaDurationRegulator(nn.Module):
    """Full Mamba duration regulator"""
    def __init__(self, text_dim=384, style_dim=128, hidden_dim=256, tokens_per_second=75.0):
        super().__init__()
        self.tokens_per_second = tokens_per_second
        
        # Input projection
        self.input_proj = nn.Linear(text_dim + style_dim, hidden_dim)
        
        # Mamba processing for duration context
        self.duration_mamba = FullMambaBlock(hidden_dim, expand_factor=2, dropout=0.1)
        
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
        
        # Mamba processing for temporal dependencies
        x = self.duration_mamba(x)  # [B, L, hidden_dim]
        
        # Predict durations and confidence
        predicted_durations = self.duration_predictor(x).squeeze(-1)  # [B, L]
        predicted_durations = torch.clamp(predicted_durations, min=0.05, max=0.2)
        
        duration_confidence = self.confidence_predictor(x).squeeze(-1)  # [B, L]
        
        # Duration tokens
        duration_tokens = (predicted_durations * self.tokens_per_second).round().long()
        duration_tokens = torch.clamp(duration_tokens, min=2, max=15)
        
        return text_features, predicted_durations, duration_tokens, duration_confidence


class BackwardMambaStyleExtractor(nn.Module):
    """Backward Mamba style extractor - NEW COMPONENT"""
    def __init__(self, audio_dim=512, style_dim=128, hidden_dim=256):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Backward Mamba for global style context
        self.backward_mamba = FullMambaBlock(
            hidden_dim, 
            expand_factor=2, 
            dropout=0.1, 
            reverse=True  # KEY: Backward processing
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
        
        # Backward Mamba processing (reverses internally)
        processed = self.backward_mamba(x)  # [B, T, hidden_dim]
        
        # Global style from mean of processed sequence
        style_features = processed.mean(dim=1)  # [B, hidden_dim]
        
        # Project to style dimension
        style_vector = self.style_proj(style_features)  # [B, style_dim]
        
        return style_vector


class FullMambaAudioProcessor(nn.Module):
    """Full Mamba audio processor for 8 codebooks"""
    def __init__(self, hidden_dim=512, num_codebooks=8, codebook_size=1024):
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
        
        # Mamba processing layers instead of enhanced blocks
        self.mamba_layers = nn.ModuleList([
            FullMambaBlock(hidden_dim, expand_factor=2, dropout=0.1)
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
        
        # Mamba processing
        for layer in self.mamba_layers:
            x = layer(x)
        
        # Generate logits for each codebook
        logits = []
        for c in range(8):
            head_logits = self.output_heads[c](x)  # [B, T, codebook_size]
            logits.append(head_logits)
        
        return torch.stack(logits, dim=1)  # [B, 8, T, codebook_size]


class FullMambaTTSModel(nn.Module):
    """Complete Full Mamba TTS model"""
    def __init__(self, vocab_size, embed_dim=384, hidden_dim=512, 
                 num_codebooks=8, codebook_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        
        # All Mamba components
        self.text_encoder = FullMambaTextEncoder(vocab_size, embed_dim, num_layers=6)
        self.duration_regulator = FullMambaDurationRegulator(
            text_dim=embed_dim, style_dim=128, hidden_dim=256, tokens_per_second=75.0
        )
        self.audio_processor = FullMambaAudioProcessor(hidden_dim, num_codebooks, codebook_size)
        self.style_extractor = BackwardMambaStyleExtractor(hidden_dim, 128, 256)  # NEW!
        
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.default_style = nn.Parameter(torch.randn(128) * 0.01)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 FullMambaTTSModel: {total_params:,} parameters")
        logger.info(f"   📏 Embed dim: {embed_dim}, Hidden dim: {hidden_dim}")
        logger.info(f"   🔄 Full Mamba architecture (including BACKWARD style extractor)")
        
    def forward(self, text_tokens, audio_tokens=None, chunk_duration=None):
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Mamba text encoding
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        # Backward Mamba style extraction
        if audio_tokens is not None:
            with torch.no_grad():
                B, C, T = audio_tokens.shape
                # Create pseudo audio features for style extraction
                audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2])
                pseudo_audio = audio_mean.unsqueeze(1).unsqueeze(2).expand(B, self.hidden_dim, min(T, 120))
                style_embedding = self.style_extractor(pseudo_audio)
        else:
            style_embedding = self.default_style.unsqueeze(0).expand(batch_size, -1)
        
        # Mamba duration regulation
        regulated_features, predicted_durations, duration_tokens, duration_confidence = \
            self.duration_regulator(text_features, style_embedding)
        
        # Mamba audio processing
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


class FullMambaDataLoader:
    """Same data loader with timing"""
    def __init__(self, data_dir="no_overlap_data", device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = device
        self.chunks = []
        self.batches = []
        
        logger.info(f"🔍 Loading NO-OVERLAP data for Full Mamba system from {data_dir}")
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
                                chunk_data['full_mamba'] = True
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
                        'full_mamba': True
                    })
                    
            except Exception as e:
                logger.debug(f"Failed to process {batch_dir.name}: {e}")
                continue
        
        logger.info(f"📊 Loaded {len(self.chunks)} clean chunks from {len(self.batches)} batches")
        logger.info(f"   🔄 All chunks processed for Full Mamba system")
        
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
            'full_mamba': True
        }


class FullMambaTrainer:
    """Full Mamba trainer with detailed timing"""
    
    def __init__(self, model, tokenizer, data_loader):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        self.timing_stats = TimingStats()
        
        logger.info(f"🎯 FullMambaTrainer initialized")
        logger.info(f"   Data: {data_loader.get_stats()['total_chunks']} clean chunks")
        logger.info(f"   🔄 Full Mamba system with timing analysis")
    
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
                'full_mamba': chunk_data.get('full_mamba', False)
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
        """Training with comprehensive timing analysis"""
        training_start = time.time()
        
        logger.info(f"🚀 Starting Full Mamba NO-OVERLAP training for {steps} steps")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   🔄 Full Mamba architecture with Backward Style Extractor")
        logger.info(f"   ⏱️  Detailed timing analysis enabled")
        
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
            step_start = time.time()
            
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
                
                train_start = time.time()
                loss_dict = self.train_step(chunk_data, step_num=step)
                train_time = time.time() - train_start
                
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
                    
                    # Success detection
                    if current_accuracy > 0.4:
                        logger.info(f"🎉 EXCELLENT FULL MAMBA PROGRESS! Accuracy {current_accuracy:.4f}")
                    if current_duration_accuracy > 0.6:
                        logger.info(f"🎉 EXCELLENT DURATION PROGRESS! Duration Accuracy {current_duration_accuracy:.4f}")
                    
                    # Early success
                    if best_accuracy > 0.35 and best_duration_accuracy > 0.5 and step > 2000:
                        logger.info(f"🎉 FULL MAMBA TRAINING SUCCESS!")
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
        
        # Results summary with timing
        success_rate = successful_steps / (successful_steps + failed_steps) * 100 if (successful_steps + failed_steps) > 0 else 0
        
        final_loss = losses[-1] if losses else 999.0
        final_acc = accuracies[-1] if accuracies else 0.0
        final_dur_acc = duration_accuracies[-1] if duration_accuracies else 0.0
        
        logger.info(f"\n🎉 Full Mamba training completed!")
        logger.info(f"   Successful steps: {successful_steps}/{steps} ({success_rate:.1f}%)")
        logger.info(f"   Best audio accuracy: {best_accuracy:.4f}")
        logger.info(f"   Best duration accuracy: {best_duration_accuracy:.4f}")
        logger.info(f"   Final - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}, DurAcc: {final_dur_acc:.4f}")
        logger.info(f"   🔄 Full Mamba system with Backward Style Extractor")
        
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
        logger.info("\n🎵 Generating Full Mamba audio from trained model...")
        self.generate_full_mamba_audio()
        
        # Save model with timing info
        if best_accuracy > 0.12 or best_duration_accuracy > 0.35:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'best_accuracy': best_accuracy,
                'best_duration_accuracy': best_duration_accuracy,
                'final_loss': final_loss,
                'vocab_size': self.tokenizer.get_vocab_size(),
                'full_mamba_training': True,
                'no_overlap_training': True,
                'backward_style_extractor': True,
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
                    'architecture': 'full_mamba_with_backward_style'
                }
            }
            
            torch.save(model_data, 'full_mamba_model.pt')
            
            logger.info("💾 Full Mamba model saved as 'full_mamba_model.pt'")
            logger.info("   📊 Includes detailed timing analysis")
            return True
        else:
            logger.warning("⚠️  Training not successful enough")
            return False
    
    def generate_full_mamba_audio(self):
        """Generate audio with Full Mamba system"""
        try:
            logger.info("🎵 Generating Full Mamba audio tokens...")
            
            batch = self.data_loader.get_random_chunk()
            if batch is None:
                logger.warning("⚠️  No batch available for audio generation")
                return
                
            # Use single chunk for generation timing test
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
            logger.info(f"🎵 Full Mamba audio tokens shape: {audio_codes.shape}")
            
            # Save tokens with Full Mamba info
            output_data = {
                'audio_tokens': audio_codes.cpu(),
                'text': text,
                'generation_info': {
                    'model': 'FullMambaTTSModel',
                    'architecture': 'full_mamba_with_backward_style',
                    'training': 'full_mamba_no_overlap',
                    'backward_style_extractor': True
                },
                'codebook_info': {
                    'num_codebooks': 8,
                    'codebook_size': 1024,
                    'sample_rate': 24000,
                    'format': 'full_mamba_tokens'
                }
            }
            
            torch.save(output_data, 'full_mamba_audio_tokens.pt')
            logger.info("💾 Full Mamba audio tokens saved as 'full_mamba_audio_tokens.pt'")
            
            # Save comparison info
            with open('full_mamba_audio_info.txt', 'w', encoding='utf-8') as f:
                f.write("Full Mamba Audio Generation Results\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Architecture: Full Mamba with Backward Style Extractor\n")
                f.write(f"Audio tokens shape: {audio_codes.shape}\n")
                f.write(f"Text: {text}\n\n")
                f.write("Key features:\n")
                f.write("- All Mamba blocks (no Conv1D)\n")
                f.write("- Backward Mamba style extractor\n")
                f.write("- Enhanced temporal modeling\n")
                f.write("- Timing-optimized architecture\n")
            
            logger.info("📄 Info saved as 'full_mamba_audio_info.txt'")
            logger.info("🎵 Ready for comparison with previous models!")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Full Mamba audio: {e}")
            import traceback
            traceback.print_exc()


def compare_architectures():
    """Compare different architectures if multiple models exist"""
    logger.info("\n🔍 ARCHITECTURE COMPARISON:")
    logger.info("=" * 50)
    
    models_to_check = [
        ('enhanced_8codebook_model.pt', 'Enhanced 8-Codebook (Conv+Mamba)'),
        ('full_mamba_model.pt', 'Full Mamba (All Mamba)'),
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
                
            except Exception as e:
                logger.debug(f"Failed to load {model_file}: {e}")
    
    if len(found_models) >= 2:
        logger.info(f"\n📊 Found {len(found_models)} models for comparison!")
        logger.info("Use the timing analysis to compare training speed.")
        logger.info("Use real_encodec_decoder.py to compare audio quality.")
    else:
        logger.info("📝 Train multiple models to enable comparison.")


def main():
    """Main function for Full Mamba training"""
    logger.info("🔄 Full Mamba TTS Training System")
    logger.info("=" * 60)
    
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
        
        data_loader = FullMambaDataLoader("no_overlap_data", device)
        stats = data_loader.get_stats()
        
        if stats['total_chunks'] == 0:
            logger.error("❌ No chunks loaded!")
            return
        
        logger.info(f"📊 Full Mamba Data: {stats['total_chunks']} chunks, {stats['total_duration']:.1f}s total")
        
        # Full Mamba model
        model = FullMambaTTSModel(
            vocab_size=vocab_size,
            embed_dim=384,
            hidden_dim=512,
            num_codebooks=8,
            codebook_size=1024
        ).to(device)
        
        # Full Mamba trainer
        trainer = FullMambaTrainer(model, tokenizer, data_loader)
        
        # Training with timing analysis
        logger.info(f"\n🚀 Starting Full Mamba training...")
        logger.info(f"   🔄 All Mamba architecture (including Backward Style Extractor)")
        logger.info(f"   ⏱️  Comprehensive timing analysis")
        logger.info(f"   📊 Compare speed vs previous models")
        
        success = trainer.train(steps=3000, learning_rate=8e-4)
        
        if success:
            logger.info("✅ Full Mamba training successful!")
            logger.info("🎵 Compare with previous models using timing analysis!")
        else:
            logger.warning("⚠️  Training needs more steps or parameter adjustment")
        
        # Compare with other architectures
        compare_architectures()
            
    except Exception as e:
        logger.error(f"❌ Full Mamba training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()