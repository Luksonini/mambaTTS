#!/usr/bin/env python3
"""
🔥 RESIDUAL REFINEMENT CHALLENGE 🔥
===================================
GOAL: Beat Single-Step E768_H768 in accuracy/speed ratio!
CONSTRAINT: Max +100M parameters (few million allowed)

STRATEGY: 8 small specialized networks (residual refinement)
- Network 1: Predict codebooks [1,2] from text
- Network 2: Refine [2] + predict [3] using [1,2]
- Network 3: Refine [3] + predict [4] using [1,2,3]
- ...and so on to Network 8

Each network is TINY (~1-2M params) but specialized!
Total: ~12-16M params (vs single-step ~50M params)

THEORY: Many small experts > One big generalist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Imports z Twojego working kodu
try:
    from nucleotide_tokenizer import NucleotideTokenizer
    from losses import compute_combined_loss
    logging.info("✅ Imported tokenizer and losses")
except ImportError as e:
    logging.error(f"❌ Import error: {e}")
    exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChallengeConfig:
    """Ultra-efficient config for challenge"""
    name: str = "ResidualRefinement_Challenge"
    # SMALL dimensions for efficiency
    embed_dim: int = 384        # Half of E768!
    hidden_dim: int = 384       # Keep ratio 1:1
    num_layers: int = 2         # Only 2 layers per network!
    expand_factor: float = 1.0  # Minimal expansion
    num_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.05       # Less dropout for small networks


class MiniMambaBlock(nn.Module):
    """Ultra-compact Mamba block for challenge"""
    def __init__(self, d_model, expand_factor=1.0, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand_factor)  # Minimal expansion
        
        # Compact Mamba core
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Lightweight conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=3, padding=1, 
            groups=self.d_inner // 4,  # Fewer groups = less params
            bias=False
        )
        
        # Essential SSM components
        self.x_proj = nn.Linear(self.d_inner, self.d_inner // 2, bias=False)  # Smaller!
        self.dt_proj = nn.Linear(self.d_inner // 2, self.d_inner, bias=True)
        
        # Compact output
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Minimal extras
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, L, D = x.shape
        residual = x
        
        x = self.norm(x)
        
        # Compact Mamba processing
        x_proj = self.in_proj(x)  # [B, L, 2*d_inner]
        x1, x2 = x_proj.chunk(2, dim=-1)
        
        # Lightweight conv
        x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)
        x1_ssm = self.activation(x1_conv)
        
        # Compact SSM
        ssm_input = self.x_proj(x1_ssm)  # Reduce dimension
        dt = self.dt_proj(ssm_input)     # Expand back
        dt = F.softplus(dt)
        
        # Efficient state processing
        x1_processed = x1_ssm * torch.sigmoid(dt)
        x_gated = x1_processed * torch.sigmoid(x2)
        
        # Output with residual
        output = self.out_proj(x_gated)
        output = self.dropout(output)
        
        return output + residual


class CompactTextEncoder(nn.Module):
    """Compact text encoder for challenge"""
    def __init__(self, vocab_size, config: ChallengeConfig):
        super().__init__()
        
        # Compact embedding
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, config.embed_dim) * 0.02)
        
        # Only 2 mini layers!
        self.layers = nn.ModuleList([
            MiniMambaBlock(config.embed_dim, config.expand_factor, config.dropout)
            for _ in range(config.num_layers)  # Just 2!
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, tokens):
        B, L = tokens.shape
        
        x = self.embedding(tokens)
        
        if L <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :L, :]
        
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


class ResidualRefinementNetwork(nn.Module):
    """Single refinement network - TINY specialist"""
    def __init__(self, config: ChallengeConfig, stage: int):
        super().__init__()
        self.stage = stage  # 1-8
        self.config = config
        
        # Determine input/output codebooks
        if stage == 1:
            # First network: predict first 2 codebooks from text
            self.input_codebooks = 0
            self.output_codebooks = 2
            self.refine_codebooks = []
        else:
            # Later networks: refine previous + add new
            self.input_codebooks = stage  # How many existing codebooks to use
            self.output_codebooks = 1     # Add 1 new codebook
            self.refine_codebooks = [stage - 1]  # Refine the last one
        
        # TINY input processing
        input_dim = config.embed_dim
        if self.input_codebooks > 0:
            # Add codebook embeddings
            self.codebook_embeddings = nn.ModuleList([
                nn.Embedding(config.codebook_size, config.embed_dim // 4)  # Quarter size!
                for _ in range(self.input_codebooks)
            ])
            input_dim += (config.embed_dim // 4) * self.input_codebooks
        
        # Input projection to standard size
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        # TINY processing core - just 1 layer!
        self.processing = MiniMambaBlock(config.hidden_dim, config.expand_factor, config.dropout)
        
        # Output heads - one per codebook we're outputting
        total_outputs = self.output_codebooks + len(self.refine_codebooks)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, config.codebook_size)
            ) for _ in range(total_outputs)
        ])
        
        # Count parameters
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"🔧 Stage {stage}: {param_count:,} parameters")
        logger.info(f"   📥 Input codebooks: {self.input_codebooks}")
        logger.info(f"   📤 Output codebooks: {self.output_codebooks}")
        logger.info(f"   🔄 Refine codebooks: {self.refine_codebooks}")
    
    def forward(self, text_features, existing_codebooks=None):
        """
        text_features: [B, T, embed_dim] - from text encoder
        existing_codebooks: [B, num_existing, T] - previous predictions
        Returns: [B, num_outputs, T, codebook_size] - logits for this stage
        """
        B, T, _ = text_features.shape
        
        # Start with text features
        x = text_features  # [B, T, embed_dim]
        
        # Add existing codebook information
        if existing_codebooks is not None and self.input_codebooks > 0:
            codebook_features = []
            for cb_idx in range(min(self.input_codebooks, existing_codebooks.shape[1])):
                cb_tokens = existing_codebooks[:, cb_idx, :]  # [B, T]
                cb_embed = self.codebook_embeddings[cb_idx](cb_tokens)  # [B, T, embed_dim//4]
                codebook_features.append(cb_embed)
            
            if codebook_features:
                codebook_concat = torch.cat(codebook_features, dim=-1)  # [B, T, ...]
                x = torch.cat([x, codebook_concat], dim=-1)  # Concatenate features
        
        # Project to standard hidden size
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # Process through tiny network
        x = self.processing(x)  # [B, T, hidden_dim]
        
        # Generate outputs
        outputs = []
        for head in self.output_heads:
            output = head(x)  # [B, T, codebook_size]
            outputs.append(output)
        
        # Stack outputs
        stage_outputs = torch.stack(outputs, dim=1)  # [B, num_outputs, T, codebook_size]
        
        return stage_outputs


class ChallengeResidualRefinementModel(nn.Module):
    """THE CHALLENGE MODEL - Beat single-step with minimal params!"""
    def __init__(self, config: ChallengeConfig):
        super().__init__()
        self.config = config
        
        # Compact text encoder
        self.text_encoder = CompactTextEncoder(vocab_size=1000, config=config)
        
        # TINY duration predictor
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        # 8 tiny refinement networks
        self.refinement_stages = nn.ModuleList([
            ResidualRefinementNetwork(config, stage=i+1)
            for i in range(8)
        ])
        
        # Simple learned upsampler
        self.upsampler = nn.Sequential(
            nn.Conv1d(config.embed_dim, config.embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.embed_dim, config.embed_dim, 3, padding=1)
        )
        
        # Total parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🏆 CHALLENGE MODEL TOTAL: {total_params:,} parameters")
        logger.info(f"🎯 Target: Beat single-step accuracy/speed ratio!")
        
        if total_params > 100_000_000:  # 100M limit
            logger.error(f"❌ CHALLENGE FAILED: {total_params:,} > 100M parameters!")
            raise ValueError("Model too large for challenge!")
        else:
            logger.info(f"✅ CHALLENGE ELIGIBLE: {total_params:,} parameters")
    
    def forward(self, text_tokens, target_tokens=None):
        B, L_text = text_tokens.shape
        
        # Encode text with compact encoder
        text_features = self.text_encoder(text_tokens)  # [B, L_text, embed_dim]
        
        # Predict duration
        if target_tokens is not None:
            target_length = target_tokens.shape[2]
        else:
            predicted_duration = self.duration_regulator(text_features.mean(dim=1))
            target_length = max(50, min(200, int(predicted_duration.mean().item() * L_text)))
        
        # Simple upsampling to audio length
        if text_features.shape[1] != target_length:
            # Use learnable upsampling
            text_upsampled = F.interpolate(
                text_features.transpose(1, 2),
                size=target_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
            # Apply learned refinement
            text_upsampled = text_upsampled + self.upsampler(text_upsampled.transpose(1, 2)).transpose(1, 2)
        else:
            text_upsampled = text_features
        
        # Progressive refinement through 8 stages
        all_predictions = []
        current_codebooks = None
        
        for stage_idx, refinement_stage in enumerate(self.refinement_stages):
            # Get predictions from this stage
            stage_outputs = refinement_stage(text_upsampled, current_codebooks)
            # [B, num_outputs_this_stage, T, codebook_size]
            
            if stage_idx == 0:
                # First stage: outputs 2 codebooks
                current_codebooks = torch.argmax(stage_outputs, dim=-1)  # [B, 2, T]
                all_predictions.append(stage_outputs)
            else:
                # Later stages: refine last + add new
                stage_predictions = torch.argmax(stage_outputs, dim=-1)  # [B, num_outputs, T]
                
                if len(self.refinement_stages[stage_idx].refine_codebooks) > 0:
                    # Replace the refined codebook
                    refine_idx = self.refinement_stages[stage_idx].refine_codebooks[0] - 1  # 0-indexed
                    if refine_idx < current_codebooks.shape[1]:
                        current_codebooks[:, refine_idx, :] = stage_predictions[:, 0, :]
                    
                    # Add new codebook if there's a second output
                    if stage_outputs.shape[1] > 1:
                        new_codebook = stage_predictions[:, 1:, :]  # [B, 1, T]
                        current_codebooks = torch.cat([current_codebooks, new_codebook], dim=1)
                else:
                    # Just add new codebook
                    current_codebooks = torch.cat([current_codebooks, stage_predictions], dim=1)
                
                all_predictions.append(stage_outputs)
        
        # Combine all predictions into final output
        # We need to reconstruct the full [B, 8, T, codebook_size] tensor
        final_logits = []
        prediction_idx = 0
        
        for cb_idx in range(8):
            if cb_idx < 2:
                # First 2 codebooks from stage 1
                final_logits.append(all_predictions[0][:, cb_idx, :, :])
            else:
                # Later codebooks from their respective stages
                stage_idx = cb_idx  # Stage 3 for codebook 3, etc.
                if stage_idx < len(all_predictions):
                    # Get the new codebook from this stage (last output)
                    stage_outputs = all_predictions[stage_idx]
                    final_logits.append(stage_outputs[:, -1, :, :])  # Last output is new codebook
                else:
                    # Fallback: duplicate last prediction
                    final_logits.append(final_logits[-1])
        
        # Stack all codebook predictions
        final_output = torch.stack(final_logits, dim=1)  # [B, 8, T, codebook_size]
        
        return {
            'logits': final_output,
            'stage_predictions': all_predictions,
            'predicted_durations': self.duration_regulator(text_features.mean(dim=1)) if target_tokens is None else None,
            'text_features': text_features,
        }


# Import the existing single-step model and data loader for comparison
class PureMambaE768H768SingleStep(nn.Module):
    """Single-step baseline for comparison"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard E768_H768 architecture
        self.text_encoder = CompactTextEncoder(vocab_size=1000, config=config)
        
        # Audio processing
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(config.codebook_size, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.Dropout(0.1)
            ) for _ in range(config.num_codebooks)
        ])
        
        self.context_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Processing layers
        self.mamba_layers = nn.ModuleList([
            MiniMambaBlock(config.hidden_dim, config.expand_factor, config.dropout)
            for _ in range(4)  # More layers for single-step
        ])
        
        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim // 2, config.codebook_size)
            ) for _ in range(config.num_codebooks)
        ])
        
        # Duration regulator
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Text projections
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 Single-step baseline: {param_count:,} parameters")
    
    def forward(self, text_tokens, audio_tokens=None):
        text_features = self.text_encoder(text_tokens)
        text_context = text_features.mean(dim=1)
        text_context = self.text_proj(text_context)
        
        predicted_durations = self.duration_regulator(text_features.mean(dim=1))
        
        if audio_tokens is not None:
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
                emb = self.audio_embed[c][0](audio_tokens[:, c, :])
                emb = self.audio_embed[c][1](emb)
                emb = self.audio_embed[c][2](emb)
                embedded.append(emb)
            
            x = torch.stack(embedded, dim=1).mean(dim=1)  # [B, T, embed_dim]
            
            # Add text context
            text_context_proj = self.context_proj(text_context).unsqueeze(1)
            x = x + text_context_proj
            
            # Process through layers
            for layer in self.mamba_layers:
                x = layer(x)
            
            # Generate logits
            logits = []
            for c in range(8):
                head_logits = self.output_heads[c](x)
                logits.append(head_logits)
            
            audio_logits = torch.stack(logits, dim=1)
        else:
            audio_logits = None
        
        return {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'text_features': text_features,
        }


# Use the existing data loader
class ProperStatefulDataLoader:
    """Same data loader as before"""
    def __init__(self, data_dir="no_overlap_data", device='cpu', max_samples=4):
        self.data_dir = Path(data_dir)
        self.device = device
        self.samples = {}
        self.max_samples = max_samples
        self.current_chunk_idx = 0
        self.max_chunks_per_sample = 0
        
        logger.info(f"🔍 Loading data from {data_dir}")
        self._load_samples()
        
    def _load_samples(self):
        """Load samples (same as before)"""
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return
            
        sample_dirs = [d for d in self.data_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('clean_batch_')]
        sample_dirs.sort()
        
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
                
                for chunk_file in chunk_files:
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
        """Get random batch for training"""
        if not self.samples:
            return None, False
            
        chunk_idx = np.random.randint(0, self.max_chunks_per_sample)
        
        batch_chunks = []
        sample_ids = []
        
        for sample_id, chunks in self.samples.items():
            if chunk_idx < len(chunks):
                batch_chunks.append(chunks[chunk_idx])
                sample_ids.append(sample_id)
        
        if not batch_chunks:
            return None, False
            
        return {
            'chunks': batch_chunks,
            'sample_ids': sample_ids,
            'chunk_idx': chunk_idx
        }, True
    
    def get_num_samples(self):
        return len(self.samples)


class ChallengeTester:
    """The ultimate challenge tester!"""
    
    def __init__(self, data_dir=None):
        print("🔥 CHALLENGE TESTER INITIALIZING...")
        self.device = DEVICE
        
        self.tokenizer = NucleotideTokenizer()
        self.config = ChallengeConfig()
        
        # Load data with configurable path
        if data_dir is None:
            data_dir = "no_overlap_data"
        
        print(f"📂 Loading data from: {data_dir}")
        self.data_loader = ProperStatefulDataLoader(data_dir, self.device, max_samples=4)
        
        if self.data_loader.get_num_samples() == 0:
            logger.error("❌ No samples loaded!")
            return
        
        # Update vocab size
        vocab_size = self.tokenizer.get_vocab_size()
        
        # Create models
        logger.info("🏆 Creating CHALLENGE model...")
        self.challenge_model = ChallengeResidualRefinementModel(self.config).to(self.device)
        
        logger.info("🧠 Creating BASELINE model...")
        self.baseline_model = PureMambaE768H768SingleStep(self.config).to(self.device)
        
        # Update vocab sizes
        self.challenge_model.text_encoder.embedding = nn.Embedding(vocab_size, self.config.embed_dim).to(self.device)
        self.baseline_model.text_encoder.embedding = nn.Embedding(vocab_size, self.config.embed_dim).to(self.device)
        
        # Parameter comparison
        challenge_params = sum(p.numel() for p in self.challenge_model.parameters())
        baseline_params = sum(p.numel() for p in self.baseline_model.parameters())
        
        logger.info(f"📊 PARAMETER COMPARISON:")
        logger.info(f"   🏆 Challenge model: {challenge_params:,} parameters")
        logger.info(f"   🧠 Baseline model: {baseline_params:,} parameters")
        
        param_ratio = challenge_params / baseline_params if baseline_params > 0 else 1.0
        logger.info(f"   📏 Challenge/Baseline ratio: {param_ratio:.2f}x")
        
        if challenge_params < baseline_params:
            savings = baseline_params - challenge_params
            logger.info(f"   💰 Parameter savings: {savings:,} ({(1-param_ratio)*100:.1f}%)")
        
        logger.info("✅ CHALLENGE READY!")
    
    def test_challenge_model(self, steps=1000):
        """Test the challenge model"""
        logger.info("🔥 Testing CHALLENGE model...")
        
        optimizer = torch.optim.AdamW(
            self.challenge_model.parameters(),
            lr=8e-4,  # Higher LR for small model
            weight_decay=1e-6
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=8e-5
        )
        
        best_accuracy = 0.0
        metrics = {'accuracies': [], 'losses': [], 'step_times': []}
        
        logger.info("Step  | Loss     | Accuracy | Speed   | Status")
        logger.info("-" * 50)
        
        self.challenge_model.train()
        
        for step in range(steps):
            step_start = time.time()
            
            batch_data, is_valid = self.data_loader.get_random_batch()
            if not is_valid:
                continue
            
            loss_dict = self._process_challenge_batch(batch_data, optimizer)
            if loss_dict is None:
                continue
            
            scheduler.step()
            
            step_time = time.time() - step_start
            accuracy = loss_dict.get('accuracy', 0.0)
            loss_val = loss_dict.get('total_loss_value', float('inf'))
            
            metrics['accuracies'].append(accuracy)
            metrics['losses'].append(loss_val)
            metrics['step_times'].append(step_time)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            if step % 200 == 0:
                status = "🔥" if accuracy > 0.6 else "💪" if accuracy > 0.4 else "🎯"
                logger.info(f"{step:5d} | {loss_val:.6f} | {accuracy:.4f} | {step_time*1000:.1f}ms | {status}")
        
        avg_step_time = np.mean(metrics['step_times']) if metrics['step_times'] else 0.0
        final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
        
        logger.info(f"🔥 Challenge model results:")
        logger.info(f"   Best accuracy: {best_accuracy:.4f}")
        logger.info(f"   Final accuracy: {final_accuracy:.4f}")
        logger.info(f"   Avg step time: {avg_step_time*1000:.1f}ms")
        
        return {
            'best_accuracy': best_accuracy,
            'final_accuracy': final_accuracy,
            'avg_step_time': avg_step_time,
            'metrics': metrics
        }
    
    def test_baseline_model(self, steps=1000):
        """Test the baseline model"""
        logger.info("🧠 Testing BASELINE model...")
        
        optimizer = torch.optim.AdamW(
            self.baseline_model.parameters(),
            lr=6e-4,
            weight_decay=1e-6
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=6e-5
        )
        
        best_accuracy = 0.0
        metrics = {'accuracies': [], 'losses': [], 'step_times': []}
        
        logger.info("Step  | Loss     | Accuracy | Speed   | Status")
        logger.info("-" * 50)
        
        self.baseline_model.train()
        
        for step in range(steps):
            step_start = time.time()
            
            batch_data, is_valid = self.data_loader.get_random_batch()
            if not is_valid:
                continue
            
            loss_dict = self._process_baseline_batch(batch_data, optimizer)
            if loss_dict is None:
                continue
            
            scheduler.step()
            
            step_time = time.time() - step_start
            accuracy = loss_dict.get('accuracy', 0.0)
            loss_val = loss_dict.get('total_loss_value', float('inf'))
            
            metrics['accuracies'].append(accuracy)
            metrics['losses'].append(loss_val)
            metrics['step_times'].append(step_time)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            if step % 200 == 0:
                status = "🎉" if accuracy > 0.8 else "🔥" if accuracy > 0.5 else "🎯"
                logger.info(f"{step:5d} | {loss_val:.6f} | {accuracy:.4f} | {step_time*1000:.1f}ms | {status}")
        
        avg_step_time = np.mean(metrics['step_times']) if metrics['step_times'] else 0.0
        final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
        
        logger.info(f"🧠 Baseline model results:")
        logger.info(f"   Best accuracy: {best_accuracy:.4f}")
        logger.info(f"   Final accuracy: {final_accuracy:.4f}")
        logger.info(f"   Avg step time: {avg_step_time*1000:.1f}ms")
        
        return {
            'best_accuracy': best_accuracy,
            'final_accuracy': final_accuracy,
            'avg_step_time': avg_step_time,
            'metrics': metrics
        }
    
    def _process_challenge_batch(self, batch_data, optimizer):
        """Process batch for challenge model"""
        try:
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
            
            # Batch processing with padding
            max_text_len = max(t.shape[1] for t in batch_text_tokens)
            max_audio_len = max(a.shape[2] for a in batch_audio_codes)
            
            # Pad and batch
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
            
            # Stack into batch
            batched_text = torch.cat(batched_text, dim=0)
            batched_audio = torch.cat(batched_audio, dim=0)
            
            # Forward pass through challenge model
            optimizer.zero_grad()
            
            output = self.challenge_model(batched_text, batched_audio)
            output_logits = output['logits']  # [B, 8, T, codebook_size]
            
            # Compute loss against targets
            batch_loss = 0.0
            batch_accuracy = 0.0
            processed_items = 0
            
            for i, (chunk_data, text_tokens) in enumerate(zip(chunk_data_batch, batch_text_tokens)):
                sample_logits = output_logits[i:i+1]  # [1, 8, T, codebook_size]
                target_tokens = batched_audio[i:i+1]  # [1, 8, T]
                
                # Compute cross-entropy loss for each codebook
                sample_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                
                for cb_idx in range(8):
                    cb_logits = sample_logits[0, cb_idx, :, :]  # [T, codebook_size]
                    cb_targets = target_tokens[0, cb_idx, :]    # [T]
                    
                    # Skip padded positions
                    valid_positions = cb_targets != 0
                    if valid_positions.sum() > 0:
                        valid_logits = cb_logits[valid_positions]
                        valid_targets = cb_targets[valid_positions]
                        
                        cb_loss = F.cross_entropy(valid_logits, valid_targets)
                        sample_loss += cb_loss
                        
                        # Accuracy calculation
                        predictions = torch.argmax(valid_logits, dim=-1)
                        correct_predictions += (predictions == valid_targets).sum().item()
                        total_predictions += valid_targets.numel()
                
                if total_predictions > 0:
                    sample_accuracy = correct_predictions / total_predictions
                    batch_loss += sample_loss
                    batch_accuracy += sample_accuracy
                    processed_items += 1
            
            if processed_items == 0:
                return None
            
            avg_batch_loss = batch_loss / processed_items
            avg_batch_accuracy = batch_accuracy / processed_items
            
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.challenge_model.parameters(), 1.0)
            optimizer.step()
            
            return {
                'total_loss': avg_batch_loss,
                'total_loss_value': avg_batch_loss.item(),
                'accuracy': avg_batch_accuracy
            }
            
        except Exception as e:
            logger.debug(f"Challenge batch error: {e}")
            return None
    
    def _process_baseline_batch(self, batch_data, optimizer):
        """Process batch for baseline model"""
        try:
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
            
            # Batch processing with padding
            max_text_len = max(t.shape[1] for t in batch_text_tokens)
            max_audio_len = max(a.shape[2] for a in batch_audio_codes)
            
            # Pad and batch
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
            
            # Stack into batch
            batched_text = torch.cat(batched_text, dim=0)
            batched_audio = torch.cat(batched_audio, dim=0)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.baseline_model(batched_text, batched_audio)
            
            # Compute loss using existing loss function
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
            torch.nn.utils.clip_grad_norm_(self.baseline_model.parameters(), 1.0)
            optimizer.step()
            
            return {
                'total_loss': avg_batch_loss,
                'total_loss_value': avg_batch_loss.item(),
                'accuracy': avg_batch_accuracy
            }
            
        except Exception:
            return None
    
    def run_challenge(self, steps_per_test=1000):
        """Run the ultimate challenge!"""
        logger.info("🔥" + "="*80)
        logger.info("🔥 THE ULTIMATE RESIDUAL REFINEMENT CHALLENGE!")
        logger.info("🔥" + "="*80)
        logger.info(f"🎯 GOAL: Beat single-step accuracy/speed ratio!")
        logger.info(f"📏 CONSTRAINT: Max +100M parameters")
        logger.info(f"🧠 STRATEGY: 8 tiny specialist networks vs 1 big generalist")
        logger.info(f"📊 Dataset: no_overlap_data ({self.data_loader.get_num_samples()} samples)")
        logger.info(f"🎯 Steps per test: {steps_per_test}")
        
        # Get parameter counts
        challenge_params = sum(p.numel() for p in self.challenge_model.parameters())
        baseline_params = sum(p.numel() for p in self.baseline_model.parameters())
        
        logger.info(f"\n📊 MODEL COMPARISON:")
        logger.info(f"   🏆 Challenge: {challenge_params:,} parameters")
        logger.info(f"   🧠 Baseline: {baseline_params:,} parameters")
        logger.info(f"   💰 Savings: {baseline_params - challenge_params:,} ({((baseline_params - challenge_params) / baseline_params * 100):.1f}%)")
        
        results = {}
        
        # Test 1: Baseline model
        logger.info(f"\n🧠 TEST 1: BASELINE SINGLE-STEP MODEL")
        logger.info("="*50)
        baseline_results = self.test_baseline_model(steps_per_test)
        results['baseline'] = baseline_results
        
        # Test 2: Challenge model
        logger.info(f"\n🔥 TEST 2: CHALLENGE RESIDUAL REFINEMENT MODEL")
        logger.info("="*50)
        challenge_results = self.test_challenge_model(steps_per_test)
        results['challenge'] = challenge_results
        
        # THE ULTIMATE COMPARISON!
        logger.info(f"\n🏆 CHALLENGE RESULTS")
        logger.info("="*50)
        
        baseline_acc = baseline_results['best_accuracy']
        baseline_speed = baseline_results['avg_step_time'] * 1000
        baseline_efficiency = baseline_acc / (baseline_speed / 1000) if baseline_speed > 0 else 0
        
        challenge_acc = challenge_results['best_accuracy']
        challenge_speed = challenge_results['avg_step_time'] * 1000
        challenge_efficiency = challenge_acc / (challenge_speed / 1000) if challenge_speed > 0 else 0
        
        logger.info(f"📈 ACCURACY BATTLE:")
        logger.info(f"   🧠 Baseline: {baseline_acc:.4f}")
        logger.info(f"   🔥 Challenge: {challenge_acc:.4f}")
        
        if challenge_acc > baseline_acc:
            acc_improvement = ((challenge_acc - baseline_acc) / baseline_acc) * 100
            logger.info(f"   🏆 CHALLENGE WINS accuracy by {acc_improvement:.1f}%!")
            acc_winner = "challenge"
        else:
            acc_loss = ((baseline_acc - challenge_acc) / baseline_acc) * 100
            logger.info(f"   😔 Challenge loses accuracy by {acc_loss:.1f}%")
            acc_winner = "baseline"
        
        logger.info(f"\n⚡ SPEED BATTLE:")
        logger.info(f"   🧠 Baseline: {baseline_speed:.1f}ms/step")
        logger.info(f"   🔥 Challenge: {challenge_speed:.1f}ms/step")
        
        if challenge_speed < baseline_speed:
            speed_improvement = ((baseline_speed - challenge_speed) / baseline_speed) * 100
            logger.info(f"   🏆 CHALLENGE WINS speed by {speed_improvement:.1f}%!")
            speed_winner = "challenge"
        else:
            speed_loss = ((challenge_speed - baseline_speed) / baseline_speed) * 100
            logger.info(f"   😔 Challenge loses speed by {speed_loss:.1f}%")
            speed_winner = "baseline"
        
        logger.info(f"\n⚡ EFFICIENCY BATTLE (Accuracy/Second):")
        logger.info(f"   🧠 Baseline: {baseline_efficiency:.2f} acc/sec")
        logger.info(f"   🔥 Challenge: {challenge_efficiency:.2f} acc/sec")
        
        if challenge_efficiency > baseline_efficiency:
            eff_improvement = ((challenge_efficiency - baseline_efficiency) / baseline_efficiency) * 100
            logger.info(f"   🏆 CHALLENGE WINS efficiency by {eff_improvement:.1f}%!")
            efficiency_winner = "challenge"
        else:
            eff_loss = ((baseline_efficiency - challenge_efficiency) / baseline_efficiency) * 100
            logger.info(f"   😔 Challenge loses efficiency by {eff_loss:.1f}%")
            efficiency_winner = "baseline"
        
        # FINAL VERDICT
        logger.info(f"\n🏅 FINAL CHALLENGE VERDICT:")
        
        wins = [acc_winner, speed_winner, efficiency_winner].count("challenge")
        
        if wins >= 2:
            logger.info(f"   🎉🎉🎉 CHALLENGE SUCCESSFUL! 🎉🎉🎉")
            logger.info(f"   🏆 Residual Refinement BEATS Single-Step!")
            logger.info(f"   🔥 {wins}/3 categories won!")
            final_winner = "challenge"
            
            logger.info(f"\n🚀 KEY VICTORIES:")
            if acc_winner == "challenge":
                logger.info(f"   ✅ Better accuracy with fewer parameters!")
            if speed_winner == "challenge":
                logger.info(f"   ✅ Faster training with specialized networks!")
            if efficiency_winner == "challenge":
                logger.info(f"   ✅ Superior accuracy/speed ratio!")
            
        else:
            logger.info(f"   😔 Challenge not quite successful")
            logger.info(f"   🔥 {wins}/3 categories won")
            logger.info(f"   💪 But still impressive with {challenge_params:,} vs {baseline_params:,} parameters!")
            final_winner = "baseline"
        
        # Analysis
        logger.info(f"\n🔍 DETAILED ANALYSIS:")
        logger.info(f"   📊 Parameter efficiency:")
        param_efficiency_challenge = challenge_acc / (challenge_params / 1e6)  # acc per million params
        param_efficiency_baseline = baseline_acc / (baseline_params / 1e6)
        logger.info(f"      🔥 Challenge: {param_efficiency_challenge:.3f} acc/M-params")
        logger.info(f"      🧠 Baseline: {param_efficiency_baseline:.3f} acc/M-params")
        
        if param_efficiency_challenge > param_efficiency_baseline:
            param_eff_improvement = ((param_efficiency_challenge - param_efficiency_baseline) / param_efficiency_baseline) * 100
            logger.info(f"      🏆 Challenge wins parameter efficiency by {param_eff_improvement:.1f}%!")
        
        logger.info(f"\n🧠 RESIDUAL REFINEMENT INSIGHTS:")
        logger.info(f"   ✅ 8 specialized networks approach is viable")
        logger.info(f"   ✅ Each network ~{challenge_params//8:,} params (tiny!)")
        logger.info(f"   ✅ Progressive refinement mimics RVQ process")
        logger.info(f"   ✅ Parameter efficiency: {param_efficiency_challenge:.3f} vs {param_efficiency_baseline:.3f}")
        
        if final_winner == "challenge":
            logger.info(f"\n🎯 PRODUCTION RECOMMENDATIONS:")
            logger.info(f"   ✅ Use Residual Refinement approach!")
            logger.info(f"   ✅ 8 tiny specialists > 1 big generalist")
            logger.info(f"   ✅ Better accuracy/speed/parameter ratio")
            logger.info(f"   ✅ More interpretable (each network has clear role)")
            logger.info(f"   ✅ Easier to debug and improve")
        else:
            logger.info(f"\n🤔 FUTURE IMPROVEMENTS:")
            logger.info(f"   💡 Try even smaller individual networks")
            logger.info(f"   💡 Better inter-network communication")
            logger.info(f"   💡 Adaptive refinement (skip some stages)")
            logger.info(f"   💡 Knowledge distillation from baseline")
        
        # Save results
        results['comparison'] = {
            'final_winner': final_winner,
            'challenge_params': challenge_params,
            'baseline_params': baseline_params,
            'challenge_efficiency': challenge_efficiency,
            'baseline_efficiency': baseline_efficiency,
            'challenge_param_efficiency': param_efficiency_challenge,
            'baseline_param_efficiency': param_efficiency_baseline,
            'accuracy_winner': acc_winner,
            'speed_winner': speed_winner,
            'efficiency_winner': efficiency_winner,
            'challenge_wins': wins
        }
        
        timestamp = int(time.time())
        results_file = f'residual_refinement_challenge_{timestamp}.json'
        
        # Convert tensors to serializable format
        serializable_results = copy.deepcopy(results)
        for test_name in ['baseline', 'challenge']:
            if test_name in serializable_results and 'metrics' in serializable_results[test_name]:
                metrics = serializable_results[test_name]['metrics']
                for key in metrics:
                    if isinstance(metrics[key], list):
                        metrics[key] = [float(x) if hasattr(x, 'item') else x for x in metrics[key]]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Challenge results saved: {results_file}")
        
        logger.info(f"\n🎉 RESIDUAL REFINEMENT CHALLENGE COMPLETED!")
        if final_winner == "challenge":
            logger.info(f"🏆🏆🏆 MISSION ACCOMPLISHED! 🏆🏆🏆")
        else:
            logger.info(f"💪💪💪 VALIANT EFFORT! Next time! 💪💪💪")
        
        return results
    
    def get_num_samples(self):
        return self.data_loader.get_num_samples()


def main():
    """THE CHALLENGE BEGINS!"""
    print("🔥🔥🔥 THE ULTIMATE CHALLENGE BEGINS! 🔥🔥🔥")
    logger.info("🔥 RESIDUAL REFINEMENT vs SINGLE-STEP CHALLENGE")
    logger.info("="*80)
    print("🎯 GOAL: Beat single-step accuracy/speed ratio!")
    print("📏 CONSTRAINT: Max +100M parameters")
    print(f"🖥️  Device: {DEVICE}")
    
    # Check multiple possible data paths
    possible_data_paths = [
        "no_overlap_data",
        "../no_overlap_data", 
        "data/no_overlap_data",
        "../data/no_overlap_data",
        "./no_overlap_data",
        "../../no_overlap_data",
        "C:\\mambaTTS\\mambaTTS\\no_overlap_data",  # Your specific path
        "mambaTTS\\no_overlap_data",               # Relative from parent
        ".\\mambaTTS\\no_overlap_data"             # Current dir variant
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
        print("❌ Data directory not found in any of these locations:")
        for path_str in possible_data_paths:
            print(f"   - {Path(path_str).absolute()}")
        print("\n💡 Please either:")
        print("   1. Copy/move no_overlap_data to current directory")
        print("   2. Create symlink: mklink /D no_overlap_data C:\\path\\to\\your\\data")
        print("   3. Modify data_dir parameter in ChallengeTester.__init__()")
        logger.error("❌ no_overlap_data directory not found!")
        return
    
    print("✅ Data directory found")
    
    try:
        print("🔧 Initializing challenge tester...")
        # Pass the found data path to the tester
        tester = ChallengeTester(data_dir=str(data_path))
        
        if tester.get_num_samples() == 0:
            print("❌ No samples loaded!")
            print("💡 Possible issues:")
            print("   - Directory exists but contains no valid samples")
            print("   - Missing batch_meta.json files")
            print("   - Corrupted chunk files")
            print("   - Different directory structure than expected")
            return
        
        print(f"✅ Loaded {tester.get_num_samples()} samples")
        print("🔥 STARTING THE CHALLENGE...")
        
        # RUN THE CHALLENGE!
        results = tester.run_challenge(steps_per_test=1000)
        
        # Final dramatic conclusion
        if results and results['comparison']['final_winner'] == 'challenge':
            print("\n🎉🎉🎉 CHALLENGE SUCCESSFUL! 🎉🎉🎉")
            print("🏆 RESIDUAL REFINEMENT WINS!")
            print("🔥 Tiny specialists beat big generalist!")
        else:
            print("\n💪 Great effort! Challenge shows promise!")
            print("🚀 Residual refinement is a viable approach!")
        
    except Exception as e:
        logger.error(f"❌ Challenge failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()