#!/usr/bin/env python3
"""
Embed Dimension Greedy Search Experiment
========================================
Unified embedding architecture with systematic embed_dim optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("🔍 EMBED DIMENSION GREEDY SEARCH - UNIFIED ARCHITECTURE")
print("=" * 65)

# Import existing components
try:
    from nucleotide_tokenizer import NucleotideTokenizer
    from losses import compute_combined_loss
    logger.info("✅ Imported tokenizer and losses")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    exit(1)


@dataclass
class EmbedDimOptimizationConfig:
    """Configuration for embed_dim optimization experiments"""
    name: str
    embed_dim: int = 384      # VARIABLE TO OPTIMIZE
    hidden_dim: int = 768     # FIXED optimal from Phase 3
    num_layers: int = 4       # FIXED optimal
    expand_factor: float = 1.5 # FIXED optimal
    num_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1
    
    def get_embedding_params_estimate(self) -> Dict[str, int]:
        """Estimate embedding parameters for this config"""
        vocab_size = 131  # From tokenizer
        
        # Text embeddings
        text_embed = vocab_size * self.embed_dim
        text_proj = self.embed_dim * self.hidden_dim if self.embed_dim != self.hidden_dim else 0
        
        # Audio embeddings (8 codebooks)
        audio_embed = 8 * self.codebook_size * self.embed_dim
        audio_proj = 8 * self.embed_dim * self.hidden_dim if self.embed_dim != self.hidden_dim else 0
        
        # Positional embeddings
        pos_embed = 2048 * self.embed_dim
        pos_proj = self.embed_dim * self.hidden_dim if self.embed_dim != self.hidden_dim else 0
        
        # Mamba processing (unchanged)
        mamba_params = self.num_layers * (self.hidden_dim * self.hidden_dim * 4)
        audio_processor = 4 * (self.hidden_dim * self.hidden_dim * 3)
        others = self.hidden_dim * 1000
        
        total_embed = text_embed + text_proj + audio_embed + audio_proj + pos_embed + pos_proj
        total_processing = mamba_params + audio_processor + others
        
        return {
            'text_embed': text_embed,
            'text_proj': text_proj,
            'audio_embed': audio_embed,
            'audio_proj': audio_proj, 
            'pos_embed': pos_embed,
            'pos_proj': pos_proj,
            'total_embedding': total_embed,
            'total_processing': total_processing,
            'total_model': total_embed + total_processing,
            'embed_ratio': self.embed_dim / self.hidden_dim
        }


class UnifiedMambaBlock(nn.Module):
    """Pure Mamba block with unified embedding support"""
    def __init__(self, d_model, expand_factor=1.5, dropout=0.1, reverse=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand_factor)
        self.reverse = reverse
        
        # Core Mamba components (L4_H768_E1.5 optimal design)
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
        
    def forward(self, x, return_state_info=False):
        B, L, D = x.shape
        residual = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # Reverse for backward processing
        if self.reverse:
            x = torch.flip(x, dims=[1])
        
        # Input projection and split
        x_proj = self.in_proj(x)  # [B, L, 2*d_inner]
        x1, x2 = x_proj.chunk(2, dim=-1)  # Each [B, L, d_inner]
        
        # Convolution
        x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)
        
        # SSM processing
        x1_ssm = self.activation(x1_conv)
        dt = self.dt_proj(x1_ssm)
        dt = F.softplus(dt)
        
        # Simplified but effective state space (stateless for speed)
        x1_processed = x1_ssm * torch.sigmoid(dt)
        
        # Gating mechanism
        x_gated = x1_processed * torch.sigmoid(x2)
        
        # Output projection with layer scaling
        output = self.out_proj(x_gated)
        output = output * self.layer_scale
        
        # Dropout and reverse back
        output = self.dropout(output)
        if self.reverse:
            output = torch.flip(output, dims=[1])
        
        final_output = output + residual
        
        return final_output


class UnifiedMambaTextEncoder(nn.Module):
    """UNIFIED Text encoder with proper embed_dim usage"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, expand_factor, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # UNIFIED: Text embedding uses embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # UNIFIED: Project to hidden_dim if different
        self.embed_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        
        # UNIFIED: Positional encoding uses embed_dim
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, embed_dim) * 0.02)
        self.pos_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        
        # Stack of Pure Mamba blocks
        self.layers = nn.ModuleList([
            UnifiedMambaBlock(hidden_dim, expand_factor, dropout, reverse=False)
            for _ in range(num_layers)
        ])
        
        # Final processing
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens, return_sequence=True, return_state_info=False):
        B, L = tokens.shape
        
        # UNIFIED: Embeddings in embed_dim space
        x = self.embedding(tokens)  # [B, L, embed_dim]
        x = self.embed_proj(x)      # [B, L, hidden_dim]
        
        # UNIFIED: Positional encoding in embed_dim space
        if L <= self.pos_encoding.shape[1]:
            pos_emb = self.pos_encoding[:, :L, :]  # [1, L, embed_dim]
            pos_emb = self.pos_proj(pos_emb)       # [1, L, hidden_dim]
            x = x + pos_emb
        
        x = self.dropout(x)
        
        # Process through Pure Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        result = x if return_sequence else x.mean(dim=1)
        return result


class UnifiedMambaAudioProcessor(nn.Module):
    """UNIFIED Audio processor with proper embed_dim usage"""
    def __init__(self, embed_dim, hidden_dim, num_codebooks=8, codebook_size=1024, expand_factor=1.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        
        # UNIFIED: Audio embeddings use embed_dim
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(codebook_size, embed_dim),        # ← UNIFIED: embed_dim
                nn.Linear(embed_dim, hidden_dim),             # ← UNIFIED: project to hidden_dim
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_codebooks)
        ])
        
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Pure Mamba processing layers
        self.mamba_layers = nn.ModuleList([
            UnifiedMambaBlock(hidden_dim, expand_factor, dropout=0.1)
            for _ in range(4)
        ])
        
        # Efficient output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, codebook_size)
            ) for _ in range(num_codebooks)
        ])
    
    def forward(self, audio_tokens, text_context, return_state_info=False):
        B, C, T = audio_tokens.shape
        
        # Ensure 8 codebooks
        if C < 8:
            padding = torch.zeros(B, 8 - C, T, dtype=audio_tokens.dtype, device=audio_tokens.device)
            audio_tokens = torch.cat([audio_tokens, padding], dim=1)
        elif C > 8:
            audio_tokens = audio_tokens[:, :8, :]
        
        # UNIFIED: Embed each codebook through embed_dim → hidden_dim
        embedded = []
        for c in range(8):
            # embed_dim → hidden_dim pipeline
            emb = self.audio_embed[c][0](audio_tokens[:, c, :])  # → embed_dim
            emb = self.audio_embed[c][1](emb)                    # → hidden_dim
            emb = self.audio_embed[c][2](emb)                    # LayerNorm
            emb = self.audio_embed[c][3](emb)                    # Dropout
            embedded.append(emb)
        
        # Combine embeddings
        x = torch.stack(embedded, dim=1).mean(dim=1)  # [B, T, hidden_dim]
        
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


class UnifiedPureMambaModel(nn.Module):
    """
    UNIFIED Pure Mamba model with proper embed_dim usage throughout
    """
    def __init__(self, config: EmbedDimOptimizationConfig):
        super().__init__()
        self.config = config
        
        # All components use UNIFIED embedding architecture
        self.text_encoder = UnifiedMambaTextEncoder(
            vocab_size=1000,  # Will be updated
            embed_dim=config.embed_dim,        # ← UNIFIED
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            expand_factor=config.expand_factor,
            dropout=config.dropout
        )
        
        # Duration regulator
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # UNIFIED Audio processor
        self.audio_processor = UnifiedMambaAudioProcessor(
            embed_dim=config.embed_dim,        # ← UNIFIED
            hidden_dim=config.hidden_dim,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            expand_factor=config.expand_factor
        )
        
        # Text projections
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Log model info with embedding breakdown
        embed_params = config.get_embedding_params_estimate()
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"🧠 {config.name}: {total_params:,} parameters")
        logger.info(f"   📏 E{config.embed_dim}_H{config.hidden_dim}_L{config.num_layers}")
        logger.info(f"   📊 Embedding params: {embed_params['total_embedding']:,} ({embed_params['total_embedding']/total_params*100:.1f}%)")
        logger.info(f"   🔢 Embed ratio: {embed_params['embed_ratio']:.2f}")
    
    def forward(self, text_tokens, audio_tokens=None, return_state_info=False):
        batch_size = text_tokens.shape[0]
        
        # Process text through UNIFIED architecture
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        # Duration prediction
        predicted_durations = self.duration_regulator(text_features.mean(dim=1))
        
        # Audio processing through UNIFIED architecture
        if audio_tokens is not None:
            regulated_context = text_context
            audio_logits = self.audio_processor(audio_tokens, regulated_context)
        else:
            audio_logits = None
        
        result = {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'text_features': text_features,
        }
        
        return result


class EmbedDimGreedySearch:
    """Greedy search for optimal embed_dim configuration"""
    
    def __init__(self, tokenizer, data_loader, device):
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = device
        
        # Phase 1: Wide range search
        self.phase_1_configs = [
            EmbedDimOptimizationConfig(name="E256_H768", embed_dim=256),
            EmbedDimOptimizationConfig(name="E512_H768", embed_dim=512), 
            EmbedDimOptimizationConfig(name="E768_H768", embed_dim=768),
        ]
        
        # Will be set based on Phase 1 results
        self.phase_2_configs = []
        
    def run_greedy_search(self, steps_per_config=500):
        """Complete greedy search: Phase 1 + Phase 2"""
        logger.info("🔍 STARTING EMBED_DIM GREEDY SEARCH")
        logger.info("=" * 50)
        
        # Phase 1: Wide range search
        logger.info("📊 PHASE 1: Wide Range Search")
        logger.info(f"Configs: {[c.name for c in self.phase_1_configs]}")
        
        phase_1_results = {}
        for config in self.phase_1_configs:
            logger.info(f"\n🔬 Testing {config.name}...")
            result = self._run_single_experiment(config, steps_per_config, "PHASE_1")
            phase_1_results[config.name] = result
            
            # Log immediate results - SAFE version
            if result.get('success', False):
                logger.info(f"✅ {config.name}: {result['best_accuracy']:.3f} accuracy, "
                          f"{result['avg_step_time']*1000:.1f}ms/step")
            else:
                logger.error(f"❌ {config.name}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Analyze Phase 1 and select best region
        best_region = self._analyze_phase_1_results(phase_1_results)
        
        # Phase 2: Refined search around winner
        logger.info(f"\n🎯 PHASE 2: Refined Search (Best region: {best_region})")
        self.phase_2_configs = self._generate_phase_2_configs(best_region)
        logger.info(f"Configs: {[c.name for c in self.phase_2_configs]}")
        
        phase_2_results = {}
        for config in self.phase_2_configs:
            logger.info(f"\n🔬 Testing {config.name}...")
            result = self._run_single_experiment(config, steps_per_config, "PHASE_2")
            phase_2_results[config.name] = result
            
            # Log immediate results - SAFE version  
            if result.get('success', False):
                logger.info(f"✅ {config.name}: {result['best_accuracy']:.3f} accuracy, "
                          f"{result['avg_step_time']*1000:.1f}ms/step")
            else:
                logger.error(f"❌ {config.name}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Final analysis
        all_results = {**phase_1_results, **phase_2_results}
        final_winner = self._analyze_final_results(all_results)
        
        return {
            'phase_1_results': phase_1_results,
            'phase_2_results': phase_2_results,
            'all_results': all_results,
            'final_winner': final_winner,
            'best_region': best_region
        }
    
    def _run_single_experiment(self, config: EmbedDimOptimizationConfig, test_steps: int, phase: str) -> Dict:
        """Run single embed_dim experiment"""
        start_time = time.time()
        
        try:
            # Create model with UNIFIED embedding architecture
            model = UnifiedPureMambaModel(config).to(self.device)
            
            # Update vocab size
            actual_vocab_size = self.tokenizer.get_vocab_size()
            model.text_encoder.embedding = nn.Embedding(actual_vocab_size, config.embed_dim).to(self.device)
            
            # Get actual parameter count
            param_count = sum(p.numel() for p in model.parameters())  # ← FIXED: model.parameters(), not self.parameters()
            embed_params = config.get_embedding_params_estimate()
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=6e-4,
                weight_decay=1e-6,
                betas=(0.9, 0.95)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=test_steps, eta_min=6e-5
            )
            
            # Tracking metrics
            metrics = {
                'accuracies': [],
                'losses': [],
                'step_times': [],
                'milestones': {}
            }
            
            best_accuracy = 0.0
            
            # Training loop (stateless for speed)
            model.train()
            for step in range(test_steps):
                step_start = time.time()
                
                try:
                    # Use stateless processing for speed
                    loss_dict, _ = self._process_batch(model, optimizer)
                    
                    if loss_dict is None:
                        continue
                    
                    scheduler.step()
                    
                    # Track metrics
                    step_time = time.time() - step_start
                    current_accuracy = loss_dict.get('accuracy', 0.0)
                    current_loss = loss_dict.get('total_loss_value', float('inf'))
                    
                    metrics['accuracies'].append(current_accuracy)
                    metrics['losses'].append(current_loss)
                    metrics['step_times'].append(step_time)
                    
                    # Update best accuracy
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    
                    # Milestone tracking
                    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                        milestone_key = f"{threshold:.1%}"
                        if milestone_key not in metrics['milestones'] and current_accuracy >= threshold:
                            metrics['milestones'][milestone_key] = step
                    
                    # Progress logging
                    if step % 100 == 0 or current_accuracy > 0.7:
                        logger.info(f"   Step {step:3d}: Loss={current_loss:.4f}, Acc={current_accuracy:.4f}")
                    
                    # Early stopping for very high accuracy
                    if current_accuracy > 0.9 and step > 300:  # ← Adjusted for 1000 steps
                        logger.info(f"   🎉 Early success at step {step}!")
                        break
                        
                except Exception as e:
                    logger.warning(f"Step {step} failed: {e}")
                    continue
            
            # Calculate final metrics
            training_time = time.time() - start_time
            final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
            avg_step_time = np.mean(metrics['step_times']) if metrics['step_times'] else 0.0
            
            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(
                best_accuracy, avg_step_time, param_count
            )
            
            result = {
                'experiment_type': f"{phase}_{config.name}",
                'config': asdict(config),
                'success': len(metrics['accuracies']) > test_steps * 0.5,
                'training_time': training_time,
                'final_accuracy': final_accuracy,
                'best_accuracy': best_accuracy,
                'avg_step_time': avg_step_time,
                'param_count': param_count,
                'embed_params': embed_params,
                'efficiency_score': efficiency_score,
                'milestones': metrics['milestones'],
                'total_steps': len(metrics['accuracies'])
            }
            
            logger.info(f"✅ {config.name} Results:")
            logger.info(f"   Best accuracy: {best_accuracy:.4f}")
            logger.info(f"   Efficiency score: {efficiency_score:.4f}")
            logger.info(f"   Embedding params: {embed_params['total_embedding']:,}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {config.name} experiment failed: {e}")
            return {
                'experiment_type': f"{phase}_{config.name}",
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    def _process_batch(self, model, optimizer):
        """Process single batch (stateless for speed)"""
        try:
            batch_data, is_valid = self.data_loader.get_random_batch()
            if not is_valid:
                return None, None
                
            chunks = batch_data['chunks']
            batch_size = len(chunks)
            
            # Prepare batch tensors
            text_tokens_batch = []
            audio_codes_batch = []
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
                        
                    text_tokens_batch.append(text_tokens)
                    audio_codes_batch.append(audio_codes)
                    chunk_data_batch.append(chunk_data)
                    
                except Exception as e:
                    continue
            
            if not text_tokens_batch:
                return None, None
            
            # Batch processing with padding
            max_text_len = max(t.shape[1] for t in text_tokens_batch)
            max_audio_len = max(a.shape[2] for a in audio_codes_batch)
            
            batch_text_tokens = []
            batch_audio_codes = []
            
            for text_tokens, audio_codes in zip(text_tokens_batch, audio_codes_batch):
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
                
                batch_text_tokens.append(text_tokens)
                batch_audio_codes.append(audio_codes)
            
            # Stack into proper batch
            batched_text = torch.cat(batch_text_tokens, dim=0)
            batched_audio = torch.cat(batch_audio_codes, dim=0)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batched_text, batched_audio)
            
            # Compute loss for entire batch
            batch_loss = 0.0
            batch_accuracy = 0.0
            processed_items = 0
            
            for i, (chunk_data, text_tokens) in enumerate(zip(chunk_data_batch, text_tokens_batch)):
                # Extract single sample results
                sample_output = {
                    'logits': output['logits'][i:i+1] if output['logits'] is not None else None,
                    'predicted_durations': output['predicted_durations'][i:i+1],
                    'text_features': output['text_features'][i:i+1]
                }
                
                # Compute loss for this sample
                loss_dict = compute_combined_loss(sample_output, chunk_data, text_tokens, self.device)
                sample_loss = loss_dict.get('total_loss')
                sample_accuracy = loss_dict.get('accuracy', 0.0)
                
                if sample_loss is not None and not torch.isnan(sample_loss):
                    batch_loss += sample_loss
                    batch_accuracy += sample_accuracy
                    processed_items += 1
            
            if processed_items == 0:
                return None, None
            
            avg_batch_loss = batch_loss / processed_items
            avg_batch_accuracy = batch_accuracy / processed_items
            
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            return {
                'total_loss': avg_batch_loss,
                'total_loss_value': avg_batch_loss.item(),
                'accuracy': avg_batch_accuracy
            }, None
            
        except Exception as e:
            return None, None
    
    def _calculate_efficiency_score(self, accuracy, step_time, param_count):
        """Calculate efficiency score: accuracy² / (time × params^0.5)"""
        if step_time <= 0 or param_count <= 0:
            return 0.0
        
        speed = 1000 / step_time  # steps/second
        params_m = param_count / 1000000  # millions
        
        # Balanced efficiency formula
        efficiency = (accuracy ** 2) / (step_time * (params_m ** 0.5))
        return efficiency
    
    def _analyze_phase_1_results(self, results):
        """Analyze Phase 1 results and determine best region"""
        logger.info("\n📊 PHASE 1 ANALYSIS:")
        logger.info("-" * 30)
        
        best_config = None
        best_score = 0.0
        
        for config_name, result in results.items():
            if not result.get('success', False):
                continue
                
            accuracy = result['best_accuracy']
            efficiency = result['efficiency_score']
            embed_dim = result['config']['embed_dim']
            
            logger.info(f"{config_name}: Acc={accuracy:.3f}, Eff={efficiency:.3f}, "
                      f"Params={result['embed_params']['total_embedding']:,}")
            
            # Primary: accuracy, Secondary: efficiency
            score = accuracy * 0.7 + efficiency * 0.3
            
            if score > best_score:
                best_score = score
                best_config = config_name
        
        logger.info(f"\n🏆 Phase 1 Winner: {best_config}")
        
        # Determine best region based on winner
        if best_config == "E256_H768":
            return "low"
        elif best_config == "E512_H768":
            return "mid"
        elif best_config == "E768_H768":
            return "high"
        else:
            return "mid"  # Default fallback
    
    def _generate_phase_2_configs(self, best_region):
        """Generate Phase 2 configs based on best region"""
        if best_region == "low":
            # Refine around E256
            return [
                EmbedDimOptimizationConfig(name="E192_H768", embed_dim=192),
                EmbedDimOptimizationConfig(name="E256_H768", embed_dim=256),
                EmbedDimOptimizationConfig(name="E320_H768", embed_dim=320),
            ]
        elif best_region == "mid":
            # Refine around E512
            return [
                EmbedDimOptimizationConfig(name="E384_H768", embed_dim=384),  # Current baseline
                EmbedDimOptimizationConfig(name="E512_H768", embed_dim=512),
                EmbedDimOptimizationConfig(name="E640_H768", embed_dim=640),
            ]
        elif best_region == "high":
            # Refine around E768
            return [
                EmbedDimOptimizationConfig(name="E640_H768", embed_dim=640),
                EmbedDimOptimizationConfig(name="E768_H768", embed_dim=768),
                EmbedDimOptimizationConfig(name="E896_H768", embed_dim=896),
            ]
    
    def _analyze_final_results(self, all_results):
        """Comprehensive analysis of all results"""
        logger.info("\n🏆 FINAL GREEDY SEARCH RESULTS")
        logger.info("=" * 50)
        
        # Filter successful results
        successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
        
        if not successful_results:
            logger.error("❌ No successful experiments!")
            return None
        
        # Sort by efficiency score
        sorted_results = sorted(
            successful_results.items(),
            key=lambda x: x[1]['efficiency_score'],
            reverse=True
        )
        
        logger.info("📊 EFFICIENCY RANKING:")
        logger.info("-" * 40)
        for i, (config_name, result) in enumerate(sorted_results[:5]):
            embed_dim = result['config']['embed_dim']
            accuracy = result['best_accuracy']
            efficiency = result['efficiency_score']
            step_time = result['avg_step_time'] * 1000
            embed_params = result['embed_params']['total_embedding']
            ratio = result['embed_params']['embed_ratio']
            
            logger.info(f"{i+1}. {config_name:12s}: Acc={accuracy:.3f}, Eff={efficiency:.3f}, "
                      f"Speed={step_time:.1f}ms, Ratio={ratio:.2f}, EmbedParams={embed_params:,}")
        
        # Find best accuracy
        best_accuracy_config = max(successful_results.items(), key=lambda x: x[1]['best_accuracy'])
        
        # Find best efficiency
        best_efficiency_config = max(successful_results.items(), key=lambda x: x[1]['efficiency_score'])
        
        # Find best speed
        best_speed_config = min(successful_results.items(), key=lambda x: x[1]['avg_step_time'])
        
        logger.info("\n🎯 CATEGORY WINNERS:")
        logger.info("-" * 30)
        logger.info(f"🏆 Best Accuracy:  {best_accuracy_config[0]} ({best_accuracy_config[1]['best_accuracy']:.3f})")
        logger.info(f"⚡ Best Efficiency: {best_efficiency_config[0]} ({best_efficiency_config[1]['efficiency_score']:.3f})")
        logger.info(f"🚄 Best Speed:     {best_speed_config[0]} ({best_speed_config[1]['avg_step_time']*1000:.1f}ms)")
        
        # Overall recommendation
        final_winner = best_efficiency_config[0]
        logger.info(f"\n🏅 FINAL RECOMMENDATION: {final_winner}")
        
        winner_result = best_efficiency_config[1]
        logger.info(f"   📊 Embed dimension: {winner_result['config']['embed_dim']}")
        logger.info(f"   🎯 Best accuracy: {winner_result['best_accuracy']:.3f}")
        logger.info(f"   ⚡ Efficiency score: {winner_result['efficiency_score']:.3f}")
        logger.info(f"   🚄 Speed: {winner_result['avg_step_time']*1000:.1f}ms/step")
        logger.info(f"   📈 Embed ratio: {winner_result['embed_params']['embed_ratio']:.2f}")
        
        # Parameter analysis
        logger.info(f"\n📊 PARAMETER BREAKDOWN:")
        logger.info("-" * 30)
        embed_breakdown = winner_result['embed_params']
        total_embed = embed_breakdown['total_embedding']
        total_model = winner_result['param_count']
        
        logger.info(f"   Text embeddings: {embed_breakdown['text_embed']:,}")
        logger.info(f"   Audio embeddings: {embed_breakdown['audio_embed']:,}")
        logger.info(f"   Positional embeddings: {embed_breakdown['pos_embed']:,}")
        logger.info(f"   Projection layers: {embed_breakdown['text_proj'] + embed_breakdown['audio_proj'] + embed_breakdown['pos_proj']:,}")
        logger.info(f"   Total embedding: {total_embed:,} ({total_embed/total_model*100:.1f}%)")
        logger.info(f"   Total model: {total_model:,}")
        
        return {
            'winner': final_winner,
            'winner_result': winner_result,
            'best_accuracy': best_accuracy_config,
            'best_efficiency': best_efficiency_config,
            'best_speed': best_speed_config,
            'ranking': sorted_results
        }
    
    def save_results(self, results, filename=None):
        """Save complete greedy search results"""
        if filename is None:
            timestamp = int(time.time())
            filename = f'embed_dim_greedy_search_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved as: {filename}")
        return filename


def create_greedy_search_plots(results):
    """Create visualization plots for greedy search results"""
    try:
        import matplotlib.pyplot as plt
        
        all_results = results.get('all_results', {})
        successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
        
        if len(successful_results) < 2:
            logger.warning("Not enough results for plotting")
            return
        
        # Extract data for plotting
        embed_dims = []
        accuracies = []
        efficiencies = []
        speeds = []
        embed_params = []
        
        for config_name, result in successful_results.items():
            embed_dims.append(result['config']['embed_dim'])
            accuracies.append(result['best_accuracy'])
            efficiencies.append(result['efficiency_score'])
            speeds.append(result['avg_step_time'] * 1000)  # Convert to ms
            embed_params.append(result['embed_params']['total_embedding'] / 1000000)  # Convert to millions
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Embed Dimension Greedy Search Results', fontsize=16)
        
        # Accuracy vs Embed Dim
        axes[0,0].scatter(embed_dims, accuracies, c='blue', s=60, alpha=0.7)
        axes[0,0].plot(embed_dims, accuracies, 'b--', alpha=0.5)
        axes[0,0].set_xlabel('Embed Dimension')
        axes[0,0].set_ylabel('Best Accuracy')
        axes[0,0].set_title('Accuracy vs Embed Dimension')
        axes[0,0].grid(True, alpha=0.3)
        
        # Efficiency vs Embed Dim
        axes[0,1].scatter(embed_dims, efficiencies, c='green', s=60, alpha=0.7)
        axes[0,1].plot(embed_dims, efficiencies, 'g--', alpha=0.5)
        axes[0,1].set_xlabel('Embed Dimension')
        axes[0,1].set_ylabel('Efficiency Score')
        axes[0,1].set_title('Efficiency vs Embed Dimension')
        axes[0,1].grid(True, alpha=0.3)
        
        # Speed vs Embed Dim
        axes[1,0].scatter(embed_dims, speeds, c='red', s=60, alpha=0.7)
        axes[1,0].plot(embed_dims, speeds, 'r--', alpha=0.5)
        axes[1,0].set_xlabel('Embed Dimension')
        axes[1,0].set_ylabel('Training Speed (ms/step)')
        axes[1,0].set_title('Speed vs Embed Dimension')
        axes[1,0].grid(True, alpha=0.3)
        
        # Embedding Parameters vs Embed Dim
        axes[1,1].scatter(embed_dims, embed_params, c='purple', s=60, alpha=0.7)
        axes[1,1].plot(embed_dims, embed_params, 'm--', alpha=0.5)
        axes[1,1].set_xlabel('Embed Dimension')
        axes[1,1].set_ylabel('Embedding Parameters (Millions)')
        axes[1,1].set_title('Embedding Parameters vs Embed Dimension')
        axes[1,1].grid(True, alpha=0.3)
        
        # Annotate winner
        if 'final_winner' in results:
            winner_config = results['final_winner']['winner']
            winner_result = results['final_winner']['winner_result']
            winner_embed_dim = winner_result['config']['embed_dim']
            
            # Highlight winner in all plots
            for ax in axes.flat:
                ax.axvline(x=winner_embed_dim, color='gold', linestyle=':', linewidth=2, alpha=0.8, label=f'Winner: {winner_config}')
                ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = int(time.time())
        plot_filename = f'embed_dim_greedy_search_plots_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Greedy search plots saved as: {plot_filename}")
        
    except ImportError:
        logger.warning("matplotlib not available - skipping plots")
    except Exception as e:
        logger.warning(f"Plot creation failed: {e}")


class ProperStatefulDataLoader:
    """
    Proper DataLoader for stateful vs stateless comparison
    Each sample (folder) has its own state that persists across chunks
    """
    def __init__(self, data_dir="no_overlap_data", device='cpu', max_samples=4):
        self.data_dir = Path(data_dir)
        self.device = device
        self.samples = {}  # sample_id -> list of chunks (in order)
        self.max_samples = max_samples
        self.current_chunk_idx = 0
        self.max_chunks_per_sample = 0
        
        logger.info(f"🔍 Loading proper stateful data from {data_dir}")
        self._load_samples()
        
    def _load_samples(self):
        """Load samples (each folder = one sample with sequential chunks)"""
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
                
                # Load ALL chunks in order for this sample
                chunk_files = sorted(batch_meta.get('chunk_files', []))
                
                for chunk_file in chunk_files:
                    chunk_path = sample_dir / chunk_file
                    if chunk_path.exists():
                        try:
                            chunk_data = torch.load(chunk_path, map_location=self.device, weights_only=False)
                            
                            if chunk_data.get('clean_chunk', False):
                                # Prepare audio codes
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
                                    
                        except Exception as e:
                            continue
                
                if sample_chunks:
                    self.samples[sample_id] = sample_chunks
                    logger.info(f"   Sample {sample_id}: {len(sample_chunks)} chunks")
                        
            except Exception as e:
                continue
        
        # Calculate max chunks per sample
        if self.samples:
            self.max_chunks_per_sample = min(len(chunks) for chunks in self.samples.values())
            logger.info(f"📊 Loaded {len(self.samples)} samples, {self.max_chunks_per_sample} chunks each")
        else:
            logger.error("❌ No samples loaded!")
    
    def get_batch_at_chunk_index(self, chunk_idx):
        """
        Get batch of chunks at specific index from all samples
        Returns: batch_data, is_valid
        """
        if chunk_idx >= self.max_chunks_per_sample:
            return None, False
            
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
    
    def get_next_batch(self):
        """Get next batch and advance chunk index"""
        batch_data, is_valid = self.get_batch_at_chunk_index(self.current_chunk_idx)
        
        if is_valid:
            self.current_chunk_idx += 1
            
        return batch_data, is_valid
    
    def reset_iterator(self):
        """Reset chunk index to beginning"""
        self.current_chunk_idx = 0
        
    def get_random_batch(self):
        """Get random batch (for stateless baseline)"""
        if not self.samples:
            return None, False
            
        chunk_idx = np.random.randint(0, self.max_chunks_per_sample)
        return self.get_batch_at_chunk_index(chunk_idx)
    
    def get_total_chunks(self):
        """Get total number of chunk indices available"""
        return self.max_chunks_per_sample
    
    def get_num_samples(self):
        """Get number of samples (folders)"""
        return len(self.samples)


def main():
    """Main function to run embed_dim greedy search"""
    logger.info("🔍 EMBED DIMENSION GREEDY SEARCH - UNIFIED ARCHITECTURE")
    logger.info("=" * 80)
    logger.info("Research Question: What is the optimal embed_dim for Pure Mamba L4_H768_E1.5?")
    logger.info("Strategy: Greedy search with unified embedding architecture")
    logger.info("Phase 1: Wide range (E256, E512, E768)")
    logger.info("Phase 2: Refined search around winner")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check data
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        logger.error("❌ no_overlap_data directory not found!")
        logger.error("   Please run batch_no_overlap_preprocessor.py first")
        return
    
    try:
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"📝 Vocabulary size: {vocab_size}")
        
        # Load data using existing stable loader
        data_loader = ProperStatefulDataLoader("no_overlap_data", device, max_samples=4)
        if data_loader.get_num_samples() == 0:
            logger.error("❌ No samples loaded!")
            return
        
        total_chunks = data_loader.get_total_chunks()
        num_samples = data_loader.get_num_samples()
        logger.info(f"📊 Loaded {num_samples} samples with {total_chunks} chunks each")
        
        # Create greedy search experiment
        greedy_search = EmbedDimGreedySearch(tokenizer, data_loader, device)
        
        # Run complete greedy search
        logger.info("\n🔍 Starting embed_dim greedy search...")
        logger.info("   Steps per config: 1000")
        logger.info("   Total configs: ~6 (3 Phase1 + 3 Phase2)")
        logger.info("   Expected runtime: ~2-3 hours")
        logger.info("   Methodology: Stateless processing for speed")
        
        results = greedy_search.run_greedy_search(steps_per_config=1000)
        
        # Save results
        results_file = greedy_search.save_results(results)
        
        # Create visualization
        create_greedy_search_plots(results)
        
        logger.info("\n✅ EMBED_DIM GREEDY SEARCH COMPLETED!")
        logger.info("   Results analyzed and saved")
        logger.info("   Optimal embed_dim configuration identified")
        logger.info("   Check plots for visual analysis")
        logger.info(f"   Results file: {results_file}")
        
        # Final summary
        if 'final_winner' in results and results['final_winner']:
            winner = results['final_winner']['winner']
            winner_result = results['final_winner']['winner_result']
            logger.info(f"\n🏅 OPTIMAL CONFIGURATION: {winner}")
            logger.info(f"   Embed dimension: {winner_result['config']['embed_dim']}")
            logger.info(f"   Accuracy: {winner_result['best_accuracy']:.3f}")
            logger.info(f"   Efficiency score: {winner_result['efficiency_score']:.3f}")
            logger.info(f"   Embed ratio: {winner_result['embed_params']['embed_ratio']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Greedy search failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()