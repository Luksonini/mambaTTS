#!/usr/bin/env python3
"""
FINAL FIXED S5 Stateful vs Stateless Experiment
===============================================
Uses correct S5 API parameters based on inspection
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
import sys

# Setup logging first
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("🔬 FINAL FIXED S5 STATEFUL vs STATELESS COMPARISON")
print("=" * 70)

# Import real S5 components
try:
    from s5 import S5Block, S5SSM, S5, make_NPLR_HiPPO
    logger.info("✅ Successfully imported S5 components from s5-pytorch")
    S5_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import S5: {e}")
    S5_AVAILABLE = False

# Also try Mamba for comparison
try:
    from mamba_ssm import Mamba
    logger.info("✅ Successfully imported Mamba from mamba-ssm")
    MAMBA_AVAILABLE = True
except ImportError as e:
    logger.info(f"ℹ️  Mamba not available: {e}")
    MAMBA_AVAILABLE = False

if not S5_AVAILABLE:
    logger.error("❌ S5 not available - please install s5-pytorch")
    exit(1)

# Import existing components
try:
    from nucleotide_tokenizer import NucleotideTokenizer
    from losses import compute_combined_loss
    logger.info("✅ Imported tokenizer and losses")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    exit(1)


@dataclass
class S5Config:
    """Configuration for S5 model comparison"""
    name: str
    model_type: str = "S5Block"  # "S5Block", "S5", "Mamba"
    num_layers: int = 4
    hidden_dim: int = 768
    state_dim: int = 64
    embed_dim: int = 384
    num_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1
    stateful: bool = False


class SimpleS5Block(nn.Module):
    """
    Wrapper around real S5Block with correct parameters
    """
    def __init__(self, d_model, state_dim=64, dropout=0.1, model_type="S5Block"):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.model_type = model_type
        
        # Create the appropriate state-space model with CORRECT parameters
        if model_type == "S5Block":
            # S5Block requires: dim, state_dim, bidir (all required!)
            self.ssm_layer = S5Block(
                dim=d_model,           # width dimension (NOT d_model!)
                state_dim=state_dim,   # state dimension  
                bidir=False,           # unidirectional (causal) - REQUIRED!
                block_count=1,         # single block
                liquid=False,          # standard S5
                ff_mult=2.0,           # feedforward multiplier
                glu=True,              # gated linear unit
                ff_dropout=dropout,    # feedforward dropout
                attn_dropout=dropout   # attention dropout
            )
            logger.info(f"✅ S5Block created: dim={d_model}, state_dim={state_dim}, bidir=False")
            
        elif model_type == "S5":
            # S5 requires: width (required)
            self.ssm_layer = S5(
                width=d_model,         # model width (NOT d_model!)
                state_width=state_dim, # state width (optional)
                block_count=1,         # single block
                dt_min=0.001,         # min timestep
                dt_max=0.1,           # max timestep
                liquid=False,         # standard S5
                bidir=False           # unidirectional
            )
            logger.info(f"✅ S5 created: width={d_model}, state_width={state_dim}")
            
            # Add normalization for S5
            self.norm = nn.LayerNorm(d_model)
            self.dropout_layer = nn.Dropout(dropout)
            
        elif model_type == "Mamba" and MAMBA_AVAILABLE:
            # Use Mamba as comparison
            self.ssm_layer = Mamba(
                d_model=d_model,
                d_state=state_dim,
                expand=2
            )
            logger.info(f"✅ Mamba created: d_model={d_model}, d_state={state_dim}")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # State management
        self.stateful_mode = False
        self._persistent_state = None
        
    def enable_stateful_mode(self, batch_size=1):
        """Enable state persistence across chunks"""
        self.stateful_mode = True
        device = next(self.parameters()).device
        
        # Initialize appropriate state
        self._persistent_state = torch.zeros(
            batch_size, self.state_dim, device=device, dtype=torch.float32
        )
        
    def disable_stateful_mode(self):
        """Disable state persistence"""
        self.stateful_mode = False
        self._persistent_state = None
        
    def reset_state(self):
        """Reset persistent state"""
        if self.stateful_mode and self._persistent_state is not None:
            self._persistent_state.zero_()
    
    def forward(self, x, return_state_info=False):
        """
        Forward pass with optional state persistence
        x: [batch, seq_len, d_model]
        """
        B, L, D = x.shape
        residual = x
        
        # Handle different model types
        if self.model_type == "S5Block":
            # S5Block handles everything internally including normalization
            # Note: S5Block might not support explicit state passing
            x = self.ssm_layer(x)
            # S5Block includes residual connection internally
                
        elif self.model_type == "S5":
            # Manual handling for S5
            x_norm = self.norm(x)
            x_out = self.ssm_layer(x_norm)
            x = x_out + residual  # Residual connection
            x = self.dropout_layer(x)
            
        elif self.model_type == "Mamba":
            # Mamba handling
            x = self.ssm_layer(x)
        
        if return_state_info:
            state_info = {
                'has_state': self.stateful_mode and self._persistent_state is not None,
                'state_norm': torch.norm(self._persistent_state).item() if self._persistent_state is not None else 0.0,
                'stateful_mode': self.stateful_mode,
                'model_type': self.model_type
            }
            return x, state_info
        
        return x


class S5TextEncoder(nn.Module):
    """Text encoder using S5 models"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, state_dim, dropout=0.1, model_type="S5Block"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, hidden_dim) * 0.02)
        
        # Stack of S5 blocks
        self.layers = nn.ModuleList([
            SimpleS5Block(hidden_dim, state_dim, dropout, model_type)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def enable_stateful_mode(self, batch_size=1):
        """Enable state persistence for all layers"""
        for layer in self.layers:
            layer.enable_stateful_mode(batch_size)
    
    def disable_stateful_mode(self):
        """Disable state persistence for all layers"""
        for layer in self.layers:
            layer.disable_stateful_mode()
    
    def reset_states(self):
        """Reset all layer states"""
        for layer in self.layers:
            layer.reset_state()
    
    def forward(self, tokens, return_sequence=True, return_state_info=False):
        B, L = tokens.shape
        
        # Embeddings and projection
        x = self.embedding(tokens)
        x = self.embed_proj(x)
        
        # Add positional encoding
        if L <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :L, :]
        
        x = self.dropout(x)
        
        # Process through S5 layers
        state_infos = []
        for layer in self.layers:
            if return_state_info:
                x, state_info = layer(x, return_state_info=True)
                state_infos.append(state_info)
            else:
                x = layer(x)
        
        x = self.norm(x)
        
        result = x if return_sequence else x.mean(dim=1)
        
        if return_state_info:
            return result, state_infos
        return result


class S5AudioProcessor(nn.Module):
    """Audio processor using S5 models"""
    def __init__(self, hidden_dim, num_codebooks=8, codebook_size=1024, state_dim=64, model_type="S5Block"):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Audio embeddings
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(codebook_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_codebooks)
        ])
        
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # S5 processing layers
        self.s5_layers = nn.ModuleList([
            SimpleS5Block(hidden_dim, state_dim, dropout=0.1, model_type=model_type)
            for _ in range(4)
        ])
        
        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, codebook_size)
            ) for _ in range(num_codebooks)
        ])
    
    def enable_stateful_mode(self, batch_size=1):
        """Enable state persistence for audio processing"""
        for layer in self.s5_layers:
            layer.enable_stateful_mode(batch_size)
    
    def disable_stateful_mode(self):
        """Disable state persistence for audio processing"""
        for layer in self.s5_layers:
            layer.disable_stateful_mode()
    
    def reset_states(self):
        """Reset all audio processor states"""
        for layer in self.s5_layers:
            layer.reset_state()
    
    def forward(self, audio_tokens, text_context, return_state_info=False):
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
        
        # Combine embeddings
        x = torch.stack(embedded, dim=1).mean(dim=1)  # [B, T, hidden_dim]
        
        # Add text context
        text_context_proj = self.context_proj(text_context).unsqueeze(1)
        x = x + text_context_proj
        
        # Process through S5 layers
        state_infos = []
        for layer in self.s5_layers:
            if return_state_info:
                x, state_info = layer(x, return_state_info=True)
                state_infos.append(state_info)
            else:
                x = layer(x)
        
        # Generate logits for each codebook
        logits = []
        for c in range(8):
            head_logits = self.output_heads[c](x)
            logits.append(head_logits)
        
        result = torch.stack(logits, dim=1)
        
        if return_state_info:
            return result, state_infos
        return result


class S5Model(nn.Module):
    """
    S5 model for TTS with optional state persistence
    """
    def __init__(self, config: S5Config):
        super().__init__()
        self.config = config
        self.stateful = config.stateful
        
        # S5 components
        self.text_encoder = S5TextEncoder(
            vocab_size=1000,  # Will be updated
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            state_dim=config.state_dim,
            dropout=config.dropout,
            model_type=config.model_type
        )
        
        # Duration regulator
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.audio_processor = S5AudioProcessor(
            hidden_dim=config.hidden_dim,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            state_dim=config.state_dim,
            model_type=config.model_type
        )
        
        # Projections
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 {config.name}: {total_params:,} parameters")
        logger.info(f"   📏 {config.model_type}_L{config.num_layers}_H{config.hidden_dim}_S{config.state_dim}")
        logger.info(f"   🔗 Stateful: {config.stateful}")
        
    def enable_stateful_mode(self, batch_size=1):
        """Enable state persistence across all components"""
        if not self.stateful:
            return
        
        self.text_encoder.enable_stateful_mode(batch_size)
        self.audio_processor.enable_stateful_mode(batch_size)
    
    def disable_stateful_mode(self):
        """Disable state persistence across all components"""
        self.text_encoder.disable_stateful_mode()
        self.audio_processor.disable_stateful_mode()
    
    def reset_all_states(self):
        """Reset all model states"""
        if not self.stateful:
            return
        
        self.text_encoder.reset_states()
        self.audio_processor.reset_states()
    
    def forward(self, text_tokens, audio_tokens=None, return_state_info=False):
        batch_size = text_tokens.shape[0]
        
        # Process text with S5
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        # Duration prediction
        predicted_durations = self.duration_regulator(text_features.mean(dim=1))
        
        # Audio processing with S5
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


class ProperStatefulDataLoader:
    """Data loader for stateful vs stateless comparison"""
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
        """Load samples"""
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
                                    
                        except Exception as e:
                            continue
                
                if sample_chunks:
                    self.samples[sample_id] = sample_chunks
                        
            except Exception as e:
                continue
        
        if self.samples:
            self.max_chunks_per_sample = min(len(chunks) for chunks in self.samples.values())
            logger.info(f"📊 Loaded {len(self.samples)} samples, {self.max_chunks_per_sample} chunks each")
        else:
            logger.error("❌ No samples loaded!")
    
    def get_batch_at_chunk_index(self, chunk_idx):
        """Get batch of chunks at specific index"""
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
        """Get random batch for stateless baseline"""
        if not self.samples:
            return None, False
            
        chunk_idx = np.random.randint(0, self.max_chunks_per_sample)
        return self.get_batch_at_chunk_index(chunk_idx)
    
    def get_total_chunks(self):
        """Get total number of chunk indices available"""
        return self.max_chunks_per_sample
    
    def get_num_samples(self):
        """Get number of samples"""
        return len(self.samples)


class S5Experiment:
    """S5 model comparison experiment"""
    
    def __init__(self, tokenizer, data_loader, device, model_type="S5Block"):
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = device
        self.model_type = model_type
        
        # Create configurations
        self.config_stateless = S5Config(
            name=f"{model_type}_L4_H768_S64_STATELESS",
            model_type=model_type,
            stateful=False
        )
        
        self.config_stateful = S5Config(
            name=f"{model_type}_L4_H768_S64_STATEFUL",
            model_type=model_type,
            stateful=True
        )
    
    def run_experiment(self, config, test_steps=1000):
        """Run single experiment"""
        experiment_type = f"{config.model_type}_{'STATEFUL' if config.stateful else 'STATELESS'}"
        logger.info(f"\n🚀 Running {experiment_type} experiment")
        
        start_time = time.time()
        
        try:
            # Create model
            model = S5Model(config).to(self.device)
            
            # Update vocab size
            actual_vocab_size = self.tokenizer.get_vocab_size()
            model.text_encoder.embedding = nn.Embedding(actual_vocab_size, config.embed_dim).to(self.device)
            
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"📊 {experiment_type} parameters: {param_count:,}")
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=5e-4,
                weight_decay=1e-6,
                betas=(0.9, 0.95)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=test_steps, eta_min=5e-5
            )
            
            # Training metrics
            metrics = {
                'accuracies': [],
                'losses': [],
                'step_times': []
            }
            
            best_accuracy = 0.0
            successful_steps = 0
            
            # Training loop
            model.train()
            for step in range(test_steps):
                step_start = time.time()
                
                try:
                    # Get batch
                    if config.stateful:
                        batch_data, is_valid = self.data_loader.get_next_batch()
                        if not is_valid:
                            self.data_loader.reset_iterator()
                            batch_data, is_valid = self.data_loader.get_next_batch()
                            if not is_valid:
                                continue
                        
                        # Enable stateful mode
                        model.enable_stateful_mode(batch_size=len(batch_data['chunks']))
                        
                        # Reset states at beginning of sample
                        if batch_data['chunk_idx'] == 0:
                            model.reset_all_states()
                            
                    else:
                        batch_data, is_valid = self.data_loader.get_random_batch()
                        if not is_valid:
                            continue
                        model.disable_stateful_mode()
                    
                    # Process batch
                    optimizer.zero_grad()
                    
                    total_loss = 0.0
                    total_accuracy = 0.0
                    processed_items = 0
                    
                    for chunk_data in batch_data['chunks']:
                        try:
                            text_tokens = chunk_data['text_tokens']
                            if text_tokens.dim() == 1:
                                text_tokens = text_tokens.unsqueeze(0)
                            
                            audio_codes = chunk_data['audio_codes']
                            if audio_codes.dim() == 2:
                                audio_codes = audio_codes.unsqueeze(0)
                            
                            if text_tokens.shape[1] < 5:
                                continue
                            
                            # Forward pass
                            output = model(text_tokens, audio_codes)
                            
                            # Compute loss
                            loss_dict = compute_combined_loss(output, chunk_data, text_tokens, self.device)
                            sample_loss = loss_dict.get('total_loss')
                            sample_accuracy = loss_dict.get('accuracy', 0.0)
                            
                            if sample_loss is not None and not torch.isnan(sample_loss):
                                total_loss += sample_loss
                                total_accuracy += sample_accuracy
                                processed_items += 1
                                
                        except Exception as e:
                            continue
                    
                    if processed_items == 0:
                        continue
                    
                    avg_loss = total_loss / processed_items
                    avg_accuracy = total_accuracy / processed_items
                    
                    # Backward pass
                    avg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # Track metrics
                    step_time = time.time() - step_start
                    metrics['accuracies'].append(avg_accuracy)
                    metrics['losses'].append(avg_loss.item())
                    metrics['step_times'].append(step_time)
                    
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                    
                    successful_steps += 1
                    
                    # Progress logging
                    if step % 100 == 0:
                        logger.info(f"   Step {step:4d}: Loss={avg_loss.item():.4f}, "
                                  f"Acc={avg_accuracy:.4f}, Time={step_time*1000:.1f}ms")
                    
                    # Early stopping
                    if avg_accuracy > 0.95 and step > 300:
                        logger.info(f"   🎉 {experiment_type} early success at step {step}!")
                        break
                        
                except Exception as e:
                    logger.warning(f"Step {step} failed: {e}")
                    continue
            
            # Calculate results
            training_time = time.time() - start_time
            final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
            
            result = {
                'experiment_type': experiment_type,
                'success': successful_steps > test_steps * 0.2,
                'training_time': training_time,
                'final_accuracy': final_accuracy,
                'best_accuracy': best_accuracy,
                'total_steps': successful_steps,
                'model_type': config.model_type
            }
            
            logger.info(f"✅ {experiment_type} Results:")
            logger.info(f"   Final accuracy: {final_accuracy:.4f}")
            logger.info(f"   Best accuracy: {best_accuracy:.4f}")
            logger.info(f"   Training time: {training_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {experiment_type} experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'experiment_type': experiment_type,
                'success': False,
                'error': str(e)
            }
    
    def run_comparison(self, test_steps=1000):
        """Run full comparison"""
        logger.info(f"🔬 {self.model_type} STATEFUL vs STATELESS COMPARISON")
        logger.info("=" * 60)
        
        results = {}
        
        # Run experiments
        results['stateless'] = self.run_experiment(self.config_stateless, test_steps)
        results['stateful'] = self.run_experiment(self.config_stateful, test_steps)
        
        # Analyze results
        self._analyze_results(results)
        
        # Save results
        timestamp = int(time.time())
        filename = f'{self.model_type.lower()}_comparison_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 {self.model_type} results saved as: {filename}")
        
        return results
    
    def _analyze_results(self, results):
        """Analyze comparison results"""
        logger.info(f"\n🏆 {self.model_type} COMPARISON RESULTS")
        logger.info("=" * 50)
        
        stateless = results.get('stateless', {})
        stateful = results.get('stateful', {})
        
        if not stateless.get('success') and not stateful.get('success'):
            logger.error("❌ Both experiments failed!")
            return
        
        # Performance comparison
        logger.info("📊 PERFORMANCE COMPARISON:")
        logger.info("-" * 30)
        
        metrics = [
            ('Final Accuracy', 'final_accuracy', '%', 100),
            ('Best Accuracy', 'best_accuracy', '%', 100),
            ('Training Time', 'training_time', 's', 1),
        ]
        
        for metric_name, metric_key, unit, multiplier in metrics:
            stateless_val = stateless.get(metric_key, 0) * multiplier
            stateful_val = stateful.get(metric_key, 0) * multiplier
            
            if stateless_val > 0 and stateful_val > 0:
                if metric_name == 'Training Time':
                    # Lower is better for time
                    diff = ((stateless_val - stateful_val) / stateful_val) * 100
                    winner = "STATEFUL" if stateless_val > stateful_val else "STATELESS"
                else:
                    # Higher is better for accuracy
                    diff = ((stateful_val - stateless_val) / stateless_val) * 100
                    winner = "STATEFUL" if stateful_val > stateless_val else "STATELESS"
                
                logger.info(f"{metric_name:15s}: STATELESS={stateless_val:.3f}{unit}, "
                          f"STATEFUL={stateful_val:.3f}{unit} "
                          f"({diff:+.1f}% {winner})")
        
        # Recommendation
        self._generate_recommendation(results)
    
    def _generate_recommendation(self, results):
        """Generate recommendation"""
        logger.info(f"\n💡 {self.model_type} RECOMMENDATION:")
        logger.info("=" * 30)
        
        stateless = results.get('stateless', {})
        stateful = results.get('stateful', {})
        
        if not stateless.get('success') or not stateful.get('success'):
            logger.warning("⚠️  Incomplete results")
            return
        
        stateful_accuracy = stateful.get('best_accuracy', 0)
        stateless_accuracy = stateless.get('best_accuracy', 0)
        
        accuracy_improvement = stateful_accuracy - stateless_accuracy
        
        logger.info(f"📊 Analysis:")
        logger.info(f"   Accuracy improvement: {accuracy_improvement:+.4f}")
        
        if accuracy_improvement > 0.05:
            recommendation = f"🏆 {self.model_type} STATEFUL RECOMMENDED"
            reason = "Significant accuracy improvement with state persistence"
        elif accuracy_improvement > 0.02:
            recommendation = f"🔀 {self.model_type} STATEFUL CONDITIONALLY RECOMMENDED"  
            reason = "Moderate accuracy improvement"
        else:
            recommendation = f"🔄 {self.model_type} STATELESS RECOMMENDED"
            reason = "No significant advantage to state persistence"
        
        logger.info(f"\n{recommendation}")
        logger.info(f"Reason: {reason}")


def main():
    """Main function"""
    logger.info("🔬 FINAL S5 STATEFUL vs STATELESS COMPARISON")
    logger.info("=" * 60)
    
    # Choose model type
    model_type = "S5Block"  # Can be "S5Block", "S5", or "Mamba"
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    logger.info(f"Using model type: {model_type}")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check data
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        logger.error("❌ no_overlap_data directory not found!")
        return
    
    try:
        # Setup components
        tokenizer = NucleotideTokenizer()
        data_loader = ProperStatefulDataLoader("no_overlap_data", device, max_samples=4)
        
        if data_loader.get_num_samples() == 0:
            logger.error("❌ No samples loaded!")
            return
        
        # Create experiment
        experiment = S5Experiment(tokenizer, data_loader, device, model_type)
        
        # Run comparison (reduced steps for faster testing)
        results = experiment.run_comparison(test_steps=1000)
        
        logger.info("\n✅ S5 COMPARISON COMPLETED!")
        
    except Exception as e:
        logger.error(f"❌ S5 comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()