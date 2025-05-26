#!/usr/bin/env python3
"""
Pure Mamba E768_H768 Refinement Test
====================================
Test Pure Mamba E768_H768 (zwycięzca greedy search) w dwóch wariantach:
1. Single-step generation (baseline)
2. Iterative refinement (3-10 kroków)

Dataset: no_overlap_data (same as greedy search)
Fragments: ~10 sekund każdy
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
class E768H768Config:
    """Optimal config z greedy search"""
    name: str = "E768_H768_L4_E1.5"
    embed_dim: int = 768        # Winner z greedy search
    hidden_dim: int = 768       # Perfect ratio
    num_layers: int = 4         # L4 optimal
    expand_factor: float = 1.5  # E1.5 optimal
    num_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1


class UnifiedMambaBlock(nn.Module):
    """Pure Mamba block - E768_H768 optimized"""
    def __init__(self, d_model, expand_factor=1.5, dropout=0.1, reverse=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand_factor)
        self.reverse = reverse
        
        # Core Mamba components (optimal design)
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
        
        # Simplified but effective state space
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


class E768H768TextEncoder(nn.Module):
    """Text encoder z E768_H768 architecture"""
    def __init__(self, vocab_size, config: E768H768Config):
        super().__init__()
        
        # E768_H768: Perfect ratio embedding
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)  # 768
        # No projection needed - perfect ratio!
        
        # Positional encoding (E768)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, config.embed_dim) * 0.02)
        
        # Stack of Pure Mamba blocks
        self.layers = nn.ModuleList([
            UnifiedMambaBlock(config.hidden_dim, config.expand_factor, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Final processing
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, tokens, return_sequence=True):
        B, L = tokens.shape
        
        # E768 embedding (no projection needed!)
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
    """Audio processor z E768_H768 architecture"""
    def __init__(self, config: E768H768Config):
        super().__init__()
        
        # E768 audio embeddings (perfect ratio)
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(config.codebook_size, config.embed_dim),  # E768
                nn.LayerNorm(config.embed_dim),
                nn.Dropout(0.1)
            ) for _ in range(config.num_codebooks)
        ])
        
        self.context_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Pure Mamba processing layers
        self.mamba_layers = nn.ModuleList([
            UnifiedMambaBlock(config.hidden_dim, config.expand_factor, dropout=0.1)
            for _ in range(4)
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


class PureMambaE768H768SingleStep(nn.Module):
    """Single-step Pure Mamba E768_H768 (baseline)"""
    def __init__(self, config: E768H768Config):
        super().__init__()
        self.config = config
        
        self.text_encoder = E768H768TextEncoder(
            vocab_size=1000,  # Will be updated
            config=config
        )
        
        # Duration regulator
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.audio_processor = E768H768AudioProcessor(config)
        
        # Text projections
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Log model info
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 PureMambaE768H768SingleStep: {param_count:,} parameters")
        logger.info(f"   📏 E{config.embed_dim}_H{config.hidden_dim}_L{config.num_layers}_E{config.expand_factor}")
    
    def forward(self, text_tokens, audio_tokens=None):
        # Process text through E768_H768 architecture
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        # Duration prediction
        predicted_durations = self.duration_regulator(text_features.mean(dim=1))
        
        # Audio processing
        if audio_tokens is not None:
            audio_logits = self.audio_processor(audio_tokens, text_context)
        else:
            audio_logits = None
        
        return {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'text_features': text_features,
        }


class PureMambaE768H768TwoStepHierarchical(nn.Module):
    """2-Step Hierarchical Pure Mamba E768_H768 (Feature-level: Coarse → Fine)"""
    def __init__(self, config: E768H768Config):
        super().__init__()
        self.config = config
        
        # Same text encoder as single-step
        self.text_encoder = E768H768TextEncoder(
            vocab_size=1000,
            config=config
        )
        
        # Duration regulator (borrowed from single-step)
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Coarse feature processing (structural audio features)
        self.coarse_layers = nn.ModuleList([
            UnifiedMambaBlock(config.hidden_dim, config.expand_factor, config.dropout)
            for _ in range(2)  # 2 layers for coarse processing
        ])
        
        # Coarse to compressed features
        self.coarse_compression = nn.Linear(config.hidden_dim, config.hidden_dim // 2)  # H768 → H384
        
        # Fine feature processing (detailed audio features, conditioned on coarse)
        self.fine_layers = nn.ModuleList([
            UnifiedMambaBlock(config.hidden_dim, config.expand_factor, config.dropout)
            for _ in range(2)  # 2 layers for fine processing
        ])
        
        # Fine to compressed features  
        self.fine_compression = nn.Linear(config.hidden_dim, config.hidden_dim // 2)  # H768 → H384
        
        # Final combination projection
        self.feature_combination = nn.Linear(config.hidden_dim, config.hidden_dim)  # H768 → H768
        
        # Output heads for all 8 codebooks
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim // 2, config.codebook_size)
            ) for _ in range(config.num_codebooks)
        ])
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"🔄 PureMambaE768H768TwoStepHierarchical (Feature-level): {param_count:,} parameters")
        logger.info(f"🔄 PureMambaE768H768TwoStepHierarchical (Feature-level): {param_count:,} parameters")
    
    def forward(self, text_tokens, target_tokens=None):
        B, L_text = text_tokens.shape
        
        # Step 1: Text encoding
        text_features = self.text_encoder(text_tokens, return_sequence=True)  # [B, L_text, H768]
        
        # Step 2: Duration regulation - expand to audio length
        if target_tokens is not None:
            # Training: match target audio length
            target_length = target_tokens.shape[2]  # Audio timesteps
        else:
            # Inference: predict duration and expand
            predicted_durations = self.duration_regulator(text_features.mean(dim=1))  # [B, 1]
            target_length = max(50, min(200, int(predicted_durations.mean().item() * L_text)))
        
        # Expand text features to audio length using interpolation
        if text_features.shape[1] != target_length:
            # Interpolate text features to match audio length
            text_features_expanded = F.interpolate(
                text_features.transpose(1, 2),  # [B, H768, L_text]
                size=target_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [B, target_length, H768]
        else:
            text_features_expanded = text_features
        
        # Step 3: Coarse feature processing (structural audio features)
        coarse_features = text_features_expanded
        for layer in self.coarse_layers:
            coarse_features = layer(coarse_features)  # [B, target_length, H768]
        
        # Compress coarse features
        coarse_compressed = self.coarse_compression(coarse_features)  # [B, target_length, H384]
        
        # Step 4: Fine feature processing (detailed features, conditioned on coarse)
        # Combine original text features with coarse features for fine processing
        fine_input = text_features_expanded + coarse_features  # [B, target_length, H768]
        
        fine_features = fine_input
        for layer in self.fine_layers:
            fine_features = layer(fine_features)  # [B, target_length, H768]
        
        # Compress fine features
        fine_compressed = self.fine_compression(fine_features)  # [B, target_length, H384]
        
        # Step 5: Combine coarse and fine features
        combined_features = torch.cat([coarse_compressed, fine_compressed], dim=-1)  # [B, target_length, H768]
        combined_features = self.feature_combination(combined_features)  # [B, target_length, H768]
        
        # Step 6: Generate logits for all 8 codebooks
        all_logits = []
        for cb_idx in range(self.config.num_codebooks):
            cb_logits = self.output_heads[cb_idx](combined_features)  # [B, target_length, codebook_size]
            all_logits.append(cb_logits)
        
        # Stack all codebook logits
        output_logits = torch.stack(all_logits, dim=1)  # [B, 8, target_length, codebook_size]
        
        return output_logits


class ProperStatefulDataLoader:
    """DataLoader z greedy search (same dataset)"""
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
        """Load samples (same as greedy search)"""
        print(f"📂 Checking directory: {self.data_dir}")
        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return
            
        sample_dirs = [d for d in self.data_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('clean_batch_')]
        sample_dirs.sort()
        print(f"📁 Found {len(sample_dirs)} sample directories")
        
        for sample_dir in sample_dirs[:self.max_samples]:
            print(f"   Processing: {sample_dir.name}")
            try:
                meta_path = sample_dir / "batch_meta.json"
                if not meta_path.exists():
                    print(f"   ⚠️  No meta file in {sample_dir.name}")
                    continue
                    
                with open(meta_path, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)
                
                sample_id = batch_meta.get('batch_idx', len(self.samples))
                sample_chunks = []
                
                chunk_files = sorted(batch_meta.get('chunk_files', []))
                print(f"   📋 {len(chunk_files)} chunk files to process")
                
                for i, chunk_file in enumerate(chunk_files):
                    if i % 20 == 0:  # Progress every 20 chunks
                        print(f"      Processing chunk {i+1}/{len(chunk_files)}")
                    
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
                            if i % 50 == 0:  # Only print occasional errors
                                print(f"      ⚠️  Error loading chunk {i}: {e}")
                            continue
                
                if sample_chunks:
                    self.samples[sample_id] = sample_chunks
                    print(f"   ✅ Sample {sample_id}: {len(sample_chunks)} chunks loaded")
                    logger.info(f"   Sample {sample_id}: {len(sample_chunks)} chunks")
                        
            except Exception as e:
                print(f"   ❌ Error processing {sample_dir.name}: {e}")
                continue
        
        if self.samples:
            self.max_chunks_per_sample = min(len(chunks) for chunks in self.samples.values())
            print(f"✅ Final result: {len(self.samples)} samples, {self.max_chunks_per_sample} chunks each")
            logger.info(f"📊 Loaded {len(self.samples)} samples, {self.max_chunks_per_sample} chunks each")
        else:
            print("❌ NO SAMPLES LOADED!")
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
        """Get number of loaded samples"""
        return len(self.samples)
    
    def get_total_chunks(self):
        """Get total number of chunk indices available"""
        return self.max_chunks_per_sample


class E768H768RefinementTester:
    """Main tester dla E768_H768 Single vs Refinement"""
    
    def __init__(self):
        print("🔧 INITIALIZING E768H768RefinementTester...")
        self.device = DEVICE
        print(f"   Device: {self.device}")
        
        print("📝 Creating tokenizer...")
        self.tokenizer = NucleotideTokenizer()
        print("✅ Tokenizer created")
        
        print("⚙️  Creating config...")
        self.config = E768H768Config()
        print("✅ Config created")
        
        print("📊 Loading data...")
        # Load data (same as greedy search)
        self.data_loader = ProperStatefulDataLoader("no_overlap_data", self.device, max_samples=4)
        print("✅ Data loader created")
        
        if self.data_loader.get_num_samples() == 0:
            print("❌ No samples loaded!")
            logger.error("❌ No samples loaded!")
            return
        
        print(f"✅ Loaded {self.data_loader.get_num_samples()} samples")
        
        # Update vocab size
        vocab_size = self.tokenizer.get_vocab_size()
        print(f"📝 Vocabulary size: {vocab_size}")
        logger.info(f"📝 Vocabulary size: {vocab_size}")
        
        print("🧠 Creating single-step model...")
        # Create models
        self.single_step_model = PureMambaE768H768SingleStep(self.config).to(self.device)
        print("✅ Single-step model created")
        
        print("🔄 Creating 2-step hierarchical model...")
        self.refinement_model = PureMambaE768H768TwoStepHierarchical(self.config).to(self.device)
        print("✅ 2-step hierarchical model created")
        
        print("🔧 Updating vocab sizes...")
        # Update vocab sizes
        self.single_step_model.text_encoder.embedding = nn.Embedding(vocab_size, self.config.embed_dim).to(self.device)
        self.refinement_model.text_encoder.embedding = nn.Embedding(vocab_size, self.config.embed_dim).to(self.device)
        print("✅ Vocab sizes updated")
        
        logger.info("✅ Models ready for testing")
        print("✅ INITIALIZATION COMPLETE!")
    
    def test_single_step(self, steps=1000):
        """Test Single-step E768_H768"""
        logger.info("🎯 Testing Single-step E768_H768...")
        
        optimizer = torch.optim.AdamW(
            self.single_step_model.parameters(),
            lr=6e-4,
            weight_decay=1e-6,
            betas=(0.9, 0.95)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=6e-5
        )
        
        best_accuracy = 0.0
        metrics = {'accuracies': [], 'losses': [], 'step_times': []}
        
        logger.info("Step  | Loss     | Accuracy | Speed   | Status")
        logger.info("-" * 50)
        
        self.single_step_model.train()
        
        for step in range(steps):
            step_start = time.time()
            
            # Get batch
            batch_data, is_valid = self.data_loader.get_random_batch()
            if not is_valid:
                continue
            
            # Process batch
            loss_dict = self._process_single_step_batch(batch_data, optimizer)
            if loss_dict is None:
                continue
            
            scheduler.step()
            
            # Track metrics
            step_time = time.time() - step_start
            accuracy = loss_dict.get('accuracy', 0.0)
            loss_val = loss_dict.get('total_loss_value', float('inf'))
            
            metrics['accuracies'].append(accuracy)
            metrics['losses'].append(loss_val)
            metrics['step_times'].append(step_time)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            # Progress
            if step % 200 == 0:
                status = "🎉" if accuracy > 0.8 else "🔥" if accuracy > 0.5 else "🎯"
                logger.info(f"{step:5d} | {loss_val:.6f} | {accuracy:.4f} | {step_time*1000:.1f}ms | {status}")
        
        avg_step_time = np.mean(metrics['step_times']) if metrics['step_times'] else 0.0
        final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
        
        logger.info(f"✅ Single-step results:")
        logger.info(f"   Best accuracy: {best_accuracy:.4f}")
        logger.info(f"   Final accuracy: {final_accuracy:.4f}")
        logger.info(f"   Avg step time: {avg_step_time*1000:.1f}ms")
        
        return {
            'best_accuracy': best_accuracy,
            'final_accuracy': final_accuracy,
            'avg_step_time': avg_step_time,
            'metrics': metrics
        }
    
    def test_refinement(self, steps=1000, refinement_steps_range=(3, 8)):
        """Test 2-Step Hierarchical E768_H768"""
        logger.info(f"🔄 Testing 2-Step Hierarchical E768_H768...")
        
        optimizer = torch.optim.AdamW(
            self.refinement_model.parameters(),
            lr=4e-4,  # Slightly lower than single-step
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=4e-5
        )
        
        best_accuracy = 0.0
        metrics = {'accuracies': [], 'losses': [], 'step_times': []}
        
        logger.info("Step  | Loss     | Accuracy | Speed   | Status")
        logger.info("-" * 50)
        
        self.refinement_model.train()
        
        for step in range(steps):
            step_start = time.time()
            
            # Get batch
            batch_data, is_valid = self.data_loader.get_random_batch()
            if not is_valid:
                continue
            
            # Process batch with 2-step hierarchical
            loss_dict = self._process_hierarchical_batch(batch_data, optimizer)
            if loss_dict is None:
                continue
            
            scheduler.step()
            
            # Track metrics
            step_time = time.time() - step_start
            accuracy = loss_dict.get('accuracy', 0.0)
            loss_val = loss_dict.get('total_loss_value', float('inf'))
            
            metrics['accuracies'].append(accuracy)
            metrics['losses'].append(loss_val)
            metrics['step_times'].append(step_time)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            # Progress
            if step % 200 == 0:
                status = "🎉" if accuracy > 0.8 else "🔥" if accuracy > 0.5 else "🔄"
                logger.info(f"{step:5d} | {loss_val:.6f} | {accuracy:.4f} | {step_time*1000:.1f}ms | {status}")
        
        avg_step_time = np.mean(metrics['step_times']) if metrics['step_times'] else 0.0
        final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
        
        logger.info(f"✅ 2-Step Hierarchical results:")
        logger.info(f"   Best accuracy: {best_accuracy:.4f}")
        logger.info(f"   Final accuracy: {final_accuracy:.4f}")
        logger.info(f"   Avg step time: {avg_step_time*1000:.1f}ms")
        
        return {
            'best_accuracy': best_accuracy,
            'final_accuracy': final_accuracy,
            'avg_step_time': avg_step_time,
            'metrics': metrics
        }
    
    def _process_single_step_batch(self, batch_data, optimizer):
        """Process batch for single-step model"""
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
            output = self.single_step_model(batched_text, batched_audio)
            
            # Compute loss
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
            torch.nn.utils.clip_grad_norm_(self.single_step_model.parameters(), 1.0)
            optimizer.step()
            
            return {
                'total_loss': avg_batch_loss,
                'total_loss_value': avg_batch_loss.item(),
                'accuracy': avg_batch_accuracy
            }
            
        except Exception:
            return None
    
    def _process_hierarchical_batch(self, batch_data, optimizer):
        """Process batch for 2-step hierarchical model"""
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
            
            # Forward pass through 2-step hierarchical model
            optimizer.zero_grad()
            
            # 2-step hierarchical prediction
            output_logits = self.refinement_model(batched_text, batched_audio)
            
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
            torch.nn.utils.clip_grad_norm_(self.refinement_model.parameters(), 1.0)
            optimizer.step()
            
            return {
                'total_loss': avg_batch_loss,
                'total_loss_value': avg_batch_loss.item(),
                'accuracy': avg_batch_accuracy
            }
            
        except Exception as e:
            print(f"❌ Hierarchical batch error: {e}")
            return None
    
    def run_comparison_test(self, steps_per_test=1000):
        """Run complete comparison test"""
        logger.info("🧬" + "="*80)
        logger.info("🧬 PURE MAMBA E768_H768: SINGLE-STEP vs 2-STEP HIERARCHICAL")
        logger.info("🧬" + "="*80)
        logger.info(f"📊 Dataset: no_overlap_data ({self.data_loader.get_num_samples()} samples)")
        logger.info(f"📏 Config: E{self.config.embed_dim}_H{self.config.hidden_dim}_L{self.config.num_layers}_E{self.config.expand_factor}")
        logger.info(f"🎯 Steps per test: {steps_per_test}")
        
        results = {}
        
        # Test 1: Single-step E768_H768
        logger.info(f"\n🎯 TEST 1: SINGLE-STEP E768_H768")
        logger.info("="*50)
        single_step_results = self.test_single_step(steps_per_test)
        results['single_step'] = single_step_results
        
        # Test 2: 2-Step Hierarchical E768_H768
        logger.info(f"\n🔄 TEST 2: 2-STEP HIERARCHICAL E768_H768")
        logger.info("="*50)
        refinement_results = self.test_refinement(steps_per_test)
        results['refinement'] = refinement_results
        
        # Comparison analysis
        logger.info(f"\n📊 COMPARISON ANALYSIS")
        logger.info("="*50)
        
        single_acc = single_step_results['best_accuracy']
        single_speed = single_step_results['avg_step_time'] * 1000
        
        ref_acc = refinement_results['best_accuracy']
        ref_speed = refinement_results['avg_step_time'] * 1000
        # Note: No avg_refinement_steps for 2-step hierarchical
        
        logger.info(f"📈 ACCURACY COMPARISON:")
        logger.info(f"   Single-step: {single_acc:.4f}")
        logger.info(f"   2-Step Hierarchical: {ref_acc:.4f}")
        if ref_acc > single_acc:
            improvement = ((ref_acc - single_acc) / single_acc) * 100
            logger.info(f"   🏆 2-Step Hierarchical wins by {improvement:.1f}%")
        else:
            logger.info(f"   🏆 Single-step wins")
        
        logger.info(f"\n⚡ SPEED COMPARISON:")
        logger.info(f"   Single-step: {single_speed:.1f}ms/step")
        logger.info(f"   2-Step Hierarchical: {ref_speed:.1f}ms/step")
        if single_speed < ref_speed:
            slowdown = ((ref_speed - single_speed) / single_speed) * 100
            logger.info(f"   🚄 Single-step faster by {slowdown:.1f}%")
        else:
            logger.info(f"   🚄 2-Step Hierarchical faster")
        
        # Overall winner
        logger.info(f"\n🏅 OVERALL ASSESSMENT:")
        
        # Efficiency calculation
        single_efficiency = single_acc / (single_speed / 1000)  # accuracy per second
        ref_efficiency = ref_acc / (ref_speed / 1000)
        
        logger.info(f"   Single-step efficiency: {single_efficiency:.2f} acc/sec")
        logger.info(f"   2-Step Hierarchical efficiency: {ref_efficiency:.2f} acc/sec")
        
        if ref_acc > single_acc * 1.05:  # If hierarchical is >5% better
            logger.info(f"   🏆 WINNER: 2-Step Hierarchical (significant accuracy gain)")
            winner = "hierarchical"
        elif single_efficiency > ref_efficiency * 1.2:  # If single-step is >20% more efficient
            logger.info(f"   🏆 WINNER: Single-step (better efficiency)")
            winner = "single_step"
        else:
            logger.info(f"   🤝 TIE: Both approaches have merits")
            winner = "tie"
        
        results['comparison'] = {
            'winner': winner,
            'single_efficiency': single_efficiency,
            'hierarchical_efficiency': ref_efficiency,
            'accuracy_improvement': ((ref_acc - single_acc) / single_acc) * 100 if single_acc > 0 else 0,
            'speed_difference': ((ref_speed - single_speed) / single_speed) * 100 if single_speed > 0 else 0
        }
        
        # Save results
        timestamp = int(time.time())
        results_file = f'e768_h768_hierarchical_comparison_{timestamp}.json'
        
        # Convert tensors to serializable format
        serializable_results = copy.deepcopy(results)
        for test_name in ['single_step', 'refinement']:
            if test_name in serializable_results and 'metrics' in serializable_results[test_name]:
                metrics = serializable_results[test_name]['metrics']
                for key in metrics:
                    if isinstance(metrics[key], list):
                        metrics[key] = [float(x) if hasattr(x, 'item') else x for x in metrics[key]]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Results saved: {results_file}")
        
        # Final recommendations
        logger.info(f"\n🎯 PRODUCTION RECOMMENDATIONS:")
        if winner == "hierarchical":
            logger.info(f"   ✅ Use 2-Step Hierarchical E768_H768 for best quality")
            logger.info(f"   ✅ Coarse→Fine approach works excellently")
            logger.info(f"   ✅ Balanced complexity vs performance")
        elif winner == "single_step":
            logger.info(f"   ✅ Use Single-step E768_H768 for efficiency")
            logger.info(f"   ✅ {single_speed:.0f}ms/step training speed")
            logger.info(f"   ✅ Simpler architecture and deployment")
        else:
            logger.info(f"   🤝 Choose based on requirements:")
            logger.info(f"   📈 Quality priority → 2-Step Hierarchical")
            logger.info(f"   ⚡ Speed priority → Single-step")
        
        logger.info(f"\n🎉 E768_H768 HIERARCHICAL COMPARISON COMPLETED!")
        
        
        return results
    
    def get_num_samples(self):
        """Get number of loaded samples"""
        return self.data_loader.get_num_samples()


def main():
    """Main function"""
    print("🧬 MAIN FUNCTION STARTED!")
    logger.info("🧬 PURE MAMBA E768_H768: SINGLE vs 2-STEP HIERARCHICAL TEST")
    logger.info("="*80)
    print("🧬 STARTING E768_H768 vs 2-STEP HIERARCHICAL TEST...")
    print(f"🖥️  Device: {DEVICE}")
    print("📊 Loading data...")
    
    # Check data
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        print("❌ no_overlap_data directory not found!")
        logger.error("❌ no_overlap_data directory not found!")
        logger.error("   Please ensure greedy search data is available")
        return
    
    print("✅ Data directory found")
    
    # Device setup
    logger.info(f"🖥️  Device: {DEVICE}")
    print(f"🔧 Creating tester...")
    
    try:
        # Create tester 
        print("📝 Initializing tokenizer...")
        tester = E768H768RefinementTester()
        print("✅ Tester created!")
        
        if tester.get_num_samples() == 0:
            print("❌ No samples loaded!")
            logger.error("❌ No samples loaded!")
            return
        
        print(f"✅ Loaded {tester.get_num_samples()} samples")
        
        # Run comparison
        results = tester.run_comparison_test(steps_per_test=1000)
        
        # Final summary
        if results:
            winner = results['comparison']['winner']
            if winner == "hierarchical":
                logger.info("🏆 2-STEP HIERARCHICAL E768_H768 WINS!")
                logger.info("   Coarse→Fine approach provides better quality")
            elif winner == "single_step":
                logger.info("🏆 SINGLE-STEP E768_H768 WINS!")
                logger.info("   Efficiency and simplicity triumph")
            else:
                logger.info("🤝 BOTH APPROACHES VIABLE!")
                logger.info("   Choice depends on specific requirements")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()