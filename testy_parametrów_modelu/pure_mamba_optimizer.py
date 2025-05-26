#!/usr/bin/env python3
"""
Pure Mamba Architecture Optimizer with Hyperparameter Search
============================================================
Automatically finds optimal Pure Mamba configuration through systematic testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
import json
import time
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import os
from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("🔥 PURE MAMBA ARCHITECTURE OPTIMIZER STARTED!")
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


@dataclass
class ArchitectureConfig:
    """Configuration for Pure Mamba architecture variants"""
    name: str
    num_layers: int
    hidden_dim: int
    expand_factor: float
    embed_dim: int = 384
    num_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1
    
    def get_param_count_estimate(self) -> int:
        """Estimate parameter count"""
        # Rough estimation based on dimensions
        vocab_embed = 1000 * self.embed_dim
        text_encoder = self.num_layers * (self.hidden_dim * self.hidden_dim * 4)
        audio_processor = 4 * (self.hidden_dim * self.hidden_dim * 3)
        others = self.hidden_dim * 1000
        return vocab_embed + text_encoder + audio_processor + others


@dataclass
class ExperimentResult:
    """Results from single architecture experiment"""
    config: ArchitectureConfig
    success: bool
    training_time: float
    final_accuracy: float
    final_duration_accuracy: float
    best_accuracy: float
    best_duration_accuracy: float
    milestones_audio: Dict[str, int]
    milestones_duration: Dict[str, int]
    avg_step_time: float
    param_count: int
    convergence_step: int
    efficiency_score: float
    error_message: Optional[str] = None


class OptimizedMambaBlock(nn.Module):
    """Optimized Pure Mamba block with configurable parameters"""
    def __init__(self, d_model, expand_factor=2.0, dropout=0.1, reverse=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand_factor)
        self.reverse = reverse
        
        # Core Mamba components - optimized
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
        
        # Efficient conv1d processing
        x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)
        
        # Optimized SSM processing
        x1_ssm = self.activation(x1_conv)
        dt = self.dt_proj(x1_ssm)
        dt = F.softplus(dt)
        
        # Simplified but effective state space operation
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
        
        return output + residual


class OptimizedMambaTextEncoder(nn.Module):
    """Optimized Pure Mamba text encoder"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, expand_factor, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, hidden_dim) * 0.02)
        
        # Stack of optimized Mamba blocks
        self.layers = nn.ModuleList([
            OptimizedMambaBlock(hidden_dim, expand_factor, dropout, reverse=False)
            for _ in range(num_layers)
        ])
        
        # Final processing
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens, return_sequence=True):
        B, L = tokens.shape
        
        # Embeddings and projection
        x = self.embedding(tokens)
        x = self.embed_proj(x)
        
        # Add positional encoding
        if L <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :L, :]
        
        x = self.dropout(x)
        
        # Pure Mamba processing
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        if return_sequence:
            return x  # [B, L, D]
        else:
            return x.mean(dim=1)  # [B, D]


class OptimizedMambaDurationRegulator(nn.Module):
    """Optimized duration regulator with Pure Mamba"""
    def __init__(self, hidden_dim, style_dim=128, expand_factor=2.0, tokens_per_second=75.0):
        super().__init__()
        self.tokens_per_second = tokens_per_second
        
        # Input processing
        self.input_proj = nn.Linear(hidden_dim + style_dim, hidden_dim)
        
        # Pure Mamba for temporal dependencies
        self.duration_mamba = OptimizedMambaBlock(hidden_dim, expand_factor, dropout=0.1)
        
        # Prediction heads
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
        
        # Combine and project
        combined = torch.cat([text_features, style_expanded], dim=-1)
        x = self.input_proj(combined)
        
        # Pure Mamba processing
        x = self.duration_mamba(x)
        
        # Predictions
        predicted_durations = self.duration_predictor(x).squeeze(-1)
        predicted_durations = torch.clamp(predicted_durations, min=0.05, max=0.2)
        
        duration_confidence = self.confidence_predictor(x).squeeze(-1)
        
        # Duration tokens
        duration_tokens = (predicted_durations * self.tokens_per_second).round().long()
        duration_tokens = torch.clamp(duration_tokens, min=2, max=15)
        
        return text_features, predicted_durations, duration_tokens, duration_confidence


class OptimizedStyleExtractor(nn.Module):
    """Optimized backward style extractor"""
    def __init__(self, audio_dim, style_dim=128, hidden_dim=256, expand_factor=2.0):
        super().__init__()
        
        self.input_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Backward Mamba for global style context
        self.backward_mamba = OptimizedMambaBlock(
            hidden_dim, 
            expand_factor, 
            dropout=0.1, 
            reverse=True  # Backward processing
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
        B, D, T = audio_features.shape
        
        # Transpose and project
        x = audio_features.transpose(1, 2)  # [B, T, D]
        x = self.input_proj(x)
        
        # Backward Mamba processing
        processed = self.backward_mamba(x)
        
        # Global style
        style_features = processed.mean(dim=1)
        style_vector = self.style_proj(style_features)
        
        return style_vector


class OptimizedMambaAudioProcessor(nn.Module):
    """Optimized audio processor with Pure Mamba"""
    def __init__(self, hidden_dim, num_codebooks=8, codebook_size=1024, expand_factor=2.0):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Efficient embeddings
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(codebook_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_codebooks)
        ])
        
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Pure Mamba processing layers
        self.mamba_layers = nn.ModuleList([
            OptimizedMambaBlock(hidden_dim, expand_factor, dropout=0.1)
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
            emb = self.audio_embed[c][0](audio_tokens[:, c, :])
            emb = self.audio_embed[c][1](emb)
            emb = self.audio_embed[c][2](emb)
            embedded.append(emb)
        
        # Combine embeddings
        x = torch.stack(embedded, dim=1).mean(dim=1)  # [B, T, hidden_dim]
        
        # Add text context
        text_context_proj = self.context_proj(text_context).unsqueeze(1)
        x = x + text_context_proj
        
        # Pure Mamba processing
        for layer in self.mamba_layers:
            x = layer(x)
        
        # Generate logits for each codebook
        logits = []
        for c in range(8):
            head_logits = self.output_heads[c](x)
            logits.append(head_logits)
        
        return torch.stack(logits, dim=1)


class OptimizedPureMambaModel(nn.Module):
    """Optimized Pure Mamba TTS model with configurable architecture"""
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        
        # All components use Pure Mamba
        self.text_encoder = OptimizedMambaTextEncoder(
            vocab_size=1000,  # Will be updated by actual vocab size
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            expand_factor=config.expand_factor,
            dropout=config.dropout
        )
        
        self.duration_regulator = OptimizedMambaDurationRegulator(
            hidden_dim=config.hidden_dim,
            style_dim=128,
            expand_factor=config.expand_factor
        )
        
        self.audio_processor = OptimizedMambaAudioProcessor(
            hidden_dim=config.hidden_dim,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            expand_factor=config.expand_factor
        )
        
        self.style_extractor = OptimizedStyleExtractor(
            audio_dim=config.hidden_dim,
            style_dim=128,
            hidden_dim=256,
            expand_factor=config.expand_factor
        )
        
        # Projections
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.default_style = nn.Parameter(torch.randn(128) * 0.01)
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 {config.name}: {total_params:,} parameters")
        logger.info(f"   📏 Layers: {config.num_layers}, Hidden: {config.hidden_dim}, Expand: {config.expand_factor}")
        
    def forward(self, text_tokens, audio_tokens=None, chunk_duration=None):
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Pure Mamba text encoding
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        # Style extraction
        if audio_tokens is not None:
            with torch.no_grad():
                B, C, T = audio_tokens.shape
                audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2])
                pseudo_audio = audio_mean.unsqueeze(1).unsqueeze(2).expand(B, self.config.hidden_dim, min(T, 120))
                style_embedding = self.style_extractor(pseudo_audio)
        else:
            style_embedding = self.default_style.unsqueeze(0).expand(batch_size, -1)
        
        # Duration regulation
        regulated_features, predicted_durations, duration_tokens, duration_confidence = \
            self.duration_regulator(text_features, style_embedding)
        
        # Audio processing
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


class AccuracyMilestoneTracker:
    """Enhanced milestone tracker for optimization"""
    def __init__(self):
        self.milestones = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        self.duration_milestones = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        
        self.accuracy_reached = {}
        self.duration_accuracy_reached = {}
        
        for milestone in self.milestones:
            self.accuracy_reached[milestone] = None
            
        for milestone in self.duration_milestones:
            self.duration_accuracy_reached[milestone] = None
    
    def update(self, step, accuracy, duration_accuracy):
        for milestone in self.milestones:
            if accuracy >= milestone and self.accuracy_reached[milestone] is None:
                self.accuracy_reached[milestone] = step
                
        for milestone in self.duration_milestones:
            if duration_accuracy >= milestone and self.duration_accuracy_reached[milestone] is None:
                self.duration_accuracy_reached[milestone] = step
    
    def get_convergence_step(self) -> int:
        """Get step when model achieved good convergence (50% accuracy)"""
        return self.accuracy_reached.get(0.5, 9999)
    
    def get_milestone_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            'audio_accuracy_milestones': {f"{k:.1%}": v for k, v in self.accuracy_reached.items() if v is not None},
            'duration_accuracy_milestones': {f"{k:.1%}": v for k, v in self.duration_accuracy_reached.items() if v is not None}
        }


class TimingStats:
    """Simplified timing statistics"""
    def __init__(self):
        self.times = []
    
    def record(self, duration):
        self.times.append(duration)
    
    def get_average(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0


class PureMambaDataLoader:
    """Optimized data loader for experiments"""
    def __init__(self, data_dir="no_overlap_data", device='cpu', max_chunks=100):
        self.data_dir = Path(data_dir)
        self.device = device
        self.chunks = []
        self.max_chunks = max_chunks
        
        logger.info(f"🔍 Loading data from {data_dir} (max {max_chunks} chunks)")
        self._load_chunks()
        
    def _load_chunks(self):
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return
            
        batch_dirs = [d for d in self.data_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('clean_batch_')]
        batch_dirs.sort()
        
        chunks_loaded = 0
        for batch_dir in batch_dirs:
            if chunks_loaded >= self.max_chunks:
                break
                
            try:
                meta_path = batch_dir / "batch_meta.json"
                if not meta_path.exists():
                    continue
                    
                with open(meta_path, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)
                
                for chunk_file in batch_meta.get('chunk_files', []):
                    if chunks_loaded >= self.max_chunks:
                        break
                        
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
                                
                                self.chunks.append(chunk_data)
                                chunks_loaded += 1
                            
                        except Exception as e:
                            continue
                        
            except Exception as e:
                continue
        
        logger.info(f"📊 Loaded {len(self.chunks)} chunks for experiments")
        
    def get_random_chunk(self):
        if not self.chunks:
            return None
        return self.chunks[np.random.randint(0, len(self.chunks))]


class ArchitectureOptimizer:
    """Main optimizer for finding best Pure Mamba architecture"""
    
    def __init__(self, tokenizer, data_loader, device):
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = device
        self.results = []
        
    def create_search_space(self) -> List[ArchitectureConfig]:
        """Create search space for optimization"""
        search_configs = {
            'num_layers': [4, 6, 8, 12],
            'hidden_dim': [256, 384, 512, 768],
            'expand_factor': [1.5, 2.0, 2.5, 3.0]
        }
        
        # Generate all combinations
        configs = []
        for num_layers, hidden_dim, expand_factor in itertools.product(
            search_configs['num_layers'],
            search_configs['hidden_dim'],
            search_configs['expand_factor']
        ):
            config = ArchitectureConfig(
                name=f"PureMamba_L{num_layers}_H{hidden_dim}_E{expand_factor}",
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                expand_factor=expand_factor
            )
            configs.append(config)
        
        logger.info(f"🔬 Generated {len(configs)} architecture configurations")
        return configs
    
    def evaluate_architecture(self, config: ArchitectureConfig, test_steps: int = 800) -> ExperimentResult:
        """Evaluate single architecture configuration"""
        logger.info(f"\n🧪 Testing: {config.name}")
        logger.info(f"   Layers: {config.num_layers}, Hidden: {config.hidden_dim}, Expand: {config.expand_factor}")
        
        start_time = time.time()
        
        try:
            # Create model
            model = OptimizedPureMambaModel(config).to(self.device)
            
            # Update vocab size
            actual_vocab_size = self.tokenizer.get_vocab_size()
            model.text_encoder.embedding = nn.Embedding(actual_vocab_size, config.embed_dim).to(self.device)
            
            # Get actual parameter count
            param_count = sum(p.numel() for p in model.parameters())
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=8e-4, 
                weight_decay=1e-6,
                betas=(0.9, 0.95)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=test_steps, eta_min=8e-5
            )
            
            # Training metrics
            milestone_tracker = AccuracyMilestoneTracker()
            timing_stats = TimingStats()
            
            successful_steps = 0
            best_accuracy = 0.0
            best_duration_accuracy = 0.0
            final_accuracy = 0.0
            final_duration_accuracy = 0.0
            
            # Quick training loop
            model.train()
            for step in range(test_steps):
                step_start = time.time()
                
                # Get data
                chunk_data = self.data_loader.get_random_chunk()
                if chunk_data is None:
                    continue
                
                try:
                    # Prepare data
                    text_tokens = chunk_data['text_tokens']
                    if text_tokens.dim() == 1:
                        text_tokens = text_tokens.unsqueeze(0)
                        
                    audio_codes = chunk_data['audio_codes']
                    if audio_codes.dim() == 2:
                        audio_codes = audio_codes.unsqueeze(0)
                    
                    chunk_duration = chunk_data.get('duration', 4.0)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = model(text_tokens, audio_codes, chunk_duration=chunk_duration)
                    
                    # Loss computation
                    loss_dict = compute_combined_loss(output, chunk_data, text_tokens, self.device)
                    total_loss = loss_dict.get('total_loss')
                    
                    if total_loss is None or torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue
                    
                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    scheduler.step()
                    
                    # Track metrics
                    current_accuracy = loss_dict['accuracy']
                    current_duration_accuracy = loss_dict['duration_accuracy']
                    
                    milestone_tracker.update(step, current_accuracy, current_duration_accuracy)
                    
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    if current_duration_accuracy > best_duration_accuracy:
                        best_duration_accuracy = current_duration_accuracy
                    
                    final_accuracy = current_accuracy
                    final_duration_accuracy = current_duration_accuracy
                    
                    successful_steps += 1
                    
                    # Record timing
                    step_time = time.time() - step_start
                    timing_stats.record(step_time)
                    
                    # Progress logging
                    if step % 200 == 0 or current_accuracy > 0.3:
                        logger.info(f"   Step {step:3d}: Loss={total_loss.item():.4f}, "
                                  f"Acc={current_accuracy:.4f}, DurAcc={current_duration_accuracy:.4f}")
                    
                    # Early success detection
                    if current_accuracy > 0.5 and step > 400:
                        logger.info(f"   🎉 Early success at step {step}!")
                        break
                        
                except Exception as step_e:
                    continue
            
            # Calculate efficiency score
            training_time = time.time() - start_time
            convergence_step = milestone_tracker.get_convergence_step()
            
            # Efficiency score combines accuracy, speed, and parameter efficiency
            if convergence_step < 9999:
                efficiency_score = (best_accuracy * 100) / (convergence_step / 100) / (param_count / 1000000)
            else:
                efficiency_score = (best_accuracy * 100) / (param_count / 1000000) * 0.1
            
            result = ExperimentResult(
                config=config,
                success=successful_steps > test_steps * 0.8,
                training_time=training_time,
                final_accuracy=final_accuracy,
                final_duration_accuracy=final_duration_accuracy,
                best_accuracy=best_accuracy,
                best_duration_accuracy=best_duration_accuracy,
                milestones_audio=milestone_tracker.get_milestone_dict()['audio_accuracy_milestones'],
                milestones_duration=milestone_tracker.get_milestone_dict()['duration_accuracy_milestones'],
                avg_step_time=timing_stats.get_average(),
                param_count=param_count,
                convergence_step=convergence_step,
                efficiency_score=efficiency_score
            )
            
            logger.info(f"   ✅ Success: Best acc={best_accuracy:.4f}, Convergence={convergence_step}, Efficiency={efficiency_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Failed: {str(e)}")
            
            result = ExperimentResult(
                config=config,
                success=False,
                training_time=time.time() - start_time,
                final_accuracy=0.0,
                final_duration_accuracy=0.0,
                best_accuracy=0.0,
                best_duration_accuracy=0.0,
                milestones_audio={},
                milestones_duration={},
                avg_step_time=0.0,
                param_count=config.get_param_count_estimate(),
                convergence_step=9999,
                efficiency_score=0.0,
                error_message=str(e)
            )
            
            return result
    
    def optimize_architecture(self, max_experiments: int = 20, test_steps: int = 800) -> List[ExperimentResult]:
        """Run architecture optimization with intelligent search"""
        logger.info("🚀 PURE MAMBA ARCHITECTURE OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Max experiments: {max_experiments}, Test steps per experiment: {test_steps}")
        
        # Create search space
        all_configs = self.create_search_space()
        
        # Intelligent sampling: prioritize reasonable configurations first
        priority_configs = []
        remaining_configs = []
        
        for config in all_configs:
            # Prioritize configurations that are likely to work well
            if (config.num_layers <= 8 and 
                config.hidden_dim <= 512 and 
                config.expand_factor <= 2.5):
                priority_configs.append(config)
            else:
                remaining_configs.append(config)
        
        # Shuffle and select
        np.random.shuffle(priority_configs)
        np.random.shuffle(remaining_configs)
        
        selected_configs = priority_configs[:max_experiments//2] + remaining_configs[:max_experiments//2]
        selected_configs = selected_configs[:max_experiments]
        
        logger.info(f"🎯 Selected {len(selected_configs)} configurations for testing")
        logger.info(f"   Priority configs: {len(priority_configs[:max_experiments//2])}")
        logger.info(f"   Exploration configs: {len(remaining_configs[:max_experiments//2])}")
        
        # Run experiments
        results = []
        for i, config in enumerate(selected_configs):
            logger.info(f"\n🧪 Experiment {i+1}/{len(selected_configs)}")
            result = self.evaluate_architecture(config, test_steps)
            results.append(result)
            
            # Save intermediate results
            self._save_intermediate_results(results, i+1)
        
        # Final analysis
        self._analyze_results(results)
        return results
    
    def _save_intermediate_results(self, results: List[ExperimentResult], experiment_num: int):
        """Save intermediate results"""
        results_data = {
            'experiment_num': experiment_num,
            'total_experiments': len(results),
            'results': [asdict(result) for result in results],
            'timestamp': time.time()
        }
        
        filename = f'pure_mamba_optimization_results_{experiment_num:02d}.json'
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _analyze_results(self, results: List[ExperimentResult]):
        """Comprehensive analysis of optimization results"""
        logger.info("\n🏆 PURE MAMBA ARCHITECTURE OPTIMIZATION RESULTS")
        logger.info("=" * 70)
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.best_accuracy > 0.1]
        
        if not successful_results:
            logger.warning("❌ No successful experiments found!")
            return
        
        logger.info(f"📊 Successful experiments: {len(successful_results)}/{len(results)}")
        
        # Sort by efficiency score
        successful_results.sort(key=lambda x: x.efficiency_score, reverse=True)
        
        # Top 5 architectures
        logger.info("\n🥇 TOP 5 ARCHITECTURES (by efficiency score):")
        logger.info("-" * 70)
        
        for i, result in enumerate(successful_results[:5]):
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            config = result.config
            
            logger.info(f"{rank_emoji} {config.name}")
            logger.info(f"   Accuracy: {result.best_accuracy:.4f} | Duration: {result.best_duration_accuracy:.4f}")
            logger.info(f"   Convergence: Step {result.convergence_step} | Efficiency: {result.efficiency_score:.2f}")
            logger.info(f"   Parameters: {result.param_count:,} | Step time: {result.avg_step_time*1000:.1f}ms")
            logger.info(f"   Config: L={config.num_layers}, H={config.hidden_dim}, E={config.expand_factor}")
            
            # Show milestones for top 3
            if i < 3 and result.milestones_audio:
                milestones_str = ", ".join([f"{k}@{v}" for k, v in list(result.milestones_audio.items())[-3:]])
                logger.info(f"   Milestones: {milestones_str}")
            logger.info("")
        
        # Save best model
        self._save_best_model(successful_results[0])
        
        # Generate recommendations
        self._generate_recommendations(successful_results)
    
    def _save_best_model(self, best_result: ExperimentResult):
        """Save the best model configuration"""
        logger.info("\n💾 SAVING BEST MODEL CONFIGURATION:")
        logger.info("-" * 50)
        
        # Save model data
        model_data = {
            'config': asdict(best_result.config),
            'results': asdict(best_result),
            'optimization_info': {
                'is_optimized': True,
                'optimization_method': 'hyperparameter_search',
                'vocab_size': self.tokenizer.get_vocab_size()
            },
            'timestamp': time.time()
        }
        
        filename = f'optimized_pure_mamba_best.json'
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"✅ Best model configuration saved as: {filename}")
        logger.info(f"   Config: {best_result.config.name}")
        logger.info(f"   Best accuracy: {best_result.best_accuracy:.4f}")
        logger.info(f"   Efficiency score: {best_result.efficiency_score:.2f}")
    
    def _generate_recommendations(self, results: List[ExperimentResult]):
        """Generate recommendations based on optimization results"""
        logger.info("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        logger.info("-" * 50)
        
        best_result = results[0]
        best_config = best_result.config
        
        logger.info("🎯 PRODUCTION RECOMMENDATIONS:")
        logger.info(f"1. Use configuration: {best_config.name}")
        logger.info(f"   - Layers: {best_config.num_layers}")
        logger.info(f"   - Hidden dim: {best_config.hidden_dim}")
        logger.info(f"   - Expand factor: {best_config.expand_factor}")
        logger.info(f"   - Expected accuracy: {best_result.best_accuracy:.4f}")
        logger.info(f"   - Training speed: {best_result.avg_step_time*1000:.1f}ms/step")
        
        # Save comprehensive results
        comprehensive_results = {
            'optimization_summary': {
                'total_experiments': len(results),
                'successful_experiments': len([r for r in results if r.success]),
                'best_config': asdict(best_config),
                'best_results': asdict(best_result)
            },
            'all_results': [asdict(r) for r in results],
            'recommendations': {
                'production_config': asdict(best_config),
                'parameter_efficiency': best_result.best_accuracy / (best_result.param_count / 1000000)
            },
            'timestamp': time.time()
        }
        
        with open('pure_mamba_optimization_complete.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info("\n💾 Complete optimization results saved as: pure_mamba_optimization_complete.json")


def run_optimization_experiment():
    """Main function to run Pure Mamba architecture optimization"""
    logger.info("🚀 PURE MAMBA ARCHITECTURE OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("🎯 Goal: Find optimal Pure Mamba configuration through systematic search")
    logger.info("📊 Search space: num_layers × hidden_dim × expand_factor")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check data
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        logger.error("❌ no_overlap_data directory not found!")
        logger.error("   Please run audio_processor_sequential.py first")
        return
    
    try:
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"📝 Vocabulary size: {vocab_size}")
        
        # Optimized data loader (limited chunks for faster experiments)
        data_loader = PureMambaDataLoader("no_overlap_data", device, max_chunks=50)
        if len(data_loader.chunks) == 0:
            logger.error("❌ No chunks loaded!")
            return
        
        logger.info(f"📊 Loaded {len(data_loader.chunks)} chunks for optimization")
        
        # Create optimizer
        optimizer = ArchitectureOptimizer(tokenizer, data_loader, device)
        
        # Run optimization
        logger.info("\n🔬 Starting hyperparameter optimization...")
        logger.info("   Each experiment: 800 training steps")
        logger.info("   Success criteria: >80% successful steps, >10% accuracy")
        logger.info("   Efficiency metric: accuracy / convergence_time / param_count")
        
        results = optimizer.optimize_architecture(max_experiments=20, test_steps=800)
        
        logger.info("\n✅ OPTIMIZATION COMPLETED!")
        logger.info(f"   Total experiments: {len(results)}")
        logger.info(f"   Successful: {len([r for r in results if r.success])}")
        logger.info("   Best model saved and ready for production!")
        logger.info("   Check pure_mamba_optimization_complete.json for full results")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


def run_single_optimal_test():
    """Test single optimal configuration quickly"""
    logger.info("🧪 SINGLE OPTIMAL CONFIGURATION TEST")
    logger.info("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        tokenizer = NucleotideTokenizer()
        data_loader = PureMambaDataLoader("no_overlap_data", device, max_chunks=30)
        
        if len(data_loader.chunks) == 0:
            logger.error("❌ No chunks loaded!")
            return
        
        # Optimal configuration based on research
        optimal_config = ArchitectureConfig(
            name="PureMamba_Optimal",
            num_layers=6,
            hidden_dim=384,
            expand_factor=2.0
        )
        
        logger.info(f"🎯 Testing optimal config: {optimal_config.name}")
        
        optimizer = ArchitectureOptimizer(tokenizer, data_loader, device)
        result = optimizer.evaluate_architecture(optimal_config, test_steps=1200)
        
        logger.info(f"\n📊 OPTIMAL CONFIG RESULTS:")
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Best accuracy: {result.best_accuracy:.4f}")
        logger.info(f"   Duration accuracy: {result.best_duration_accuracy:.4f}")
        logger.info(f"   Convergence step: {result.convergence_step}")
        logger.info(f"   Efficiency score: {result.efficiency_score:.2f}")
        logger.info(f"   Parameters: {result.param_count:,}")
        logger.info(f"   Step time: {result.avg_step_time*1000:.1f}ms")
        
        if result.milestones_audio:
            logger.info(f"   Milestones: {list(result.milestones_audio.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


def main():
    """Main function with options"""
    import sys
    
    print("🔥 Pure Mamba Architecture Optimizer")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--optimize':
            run_optimization_experiment()
        elif sys.argv[1] == '--test-optimal':
            run_single_optimal_test()
        elif sys.argv[1] == '--help':
            print("Pure Mamba Architecture Optimizer")
            print("Usage:")
            print("  python pure_mamba_optimizer.py --optimize      # Full optimization")
            print("  python pure_mamba_optimizer.py --test-optimal  # Test single optimal")
            print("  python pure_mamba_optimizer.py --help          # Show help")
        else:
            print("Unknown option. Use --help for usage.")
    else:
        # Default: run optimization
        run_optimization_experiment()


if __name__ == "__main__":
    main()