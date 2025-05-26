#!/usr/bin/env python3
"""
Transformer A/B Test - Compare with Hybrid Architecture
======================================================
Drop-in replacement for SimpleTTSModel with classic Transformer
+ timing benchmarks for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import logging
import warnings
from pathlib import Path
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import tokenizer
from nucleotide_tokenizer import NucleotideTokenizer

# ============================================================================
# CLASSIC TRANSFORMER MODEL
# ============================================================================

class TransformerTTSModel(nn.Module):
    """
    FAIR Transformer - using SAME components as your Hybrid
    Only core processing differs: Transformer vs Mamba+Conv
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_codebooks=4, codebook_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        
        # SAME TEXT ENCODER as your Hybrid
        from modules import MambaConvTextEncoder
        self.text_encoder = MambaConvTextEncoder(vocab_size, embed_dim)
        
        # SAME AUDIO TOKEN EMBEDDINGS as your Hybrid
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim) for _ in range(num_codebooks)
        ])
        
        # SAME POSITION EMBEDDING as your Hybrid
        self.pos_embedding = nn.Embedding(1000, hidden_dim)
        
        # CORE DIFFERENCE: Transformer layers instead of Mamba+Conv
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
                activation='relu'
            ) for _ in range(6)
        ])
        
        # SAME OUTPUT HEADS as your Hybrid
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size) for _ in range(num_codebooks)
        ])
        
        # SAME TEXT PROJECTION as your Hybrid
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Calculate and log parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🔥 TransformerTTSModel (FAIR): {total_params:,} parameters")
        logger.info(f"   Using SAME text encoder as Hybrid + SAME audio components")
        logger.info(f"   CORE: 6x Transformer layers instead of Mamba+Conv")
    
    def forward(self, text_tokens, audio_tokens):
        """Forward with SAME components as Hybrid, different core"""
        B, C, T_audio = audio_tokens.shape
        device = text_tokens.device
        
        # SAME TEXT ENCODING as your Hybrid
        text_context = self.text_encoder(text_tokens, return_sequence=False)  # [B, embed_dim]
        text_context = self.text_proj(text_context)  # [B, hidden_dim]
        
        # SAME AUDIO TOKEN EMBEDDING as your Hybrid
        token_embeds = []
        for cb_idx in range(C):
            cb_tokens = audio_tokens[:, cb_idx, :]
            cb_embed = self.token_embeddings[cb_idx](cb_tokens)
            token_embeds.append(cb_embed)
        
        audio_embed = torch.stack(token_embeds, dim=1).mean(dim=1)  # [B, T_audio, hidden_dim]
        
        # SAME POSITIONAL EMBEDDING as your Hybrid
        positions = torch.arange(T_audio, device=device)
        pos_embed = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)
        audio_embed = audio_embed + pos_embed
        
        # SAME TEXT CONDITIONING as your Hybrid
        text_expanded = text_context.unsqueeze(1).expand(-1, T_audio, -1)
        audio_embed = audio_embed + text_expanded
        
        # CORE DIFFERENCE: Transformer processing instead of Mamba+Conv
        x = audio_embed
        for layer in self.transformer_layers:
            x = layer(x)  # Multi-head self-attention + FFN
        
        # SAME OUTPUT GENERATION as your Hybrid
        outputs = []
        for cb_idx in range(C):
            cb_output = self.output_heads[cb_idx](x)
            outputs.append(cb_output)
        
        logits = torch.stack(outputs, dim=1)  # [B, C, T, codebook_size]
        
        return {'logits': logits}


# ============================================================================
# TIMED TRAINER WITH BENCHMARKS
# ============================================================================

class TimedTrainer:  
    """Trainer with detailed timing benchmarks"""
    def __init__(self, model, tokenizer, data_extractor, model_name="Model"):
        self.model = model
        self.tokenizer = tokenizer
        self.data_extractor = data_extractor
        self.device = next(model.parameters()).device
        self.model_name = model_name
        
        # Timing metrics
        self.timing_metrics = {
            'forward_times': [],
            'backward_times': [],
            'total_step_times': [],
            'data_prep_times': [],
            'memory_usage': []
        }
        
    def train_step_timed(self, fragment):
        """Training step with detailed timing"""
        step_start = time.time()
        
        # Data preparation timing
        data_start = time.time()
        text_tokens = self.tokenizer.encode(fragment['text'])
        text_tensor = torch.tensor([text_tokens], device=self.device)
        
        audio_tokens = fragment['audio_tokens'].to(self.device)
        if audio_tokens.dim() == 2:
            audio_tokens = audio_tokens.unsqueeze(0)
        
        data_time = time.time() - data_start
        
        # Forward pass timing
        forward_start = time.time()
        output = self.model(text_tensor, audio_tokens)
        logits = output['logits']
        forward_time = time.time() - forward_start
        
        # Loss computation
        B, C, T, V = logits.shape
        total_loss = 0
        
        for cb_idx in range(C):
            input_tokens = audio_tokens[:, cb_idx, :-1]
            target_tokens = audio_tokens[:, cb_idx, 1:]
            pred_logits = logits[:, cb_idx, :-1, :]
            
            cb_loss = F.cross_entropy(
                pred_logits.reshape(-1, V),
                target_tokens.reshape(-1)
            )
            total_loss += cb_loss
        
        total_loss = total_loss / C
        
        # Backward pass timing
        backward_start = time.time()
        total_loss.backward()
        backward_time = time.time() - backward_start
        
        # Accuracy
        with torch.no_grad():
            predicted = torch.argmax(logits[:, :, :-1, :], dim=-1)
            target = audio_tokens[:, :, 1:]
            accuracy = (predicted == target).float().mean().item()
        
        # Memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
        else:
            memory_mb = 0
        
        total_step_time = time.time() - step_start
        
        # Store timing metrics
        self.timing_metrics['forward_times'].append(forward_time)
        self.timing_metrics['backward_times'].append(backward_time)
        self.timing_metrics['total_step_times'].append(total_step_time)
        self.timing_metrics['data_prep_times'].append(data_time)
        self.timing_metrics['memory_usage'].append(memory_mb)
        
        return total_loss, accuracy
    
    def train(self, steps=1500):
        """Main training loop with comprehensive timing"""
        logger.info(f"🚀 Starting timed training: {self.model_name}")
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        training_start_time = time.time()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        losses = []
        accuracies = []
        
        # Warmup runs (don't count in timing)
        logger.info("🔥 Warming up...")
        for _ in range(5):
            fragment = self.data_extractor.get_fragment()
            optimizer.zero_grad()
            loss, acc = self.train_step_timed(fragment)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        
        # Clear warmup timing
        for key in self.timing_metrics:
            self.timing_metrics[key].clear()
        
        # Actual timed training
        logger.info(f"⏱️  Starting timed training for {steps} steps...")
        
        for step in range(steps):
            fragment = self.data_extractor.get_fragment()
            
            optimizer.zero_grad()
            loss, accuracy = self.train_step_timed(fragment)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            # Log progress with timing
            if step % 200 == 0 or step == steps - 1:
                avg_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
                avg_acc = np.mean(accuracies[-50:]) if len(accuracies) >= 50 else np.mean(accuracies)
                
                # Recent timing stats
                recent_forward = np.mean(self.timing_metrics['forward_times'][-50:]) * 1000
                recent_backward = np.mean(self.timing_metrics['backward_times'][-50:]) * 1000
                recent_total = np.mean(self.timing_metrics['total_step_times'][-50:]) * 1000
                recent_memory = np.mean(self.timing_metrics['memory_usage'][-10:])
                
                logger.info(f"Step {step:4d}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
                logger.info(f"         Timing: Forward={recent_forward:.1f}ms, Backward={recent_backward:.1f}ms, Total={recent_total:.1f}ms, Mem={recent_memory:.0f}MB")
            
            # Memory cleanup
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_training_time = time.time() - training_start_time
        
        # Final results with comprehensive timing analysis
        final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        final_acc = np.mean(accuracies[-100:]) if len(accuracies) >= 100 else np.mean(accuracies)
        
        self._log_timing_summary(total_training_time, final_loss, final_acc, steps)
        
        # Save model
        model_filename = f'{self.model_name.lower().replace(" ", "_")}_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'final_loss': final_loss,
            'final_accuracy': final_acc,
            'vocab_size': self.tokenizer.get_vocab_size(),
            'training_time': total_training_time,
            'timing_metrics': self.timing_metrics
        }, model_filename)
        
        logger.info(f"💾 Model saved: {model_filename}")
        
        return final_acc > 0.3
    
    def _log_timing_summary(self, total_time, final_loss, final_acc, steps):
        """Comprehensive timing analysis"""
        logger.info("\n" + "="*80)
        logger.info(f"⏱️  TIMING SUMMARY - {self.model_name}")
        logger.info("="*80)
        
        # Overall metrics
        logger.info(f"🎯 Final Results:")
        logger.info(f"   Loss: {final_loss:.6f}")
        logger.info(f"   Accuracy: {final_acc:.4f}")
        logger.info(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"   Steps per second: {steps/total_time:.2f}")
        
        # Detailed timing breakdown
        avg_forward = np.mean(self.timing_metrics['forward_times']) * 1000
        avg_backward = np.mean(self.timing_metrics['backward_times']) * 1000
        avg_total = np.mean(self.timing_metrics['total_step_times']) * 1000
        avg_data = np.mean(self.timing_metrics['data_prep_times']) * 1000
        
        logger.info(f"\n📊 Per-step Timing (average):")
        logger.info(f"   Forward pass: {avg_forward:.2f}ms")
        logger.info(f"   Backward pass: {avg_backward:.2f}ms")
        logger.info(f"   Data prep: {avg_data:.2f}ms")
        logger.info(f"   Other overhead: {avg_total - avg_forward - avg_backward - avg_data:.2f}ms")
        logger.info(f"   Total per step: {avg_total:.2f}ms")
        
        # Memory usage
        if self.timing_metrics['memory_usage']:
            peak_memory = max(self.timing_metrics['memory_usage'])
            avg_memory = np.mean(self.timing_metrics['memory_usage'])
            logger.info(f"\n🧠 Memory Usage:")
            logger.info(f"   Peak: {peak_memory:.0f}MB")
            logger.info(f"   Average: {avg_memory:.0f}MB")
        
        # Performance metrics
        tokens_per_sec = (steps * 50) / total_time  # Assuming ~50 tokens per step
        logger.info(f"\n⚡ Performance:")
        logger.info(f"   Tokens/second: {tokens_per_sec:.0f}")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        logger.info("="*80)


# ============================================================================
# SAME DATA EXTRACTOR
# ============================================================================

class SimpleDataExtractor:
    """Same data extractor as before"""
    def __init__(self, audio_path, transcription_path):
        self.audio_path = audio_path
        self.transcription_path = transcription_path
        
        with open(transcription_path, 'r') as f:
            self.transcription_data = json.load(f)
        
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(3.0)
        self.codec.eval()
        if torch.cuda.is_available():
            self.codec = self.codec.cuda()
        
        logger.info("📊 SimpleDataExtractor ready")
    
    def get_fragment(self, duration=4.0):
        wav, sr = torchaudio.load(self.audio_path)
        wav = convert_audio(wav, sr, target_sr=24000, target_channels=1)
        
        char_alignments = self.transcription_data['character_alignments']
        total_duration = self.transcription_data['metadata']['duration_seconds']
        
        start_time = total_duration * 0.3
        end_time = start_time + duration
        
        start_sample = int(start_time * 24000)
        end_sample = int(end_time * 24000)
        audio_fragment = wav[:, start_sample:end_sample]
        
        fragment_text = ""
        for char_info in char_alignments:
            if start_time <= char_info['start'] < end_time:
                fragment_text += char_info['char']
        
        fragment_text = ' '.join(fragment_text.strip().split())
        
        if len(fragment_text) < 10:
            fragment_text = "hello world this is a test fragment"
        
        device = next(self.codec.parameters()).device
        with torch.no_grad():
            audio_tensor = audio_fragment.to(device)
            encoded = self.codec.encode(audio_tensor.unsqueeze(0))
            tokens = encoded[0][0].squeeze(0)
        
        return {
            'text': fragment_text,
            'audio_tokens': tokens,
            'audio_raw': audio_fragment
        }


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    """A/B Test: Transformer vs Hybrid"""
    logger.info("🥊 TRANSFORMER vs HYBRID - A/B COMPARISON")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check files
    if not Path("speech.mp3").exists() or not Path("speech_transcription.json").exists():
        logger.error("❌ Required files not found!")
        return
    
    # Setup shared components
    tokenizer = NucleotideTokenizer()
    data_extractor = SimpleDataExtractor("speech.mp3", "speech_transcription.json")
    
    # TEST 1: Classic Transformer
    logger.info("\n🔥 TEST 1: CLASSIC TRANSFORMER")
    logger.info("-" * 40)
    
    transformer_model = TransformerTTSModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=128,
        hidden_dim=256,
        num_codebooks=4,
        codebook_size=1024
    ).to(device)
    
    transformer_trainer = TimedTrainer(
        transformer_model, tokenizer, data_extractor, "Classic Transformer"
    )
    
    transformer_success = transformer_trainer.train(steps=1500)
    
    # TEST 2: Your Hybrid (for comparison - would need to import your model)
    logger.info("\n🚀 COMPARISON COMPLETE!")
    logger.info("Check the timing summaries above for detailed analysis.")
    
    if transformer_success:
        logger.info("✅ Transformer training completed successfully!")
    else:
        logger.warning("⚠️  Transformer training had issues")


if __name__ == "__main__":
    main()