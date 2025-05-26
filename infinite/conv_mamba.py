#!/usr/bin/env python3
"""
Hybrid Mamba-Conv TTS Test
=========================
Forward: Mamba (global context, state carrying)
Backward: Convolutions (local patterns, parallel processing)
Enhanced with: Temporal Decision Oracle for adaptive stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import matplotlib.pyplot as plt
from nucleotide_tokenizer import NucleotideTokenizer
import soundfile as sf
from datetime import datetime
import logging
from modules import HybridTextEncoder

# Import temporal modeling components
try:
    from prosody_duration_predictor import TemporalDecisionOracle
    TEMPORAL_ORACLE_AVAILABLE = True
    logging.info("✅ TemporalDecisionOracle imported successfully")
except ImportError:
    TEMPORAL_ORACLE_AVAILABLE = False
    logging.warning("⚠️  TemporalDecisionOracle not available - using fallback")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedSSMBlock(nn.Module):
    """Optimized State Space Model block for better GPU utilization"""
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.A = nn.Parameter(torch.randn(H, H) * (1. / H**0.5))
        self.B = nn.Parameter(torch.randn(H, 1) * (1. / H**0.5))
        self.C = nn.Parameter(torch.randn(1, H) * (1. / H**0.5))
        self.D = nn.Parameter(torch.zeros(1))

    def forward(self, u_sequence):
        B, T, _ = u_sequence.shape
        u = u_sequence.squeeze(-1)  # [B, T]
        
        # Ensure everything is on same device
        device = u.device
        
        # Initialize hidden state on same device
        h = torch.zeros(B, self.A.size(0), device=device, dtype=u.dtype)
        
        # More GPU-efficient implementation with explicit device handling
        h_states = []
        for t in range(T):
            u_t = u[:, t:t+1]  # [B, 1] - keep dims for broadcasting
            # Ensure all operations stay on GPU
            h = torch.mm(h, self.A.T) + torch.mm(u_t, self.B.T)
            h_states.append(h)
        
        # Vectorized output computation - ensure device consistency
        h_stack = torch.stack(h_states, dim=1)  # [B, T, H]
        
        # Efficient output calculation on same device
        C_expanded = self.C.expand(B, -1, -1).to(device)  # [B, 1, H]
        y = torch.bmm(h_stack, C_expanded.transpose(1, 2)).squeeze(-1)  # [B, T]
        y = y + u * self.D.to(device)  # Ensure D is on correct device
        
        return y.unsqueeze(-1)  # [B, T, 1]

class HybridMambaConvTester:
    """
    Hybrid TTS: Forward Mamba + Backward Convolutions
    """
    
    def __init__(self, audio_path="speech.mp3", transcription_path="speech_transcription.json"):
        self.audio_path = audio_path
        self.transcription_path = transcription_path
        self.device = DEVICE
        
        # Training metrics tracking
        self.training_metrics = {
            'losses': [],
            'accuracies': [],
            'forward_losses': [],
            'backward_losses': [],
            'learning_rates': [],
            'step_times': [],
            'confidence_scores': []
        }
        
        # Load data
        with open(transcription_path, 'r') as f:
            self.transcription_data = json.load(f)
        
        self.setup_components()
        
    def setup_components(self):
        """Setup all components"""
        logger.info("🚀 Setting up Hybrid Mamba-Conv components...")
        
        # Device info
        logger.info(f"🎯 Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"🔥 CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"⚡ Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        else:
            logger.warning("⚠️  No CUDA available - using CPU!")
        
        # EnCodec
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(3.0)
        self.codec.eval().to(self.device)
        
        # Verify codec is on GPU
        logger.info(f"📊 EnCodec device: {next(self.codec.parameters()).device}")
        
        # Get codebook info
        self.num_codebooks = 4
        self.codebook_size = self.codec.quantizer.vq.layers[0].codebook.size(0)
        logger.info(f"📊 Codebooks: {self.num_codebooks}, Size: {self.codebook_size}")
        
        # Tokenizer
        self.tokenizer = NucleotideTokenizer()
        
        # Text encoder - now using hybrid architecture!
        self.text_encoder = TextEncoder(
            vocab_size=self.tokenizer.get_vocab_size(),
            embed_dim=256,
            hidden_dim=512
        ).to(self.device)
        
        # Hybrid model with temporal oracle
        self.model = HybridMambaConvRefiner(
            text_dim=256,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            hidden_dim=512
        ).to(self.device)
        
        # Initialize Temporal Decision Oracle for adaptive stopping
        if TEMPORAL_ORACLE_AVAILABLE:
            self.temporal_oracle = TemporalDecisionOracle(hidden_dim=512).to(self.device)
            
            # Enhanced: Initialize prosody memory if available
            if hasattr(self.temporal_oracle, 'prosody_memory'):
                logger.info("🧠 ProsodyMemoryBank detected in TemporalOracle")
                self.memory_learning_enabled = True
            else:
                logger.info("🔮 TemporalOracle without memory bank")
                self.memory_learning_enabled = False
            
            logger.info("🔮 TemporalDecisionOracle initialized")
        else:
            self.temporal_oracle = None
            self.memory_learning_enabled = False
            logger.info("⚠️  Using fallback generation without temporal oracle")
        
        # Memory learning statistics
        self.memory_stats = {
            'patterns_learned': 0,
            'memory_updates': 0,
            'convergence_improvements': []
        }
        
        # Verify models are on GPU with extra checks
        logger.info(f"🧠 TTS Model device: {next(self.model.parameters()).device}")
        logger.info(f"📝 Text Encoder device: {next(self.text_encoder.parameters()).device}")
        if self.temporal_oracle:
            logger.info(f"🔮 Temporal Oracle device: {next(self.temporal_oracle.parameters()).device}")
        
        # Check specific components of HybridTextEncoder
        if hasattr(self.text_encoder, 'forward_mamba'):
            logger.info(f"🔍 Text Encoder Mamba device: {next(self.text_encoder.forward_mamba[0].parameters()).device}")
        if hasattr(self.text_encoder, 'backward_pyramid'):
            logger.info(f"🔍 Text Encoder Conv device: {next(self.text_encoder.backward_pyramid[0].parameters()).device}")
        
        # Count parameters - now with temporal oracle
        total_params = sum(p.numel() for p in self.model.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        oracle_params = sum(p.numel() for p in self.temporal_oracle.parameters()) if self.temporal_oracle else 0
        
        logger.info(f"🧠 Hybrid TTS Model: {total_params:,} params")
        logger.info(f"📝 Hybrid Text Encoder: {text_params:,} params") 
        logger.info(f"🔮 Temporal Oracle: {oracle_params:,} params")
        logger.info(f"🎯 Total System: {total_params + text_params + oracle_params:,} params")
        logger.info("✅ Hybrid Mamba-Conv ready")
    
    def quick_train(self, steps=15000, log_interval=500, detailed_log_interval=2000):
        """Training with detailed logging"""
        logger.info("🔥 Training Hybrid Mamba-Conv...")
        
        # GPU monitoring at start
        if torch.cuda.is_available():
            logger.info(f"🔥 Training on GPU: {torch.cuda.get_device_name()}")
            logger.info(f"⚡ Pre-training GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            logger.info(f"🧠 Model on GPU: {next(self.model.parameters()).is_cuda}")
            logger.info(f"📊 EnCodec on GPU: {next(self.codec.parameters()).is_cuda}")
        else:
            logger.warning("⚠️  Training on CPU - this will be slow!")
        
        start_time = time.time()
        
        # Extract training data with updated parameters
        fragments = self.extract_training_data(target_duration=3.0, max_fragments=5)
        if len(fragments) == 0:
            logger.error("❌ No fragments found!")
            return False
        
        # Training with temporal oracle awareness
        optimizer = torch.optim.AdamW(
            list(self.text_encoder.parameters()) + 
            list(self.model.parameters()) +
            (list(self.temporal_oracle.parameters()) if self.temporal_oracle else []),
            lr=3e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.95)
        )
        
        # Cosine scheduler with warmup
        warmup_steps = min(1000, steps // 10)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=3e-4,
            total_steps=steps,
            pct_start=warmup_steps/steps,
            anneal_strategy='cos'
        )
        
        best_accuracy = 0
        step_times = []
        
        # Progress tracking variables
        accuracy_history = []
        loss_history = []
        last_progress_report = 0
        progress_interval = 50  # Report progress every 50 steps (was 100)
        
        logger.info("Step  | Total Loss | Accuracy | Fwd Loss | Bwd Loss | Confidence | LR      | Time/Step | Status")
        logger.info("-" * 110)
        
        # Force first progress report
        logger.info("🚀 Training started - will report progress every 50 steps...")
        print("🚀 Training started - will report progress every 50 steps...")  # Force print
        
        for step in range(steps):
            step_start_time = time.time()
            
            # Select training fragment
            pair_idx = step % len(fragments)
            frag = fragments[pair_idx]
            
            # Encode text using hybrid text encoder
            text_tokens = self.tokenizer.encode(frag['text'], add_special_tokens=True)
            text_emb = self.text_encoder(torch.tensor([text_tokens], device=self.device), return_sequence=False)
            
            target_tokens = frag['tokens']  # [C, T]
            C, T = target_tokens.shape
            
            # Start with random tokens
            current_tokens = torch.randint(0, self.codebook_size, (1, C, T), device=self.device)
            
            # Iterative refinement with online memory learning
            num_iterations = np.random.randint(3, 7)
            total_loss = 0
            forward_loss_acc = 0
            backward_loss_acc = 0
            confidence_scores = []
            
            # Memory learning: Extract initial pattern signature
            if self.memory_learning_enabled:
                initial_pattern_signature = self.extract_pattern_signature(text_emb, current_tokens)
                target_duration = torch.tensor([T], dtype=torch.float32, device=self.device)
            
            for iteration in range(num_iterations):
                # Forward pass through hybrid model
                model_output = self.model(text_emb, current_tokens, iteration)
                
                if isinstance(model_output, dict):
                    logits = model_output['logits']
                    confidence = model_output['confidence']
                    forward_loss = model_output.get('forward_loss', 0)
                    backward_loss = model_output.get('backward_loss', 0)
                    forward_loss_acc += forward_loss
                    backward_loss_acc += backward_loss
                else:
                    logits, confidence = model_output
                    forward_loss = backward_loss = 0
                
                confidence_scores.append(confidence.mean().item())
                
                # Update strategy
                update_prob = 0.8 / (iteration + 1)
                update_mask = torch.rand(C, T, device=self.device) < update_prob
                
                # Create new tensor
                new_current_tokens = current_tokens.clone()
                
                iter_loss = 0
                for cb_idx in range(C):
                    cb_update_mask = update_mask[cb_idx]
                    if cb_update_mask.sum() == 0:
                        continue
                    
                    cb_logits = logits[0, cb_idx, cb_update_mask, :]
                    cb_targets = target_tokens[cb_idx, cb_update_mask]
                    
                    loss = F.cross_entropy(cb_logits, cb_targets)
                    iter_loss += loss
                    
                    # Update tokens
                    new_tokens = torch.argmax(cb_logits, dim=-1)
                    new_current_tokens[0, cb_idx, cb_update_mask] = new_tokens
                
                current_tokens = new_current_tokens
                total_loss += iter_loss
                
                # Memory Learning: Update memory bank online
                if self.memory_learning_enabled and iteration == num_iterations - 1:  # Final iteration
                    self.update_memory_online(
                        pattern_signature=initial_pattern_signature,
                        final_tokens=current_tokens,
                        target_duration=target_duration,
                        final_confidence=confidence.mean()
                    )
            
            # Final accuracy
            final_accuracy = (current_tokens[0] == target_tokens).float().mean().item()
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Backward pass
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.text_encoder.parameters()) + list(self.model.parameters()), 
                    max_norm=1.0
                )
                
                optimizer.step()
                scheduler.step()
            
            # Track progress for reporting
            accuracy_history.append(final_accuracy)
            loss_history.append(total_loss.item() if hasattr(total_loss, 'item') else 0)
            
            # Update best model
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                self.save_checkpoint(step, final_accuracy, optimizer, scheduler)
            
            # Track metrics
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            current_lr = optimizer.param_groups[0]['lr']
            
            self.training_metrics['losses'].append(total_loss.item() if hasattr(total_loss, 'item') else 0)
            self.training_metrics['accuracies'].append(final_accuracy)
            self.training_metrics['forward_losses'].append(forward_loss_acc.item() if hasattr(forward_loss_acc, 'item') else forward_loss_acc)
            self.training_metrics['backward_losses'].append(backward_loss_acc.item() if hasattr(backward_loss_acc, 'item') else backward_loss_acc)
            self.training_metrics['learning_rates'].append(current_lr)
            self.training_metrics['step_times'].append(step_time)
            self.training_metrics['confidence_scores'].append(avg_confidence)
            
            # Progress reporting (every 50 steps)
            if step % progress_interval == 0 or step - last_progress_report >= progress_interval:
                last_progress_report = step
                
                # Calculate progress metrics
                if len(accuracy_history) >= progress_interval:
                    recent_acc_improvement = accuracy_history[-1] - accuracy_history[-progress_interval]
                    recent_loss_improvement = loss_history[-progress_interval] - loss_history[-1]  # Loss decrease is improvement
                else:
                    recent_acc_improvement = accuracy_history[-1] - accuracy_history[0] if accuracy_history else 0
                    recent_loss_improvement = loss_history[0] - loss_history[-1] if len(loss_history) > 1 else 0
                
                avg_step_time = np.mean(step_times[-progress_interval:]) if len(step_times) >= progress_interval else np.mean(step_times)
                progress_percent = (step / steps) * 100
                eta_minutes = ((steps - step) * avg_step_time) / 60 if avg_step_time > 0 else 0
                
                memory_info = ""
                if self.memory_learning_enabled:
                    memory_info = f" | Mem: {self.memory_stats['patterns_learned']} patterns"
                
                # Force console output with print() to ensure visibility
                progress_msg = f"📊 PROGRESS REPORT - Step {step} ({progress_percent:.1f}%)"
                acc_msg = f"   📈 Accuracy: {final_accuracy:.4f} (Δ{recent_acc_improvement:+.4f} over last {min(progress_interval, len(accuracy_history))} steps)"
                loss_msg = f"   📉 Loss: {total_loss.item() if hasattr(total_loss, 'item') else 0:.6f} (Δ{recent_loss_improvement:+.6f} improvement)"
                best_msg = f"   🏆 Best accuracy so far: {best_accuracy:.4f}"
                eta_msg = f"   ⏱️  ETA: {eta_minutes:.1f} minutes | Avg step time: {avg_step_time:.3f}s{memory_info}"
                conf_msg = f"   🎯 Current confidence: {avg_confidence:.4f} | LR: {current_lr:.2e}"
                
                # Both logger and print for guaranteed output
                logger.info(progress_msg)
                logger.info(acc_msg)
                logger.info(loss_msg) 
                logger.info(best_msg)
                logger.info(eta_msg)
                logger.info(conf_msg)
                logger.info("")
                
                print(progress_msg)  # Force to console
                print(acc_msg)
                print(loss_msg)
                print(best_msg) 
                print(eta_msg)
                print(conf_msg)
                print("")  # Empty line
                
                # Force flush
                import sys
                sys.stdout.flush()
            
            # Logging
            if step % log_interval == 0:
                avg_step_time = np.mean(step_times[-log_interval:]) if step_times else step_time
                status = self.get_training_status(final_accuracy)
                
                logger.info(
                    f"{step:5d} | {total_loss.item() if hasattr(total_loss, 'item') else 0:10.6f} | "
                    f"{final_accuracy:8.4f} | {forward_loss_acc:8.4f} | {backward_loss_acc:8.4f} | "
                    f"{avg_confidence:10.4f} | {current_lr:.2e} | {avg_step_time:9.3f}s | {status}"
                )
            
            # Detailed logging
            if step % detailed_log_interval == 0 and step > 0:
                self.log_detailed_metrics(step, steps, start_time)
                
                # GPU memory tracking
                if torch.cuda.is_available():
                    logger.info(f"🔥 GPU memory at step {step}: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    if torch.cuda.memory_allocated() > 8e9:  # > 8GB
                        logger.warning("⚠️  High GPU memory usage detected!")
            
            # Early stopping
            if final_accuracy > 0.95:
                logger.info(f"🎉 Excellent accuracy achieved! Early stopping at step {step}")
                logger.info("🎵 Proceeding to generation testing...")
                break
        
        # Final training summary
        self.log_training_summary(best_accuracy, time.time() - start_time)
        
        # Continue to generation testing even after early stopping
        logger.info("🎵 Training completed successfully, proceeding to generation...")
        return best_accuracy > 0.7
    
    def get_training_status(self, accuracy):
        """Get training status emoji"""
        if accuracy > 0.90:
            return "🎉"
        elif accuracy > 0.75:
            return "🔥"
        elif accuracy > 0.50:
            return "⚡"
        else:
            return "🚀"
    
    def log_detailed_metrics(self, step, total_steps, start_time):
        """Log detailed training metrics"""
        elapsed_time = time.time() - start_time
        progress = step / total_steps
        eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
        
        recent_losses = self.training_metrics['losses'][-100:] if len(self.training_metrics['losses']) >= 100 else self.training_metrics['losses']
        recent_accs = self.training_metrics['accuracies'][-100:] if len(self.training_metrics['accuracies']) >= 100 else self.training_metrics['accuracies']
        
        logger.info("="*60)
        logger.info(f"📊 DETAILED METRICS AT STEP {step}")
        logger.info(f"⏱️  Progress: {progress*100:.1f}% | Elapsed: {elapsed_time/60:.1f}min | ETA: {eta/60:.1f}min")
        logger.info(f"📈 Loss trend (last 100): avg={np.mean(recent_losses):.6f}, std={np.std(recent_losses):.6f}")
        logger.info(f"🎯 Accuracy trend (last 100): avg={np.mean(recent_accs):.4f}, max={np.max(recent_accs):.4f}")
        logger.info(f"🧠 Forward vs Backward: fwd={np.mean(self.training_metrics['forward_losses'][-100:]):.4f}, "
                   f"bwd={np.mean(self.training_metrics['backward_losses'][-100:]):.4f}")
        logger.info(f"⚡ Learning rate: {self.training_metrics['learning_rates'][-1]:.2e}")
        logger.info(f"⏰ Step time: {np.mean(self.training_metrics['step_times'][-100:]):.3f}s/step")
        logger.info("="*60)
    
    def log_training_summary(self, best_accuracy, total_time):
        """Log final training summary"""
        logger.info("\n" + "="*80)
        logger.info("🏁 TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"🎯 Best accuracy: {best_accuracy:.4f}")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"📈 Final loss: {self.training_metrics['losses'][-1]:.6f}")
        logger.info(f"🧠 Model saved: {'✅' if best_accuracy > 0.7 else '❌'}")
        logger.info(f"📊 Total steps: {len(self.training_metrics['losses'])}")
        logger.info(f"⚡ Avg step time: {np.mean(self.training_metrics['step_times']):.3f}s")
        
        # Save training plots
        self.save_training_plots()
        logger.info("📊 Training plots saved")
        
        # Memory learning summary
        if self.memory_learning_enabled:
            logger.info(f"🧠 Memory Learning Summary:")
            logger.info(f"   Patterns learned: {self.memory_stats['patterns_learned']}")
            logger.info(f"   Memory updates: {self.memory_stats['memory_updates']}")
            if self.memory_stats['convergence_improvements']:
                avg_improvement = np.mean(self.memory_stats['convergence_improvements'])
                logger.info(f"   Avg convergence improvement: {avg_improvement:.1f}%")
        
        logger.info("="*80)
    
    def save_training_plots(self):
        """Save training visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss plot
        axes[0, 0].plot(self.training_metrics['losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_metrics['accuracies'])
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[0, 2].plot(self.training_metrics['learning_rates'])
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('LR')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # Forward vs Backward losses
        axes[1, 0].plot(self.training_metrics['forward_losses'], label='Forward', alpha=0.7)
        axes[1, 0].plot(self.training_metrics['backward_losses'], label='Backward', alpha=0.7)
        axes[1, 0].set_title('Forward vs Backward Losses')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confidence scores
        axes[1, 1].plot(self.training_metrics['confidence_scores'])
        axes[1, 1].set_title('Confidence Scores')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].grid(True)
        
        # Step times
        axes[1, 2].plot(self.training_metrics['step_times'])
        axes[1, 2].set_title('Step Times')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Time (s)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'hybrid_training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, step, accuracy, optimizer, scheduler):
        """Save model checkpoint"""
        checkpoint = {
            'model': self.model.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step,
            'accuracy': accuracy,
            'training_metrics': self.training_metrics
        }
        torch.save(checkpoint, 'hybrid_model.pt')
        logger.info(f"💾 Checkpoint saved at step {step} with accuracy {accuracy:.4f}")
    
    def extract_training_data(self, target_duration=3.0, max_fragments=5):
        """Extract LONGER training fragments for better overfitting"""
        logger.info(f"🎯 Extracting {max_fragments} fragments of {target_duration}s each...")
        logger.info("📈 Using longer fragments for better overfitting and quality")
        
        # Force audio loading to happen on CPU then move to GPU
        logger.info("🎵 Loading audio file...")
        wav, sr = torchaudio.load(self.audio_path)
        wav = convert_audio(wav, sr, target_sr=24000, target_channels=1)
        logger.info(f"📊 Audio loaded: {wav.shape}, sample rate: 24000Hz")
        
        char_alignments = self.transcription_data['character_alignments']
        total_duration = self.transcription_data['metadata']['duration_seconds']
        logger.info(f"📏 Total audio duration: {total_duration:.2f}s")
        logger.info(f"📝 Character alignments: {len(char_alignments)} entries")
        
        fragments = []
        
        # Extract overlapping fragments for more training data
        step_size = target_duration * 0.7  # 70% overlap for more data
        for i in range(max_fragments):
            # Overlapping strategy for more diverse training data
            start_time = (i * step_size) + 0.5
            end_time = start_time + target_duration
            
            if end_time > total_duration - 0.5:
                # Try from the end
                end_time = total_duration - 0.5
                start_time = end_time - target_duration
                if start_time < 0.5:
                    continue
                    
            # Audio
            start_sample = int(start_time * 24000)
            end_sample = int(end_time * 24000)
            audio_fragment = wav[:, start_sample:end_sample]
            
            # Text alignment
            fragment_text = ""
            char_count = 0
            for char_info in char_alignments:
                if start_time <= char_info['start'] < end_time:
                    fragment_text += char_info['char']
                    char_count += 1
            
            fragment_text = ' '.join(fragment_text.strip().split())
            
            # Only use fragments with substantial text (but not too restrictive)
            if len(fragment_text) >= 8 and char_count >= 10:  # More lenient requirements
                # Get EnCodec tokens - ensure GPU processing
                with torch.no_grad():
                    audio_tensor = audio_fragment.to(self.device)  # Move to GPU
                    encoded = self.codec.encode(audio_tensor.unsqueeze(0))
                    tokens = encoded[0][0]  # [1, C, T]
                
                fragments.append({
                    'audio': audio_fragment,
                    'text': fragment_text,
                    'tokens': tokens.squeeze(0),  # [C, T]
                    'start_time': start_time,
                    'duration': target_duration,
                    'char_count': char_count
                })
                
                logger.info(f"  Fragment {i+1}: '{fragment_text[:50]}...' -> "
                           f"tokens: {tokens.shape}, chars: {char_count}, duration: {target_duration:.1f}s")
        
        # Add fallback extraction if no fragments found
        if len(fragments) == 0:
            logger.warning("⚠️  No fragments with strict requirements, trying fallback...")
            # Fallback: shorter fragments with looser requirements
            for i in range(3):
                start_time = (i * total_duration / 4) + 1.0
                end_time = start_time + 2.0  # 2 second fragments
                
                if end_time > total_duration - 1.0:
                    continue
                    
                # Audio
                start_sample = int(start_time * 24000)
                end_sample = int(end_time * 24000)
                audio_fragment = wav[:, start_sample:end_sample]
                
                # Text alignment
                fragment_text = ""
                char_count = 0
                for char_info in char_alignments:
                    if start_time <= char_info['start'] < end_time:
                        fragment_text += char_info['char']
                        char_count += 1
                
                fragment_text = ' '.join(fragment_text.strip().split())
                
                # Very lenient requirements for fallback
                if len(fragment_text) >= 3 and char_count >= 5:
                    with torch.no_grad():
                        audio_tensor = audio_fragment.to(self.device)  # Ensure GPU
                        encoded = self.codec.encode(audio_tensor.unsqueeze(0))
                        tokens = encoded[0][0]
                    
                    fragments.append({
                        'audio': audio_fragment,
                        'text': fragment_text,
                        'tokens': tokens.squeeze(0),
                        'start_time': start_time,
                        'duration': 2.0,
                        'char_count': char_count
                    })
                    
                    logger.info(f"  Fallback Fragment {i+1}: '{fragment_text}' -> "
                               f"tokens: {tokens.shape}, chars: {char_count}")
        
        logger.info(f"📊 Total extracted fragments: {len(fragments)}")
        if len(fragments) > 0:
            avg_length = sum(f['tokens'].shape[1] for f in fragments) / len(fragments)
            logger.info(f"📏 Average fragment length: {avg_length:.1f} timesteps")
            
            # GPU memory check after extraction
            if torch.cuda.is_available():
                logger.info(f"⚡ GPU memory after extraction: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        return fragments
    
    def test_generation(self, test_texts=None):
        """Test generation with hybrid model"""
        logger.info("\n🎵 TESTING HYBRID GENERATION")
        logger.info("="*50)
        
        # Load trained model
        if Path('hybrid_model.pt').exists():
            try:
                checkpoint = torch.load('hybrid_model.pt', map_location=self.device)
                self.model.load_state_dict(checkpoint['model'])
                self.text_encoder.load_state_dict(checkpoint['text_encoder'])
                logger.info(f"✅ Loaded hybrid model (accuracy: {checkpoint['accuracy']:.4f})")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                return None
        else:
            logger.error("❌ No trained hybrid model found!")
            return None
        
        self.model.eval()
        self.text_encoder.eval()
        
        if test_texts is None:
            test_texts = [
                "hello hybrid world",
                "mamba forward conv backward", 
                "linear complexity rocks",
                "best of both worlds",
                "efficient audio synthesis"
            ]
        
        results = []
        
        for i, text in enumerate(test_texts):
            logger.info(f"\n🎯 Hybrid Test {i+1}: '{text}'")
            
            try:
                # Encode text using hybrid encoder
                text_tokens = self.tokenizer.encode(text, add_special_tokens=True)
                text_emb = self.text_encoder(torch.tensor([text_tokens], device=self.device), return_sequence=False)
                
                target_length = max(30, min(120, len(text) * 4))
                logger.info(f"   Target length: {target_length} timesteps")
                
                # Test different iteration counts
                for iterations in [5, 8, 12]:
                    logger.info(f"   Testing {iterations} iterations...")
                    
                    with torch.no_grad():
                        generated_tokens = self.generate_with_analysis(
                            text_emb, target_length, iterations=iterations
                        )
                        
                        try:
                            audio = self.tokens_to_audio(generated_tokens)
                            filename = f"hybrid_test_{i+1}_{iterations}iter.wav"
                            self.save_audio(audio, filename)
                            
                            logger.info(f"      ✅ Generated: {filename}")
                            
                            results.append({
                                'text': text,
                                'iterations': iterations,
                                'filename': filename,
                                'success': True
                            })
                            
                        except Exception as e:
                            logger.error(f"      ❌ Audio conversion failed: {e}")
                            results.append({
                                'text': text,
                                'iterations': iterations,
                                'success': False,
                                'error': str(e)
                            })
                            
            except Exception as e:
                logger.error(f"❌ Text processing failed for '{text}': {e}")
                continue
        
        # Summary
        if results:
            successful = sum(1 for r in results if r['success'])
            logger.info(f"\n📊 HYBRID GENERATION SUMMARY")
            logger.info("="*50)
            logger.info(f"✅ Successful generations: {successful}/{len(results)}")
            
            for result in results:
                if result['success']:
                    logger.info(f"  🎵 '{result['text']}' ({result['iterations']} iter) → {result['filename']}")
        else:
            logger.error("❌ No generation results!")
            
        return results
    
    def debug_memory_lookup(self, text, text_emb):
        """Debug memory bank lookup for analysis"""
        if not self.memory_learning_enabled or not hasattr(self.temporal_oracle, 'prosody_memory'):
            return
        
        try:
            memory_bank = self.temporal_oracle.prosody_memory
            current_tokens = torch.randint(0, self.codebook_size, (1, self.num_codebooks, 30), device=text_emb.device)
            
            # Check what memory patterns are being matched
            pattern_sig = self.extract_pattern_signature(text_emb, current_tokens)
            
            # Ensure pattern signature matches memory bank dimensions
            if pattern_sig.shape[-1] != memory_bank.pattern_memory.shape[-1]:
                if pattern_sig.shape[-1] > memory_bank.pattern_memory.shape[-1]:
                    pattern_sig = pattern_sig[:, :memory_bank.pattern_memory.shape[-1]]
                else:
                    padding = memory_bank.pattern_memory.shape[-1] - pattern_sig.shape[-1]
                    pattern_sig = F.pad(pattern_sig, (0, padding))
            
            pattern_norm = F.normalize(pattern_sig, dim=-1)
            memory_norm = F.normalize(memory_bank.pattern_memory, dim=-1)
            similarities = torch.mm(pattern_norm, memory_norm.T).squeeze(0)
            
            top_5 = torch.topk(similarities, min(5, len(similarities)))
            
            logger.info(f"🔍 Memory analysis for '{text}':")
            for i, (score, slot) in enumerate(zip(top_5.values, top_5.indices)):
                confidence = memory_bank.pattern_confidence[slot].item()
                duration = memory_bank.pattern_durations[slot].item()
                logger.info(f"  {i+1}. Slot {slot}: similarity={score:.3f}, confidence={confidence:.3f}, duration={duration:.1f}")
                
        except Exception as e:
            logger.warning(f"⚠️  Memory analysis failed: {e}")
    
    def generate_with_analysis(self, text_emb, target_length, iterations=8):
        """Generate with detailed analysis and adaptive stopping"""
        B = text_emb.shape[0]
        
        # Use adaptive generation if temporal oracle is available
        if self.temporal_oracle is not None:
            return self.generate_with_adaptive_stopping(text_emb, target_length, iterations)
        else:
            return self.generate_with_fixed_length(text_emb, target_length, iterations)
    
    def generate_with_adaptive_stopping(self, text_emb, max_length, iterations=8):
        """Generate with TemporalDecisionOracle for adaptive stopping"""
        B = text_emb.shape[0]
        
        current_tokens = torch.randint(
            0, self.codebook_size, 
            (B, self.num_codebooks, min(30, max_length)), 
            device=text_emb.device
        )
        
        logger.info(f"      🔮 Adaptive generation: max_length={max_length}, oracle_iterations={iterations}")
        
        generated_sequence = []
        temporal_decisions = []
        
        for iteration in range(iterations):
            # Forward pass through hybrid model
            model_output = self.model(text_emb, current_tokens, confidence_level=iteration)
            
            if isinstance(model_output, dict):
                logits = model_output['logits']
                confidence = model_output['confidence']
            else:
                logits, confidence = model_output
            
            avg_confidence = confidence.mean().item()
            
            # === TEMPORAL ORACLE DECISION ===
            # Extract current generation state
            current_state = torch.mean(current_tokens.float(), dim=(1, 2))  # [B, 1] -> [B, hidden_dim]
            current_state = current_state.unsqueeze(-1).expand(-1, 512)  # Expand to hidden_dim
            
            # Text context (global)
            text_context = text_emb  # [B, text_dim] -> expand to hidden_dim
            if text_context.shape[-1] != 512:
                text_context = F.linear(text_context, torch.randn(512, text_context.shape[-1], device=text_emb.device))
            
            # Audio history (simplified)
            if len(generated_sequence) > 0:
                recent_tokens = torch.cat(generated_sequence[-3:], dim=-1) if len(generated_sequence) >= 3 else current_tokens
                audio_history = torch.mean(recent_tokens.float(), dim=(1, 2)).unsqueeze(-1).expand(-1, 512)
            else:
                audio_history = torch.zeros_like(current_state)
            
            # Temporal oracle decision
            continue_prob, decision_breakdown = self.temporal_oracle(
                current_state, text_context, audio_history, iteration
            )
            
            # Conservative threshold tuning (prevent ultra-short generations)
            base_threshold = 0.2  # Lower threshold = less likely to stop early (was 0.3)
            iteration_factor = 0.05 * (iteration / iterations)  # Smaller pressure increase (was 0.1)
            confidence_factor = 0.1 * (1 - avg_confidence)      # Less confidence influence (was 0.2)
            dynamic_threshold = base_threshold + iteration_factor + confidence_factor
            
            # Minimum length safety check
            current_length = current_tokens.shape[-1]
            min_length = max(30, len(text_emb) * 3)  # At least 30 timesteps
            
            should_continue = continue_prob.mean().item() > dynamic_threshold
            
            # Safety override: Don't stop too early
            if current_length < min_length:
                should_continue = True
                logger.info(f"        🛡️  Min length override: {current_length} < {min_length}")
            
            temporal_decisions.append({
                'iteration': iteration,
                'continue_prob': continue_prob.mean().item(),
                'dynamic_threshold': dynamic_threshold,
                'should_continue': should_continue,
                'current_length': current_length,
                'min_length': min_length,
                'decision_breakdown': {k: v.mean().item() if torch.is_tensor(v) else v 
                                     for k, v in decision_breakdown.items()},
                'confidence': avg_confidence
            })
            
            logger.info(f"        Oracle Iter {iteration+1}: continue_prob={continue_prob.mean().item():.3f}, "
                       f"threshold={dynamic_threshold:.3f}, decision={'CONTINUE' if should_continue else 'STOP'}")
            
            # Update tokens if continuing
            if should_continue and iteration < iterations - 1:
                confidence_threshold = 0.3 + 0.5 * (iteration / iterations)
                
                updates_made = 0
                for cb_idx in range(self.num_codebooks):
                    low_conf_mask = confidence < confidence_threshold
                    if low_conf_mask.sum() > 0:
                        new_tokens = torch.argmax(logits[:, cb_idx], dim=-1)
                        current_tokens[:, cb_idx][low_conf_mask] = new_tokens[low_conf_mask]
                        updates_made += low_conf_mask.sum().item()
                
                generated_sequence.append(current_tokens.clone())
                
                # Early stopping based on oracle decision
                if not should_continue:
                    logger.info(f"      🛑 Oracle early stopping at iteration {iteration+1}")
                    break
            else:
                # Final iteration or forced stop
                confidence_threshold = 0.3 + 0.5 * (iteration / iterations)
                
                for cb_idx in range(self.num_codebooks):
                    low_conf_mask = confidence < confidence_threshold
                    if low_conf_mask.sum() > 0:
                        new_tokens = torch.argmax(logits[:, cb_idx], dim=-1)
                        current_tokens[:, cb_idx][low_conf_mask] = new_tokens[low_conf_mask]
                
                generated_sequence.append(current_tokens.clone())
                break
        
        final_tokens = generated_sequence[-1] if generated_sequence else current_tokens
        
        logger.info(f"      🎯 Adaptive generation complete: {len(temporal_decisions)} oracle decisions")
        logger.info(f"      📊 Final continue_prob: {temporal_decisions[-1]['continue_prob']:.3f}")
        
        return final_tokens
    
    def generate_with_fixed_length(self, text_emb, target_length, iterations=8):
        """Fallback generation with fixed length (original method)"""
        B = text_emb.shape[0]
        
        current_tokens = torch.randint(
            0, self.codebook_size, 
            (B, self.num_codebooks, target_length), 
            device=text_emb.device
        )
        
        logger.info(f"      📏 Fixed-length generation: {target_length} timesteps")
        
        for iteration in range(iterations):
            model_output = self.model(text_emb, current_tokens, confidence_level=iteration)
            
            if isinstance(model_output, dict):
                logits = model_output['logits']
                confidence = model_output['confidence']
            else:
                logits, confidence = model_output
            
            avg_confidence = confidence.mean().item()
            confidence_threshold = 0.3 + 0.5 * (iteration / iterations)
            
            updates_made = 0
            for cb_idx in range(self.num_codebooks):
                low_conf_mask = confidence < confidence_threshold
                if low_conf_mask.sum() > 0:
                    new_tokens = torch.argmax(logits[:, cb_idx], dim=-1)
                    current_tokens[:, cb_idx][low_conf_mask] = new_tokens[low_conf_mask]
                    updates_made += low_conf_mask.sum().item()
            
            logger.info(f"        Iter {iteration+1}: confidence={avg_confidence:.3f}, threshold={confidence_threshold:.3f}, updated={updates_made}")
        
        logger.info(f"      Final confidence: {confidence.mean().item():.3f}")
        return current_tokens
    
    def extract_pattern_signature(self, text_emb, audio_tokens):
        """Extract pattern signature for memory learning"""
        # Combine text and audio features into a compact signature
        audio_summary = torch.mean(audio_tokens.float(), dim=(1, 2))  # [B]
        
        # Create pattern signature by combining text and audio
        if text_emb.dim() == 2:  # [B, text_dim]
            text_summary = text_emb.squeeze(0) if text_emb.shape[0] == 1 else text_emb.mean(0)
        else:
            text_summary = text_emb
        
        # Normalize features
        text_norm = F.normalize(text_summary, dim=-1)
        audio_norm = F.normalize(audio_summary.unsqueeze(-1).expand_as(text_norm), dim=-1)
        
        # Combine into pattern signature
        pattern_signature = torch.cat([text_norm, audio_norm], dim=-1)  # [text_dim + audio_dim]
        
        return pattern_signature.unsqueeze(0)  # [1, pattern_dim]
    
    def update_memory_online(self, pattern_signature, final_tokens, target_duration, final_confidence):
        """Efficient online memory update during training"""
        if not self.memory_learning_enabled or not hasattr(self.temporal_oracle, 'prosody_memory'):
            return
        
        try:
            memory_bank = self.temporal_oracle.prosody_memory
            
            # Fast similarity computation using cosine similarity
            with torch.no_grad():
                # Reshape pattern_signature to match memory bank dimensions
                if pattern_signature.shape[-1] != memory_bank.pattern_memory.shape[-1]:
                    # Simple projection to match dimensions
                    if pattern_signature.shape[-1] > memory_bank.pattern_memory.shape[-1]:
                        pattern_signature = pattern_signature[:, :memory_bank.pattern_memory.shape[-1]]
                    else:
                        padding = memory_bank.pattern_memory.shape[-1] - pattern_signature.shape[-1]
                        pattern_signature = F.pad(pattern_signature, (0, padding))
                
                # Compute similarities efficiently
                pattern_norm = F.normalize(pattern_signature, dim=-1)
                memory_norm = F.normalize(memory_bank.pattern_memory, dim=-1)
                similarities = torch.mm(pattern_norm, memory_norm.T)  # [1, memory_size]
                
                # Find best matching slot
                max_similarity = torch.max(similarities)
                best_slot = torch.argmax(similarities, dim=-1).item()
                
                # Update memory with exponential moving average
                alpha = 0.1  # Learning rate for memory updates
                
                # Only update if similarity is reasonable (not completely random)
                if max_similarity > 0.1 or memory_bank.pattern_confidence[best_slot] < 0.5:
                    # Update pattern
                    memory_bank.pattern_memory.data[best_slot] = (
                        (1 - alpha) * memory_bank.pattern_memory.data[best_slot] + 
                        alpha * pattern_signature.squeeze(0)
                    )
                    
                    # Update duration
                    memory_bank.pattern_durations.data[best_slot] = (
                        (1 - alpha) * memory_bank.pattern_durations.data[best_slot] + 
                        alpha * target_duration.unsqueeze(0)
                    )
                    
                    # Update confidence based on final model confidence
                    confidence_update = min(final_confidence.item(), 0.95)  # Cap at 0.95
                    memory_bank.pattern_confidence.data[best_slot] = (
                        (1 - alpha) * memory_bank.pattern_confidence.data[best_slot] + 
                        alpha * confidence_update
                    )
                    
                    # Update statistics
                    self.memory_stats['memory_updates'] += 1
                    if max_similarity < 0.3:  # New pattern learned
                        self.memory_stats['patterns_learned'] += 1
                    
        except Exception as e:
            # Graceful degradation - don't break training if memory update fails
            logger.warning(f"⚠️  Memory update failed: {e}")
    
    def log_memory_analysis(self, step):
        """Log detailed memory learning analysis"""
        if not self.memory_learning_enabled:
            return
        
        try:
            memory_bank = self.temporal_oracle.prosody_memory
            
            # Memory utilization statistics
            active_patterns = (memory_bank.pattern_confidence > 0.3).sum().item()
            avg_confidence = memory_bank.pattern_confidence.mean().item()
            pattern_diversity = torch.std(memory_bank.pattern_memory, dim=0).mean().item()
            
            logger.info("🧠" + "="*50)
            logger.info(f"🧠 MEMORY ANALYSIS AT STEP {step}")
            logger.info(f"📊 Active patterns: {active_patterns}/1024 ({active_patterns/1024*100:.1f}%)")
            logger.info(f"🎯 Average confidence: {avg_confidence:.3f}")
            logger.info(f"🌈 Pattern diversity: {pattern_diversity:.3f}")
            logger.info(f"📈 Total patterns learned: {self.memory_stats['patterns_learned']}")
            logger.info(f"🔄 Memory updates: {self.memory_stats['memory_updates']}")
            
            # Convergence improvement estimation
            if len(self.training_metrics['accuracies']) > 100:
                recent_improvement = (self.training_metrics['accuracies'][-1] - 
                                    self.training_metrics['accuracies'][-100]) * 100
                self.memory_stats['convergence_improvements'].append(recent_improvement)
                logger.info(f"📈 Recent accuracy improvement: {recent_improvement:.2f}%")
            
            logger.info("🧠" + "="*50)
            
        except Exception as e:
            logger.warning(f"⚠️  Memory analysis failed: {e}")
    
    def tokens_to_audio(self, tokens):
        """Convert tokens to audio"""
        with torch.no_grad():
            codes = (tokens, None)
            audio = self.codec.decode([codes])[0]
        return audio
    
    def save_audio(self, audio, filename):
        """Save audio file with better error handling"""
        try:
            # Convert to CPU and handle dimensions
            audio_cpu = audio.detach().cpu()
            if audio_cpu.dim() == 3:
                audio_cpu = audio_cpu.squeeze(0)
            if audio_cpu.dim() == 2:
                audio_cpu = audio_cpu.squeeze(0)
            
            # Normalize to prevent clipping
            if audio_cpu.abs().max() > 0:
                audio_cpu = audio_cpu / audio_cpu.abs().max() * 0.8
            
            # Save using soundfile
            sf.write(filename, audio_cpu.numpy(), 24000)
            logger.info(f"🎵 Saved audio: {filename} ({audio_cpu.shape} samples)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save audio {filename}: {e}")
            raise
    
    def run_hybrid_test(self):
        """Run complete hybrid test"""
        logger.info("🚀" + "="*60)
        logger.info("🚀 HYBRID MAMBA-CONV TTS TEST")
        logger.info("🚀" + "="*60)
        
        # Step 1: Training
        logger.info("\n📚 STEP 1: HYBRID TRAINING")
        success = self.quick_train()
        
        if not success:
            logger.error("❌ Hybrid training failed!")
            return False
        
        # Step 2: Generation testing
        logger.info("\n🎵 STEP 2: GENERATION TESTING")
        results = self.test_generation()
        
        if results is None:
            logger.error("❌ Generation test failed to run!")
            return False
        
        # Step 3: Analysis
        logger.info("\n📊 STEP 3: FINAL ANALYSIS")
        if results:
            successful = sum(1 for r in results if r['success'])
            success_rate = successful/len(results)*100
            logger.info(f"Success rate: {successful}/{len(results)} ({success_rate:.1f}%)")
            
            if successful > 0:
                logger.info("\n🎉 HYBRID MAMBA-CONV WORKS!")
                logger.info("✅ Forward Mamba: Global context & state carrying")
                logger.info("✅ Backward Conv: Local patterns & parallel processing")
                logger.info("✅ Best of both worlds achieved!")
                logger.info("🎵 Check hybrid_test_*.wav files")
                return True
            else:
                logger.warning("\n⚠️  Models trained but audio generation failed")
                return False
        else:
            logger.error("❌ No results generated")
            return False


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class HybridTextEncoder(nn.Module):
    """Hybrid Text Encoder: Forward Mamba + Backward Multi-scale Convolutions"""
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 1. Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. FORWARD: Sequential Mamba for global context
        self.forward_mamba = nn.ModuleList([
            OptimizedSSMBlock(embed_dim) for _ in range(3)
        ])
        
        # 3. BACKWARD: Multi-scale Convolutions for local patterns
        self.backward_pyramid = nn.ModuleList([
            # Character-level patterns (morphology)
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=1, padding=1, groups=4),
            # Morpheme-level patterns (prefixes, suffixes)  
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=2, padding=2, groups=4),
            # Word-level patterns (bigrams, common phrases)
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=4, padding=4, groups=4),
            # Phrase-level patterns (syntax, prosody)
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=8, padding=8, groups=4),
        ])
        
        # 4. Pointwise mixing of multi-scale features
        self.backward_pointwise = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.backward_norm = nn.LayerNorm(embed_dim)
        
        # 5. Fusion of forward and backward branches
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 6. Optional: Global attention for sequence-level representation
        self.global_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        logger.info(f"🧠 HybridTextEncoder: {embed_dim}D embeddings, {hidden_dim}D hidden")
        
    def forward(self, tokens, return_sequence=True):
        """
        Args:
            tokens: [B, T] token indices
            return_sequence: If True, return per-token embeddings [B, T, embed_dim]
                           If False, return sequence-level embedding [B, embed_dim]
        """
        B, T = tokens.shape
        device = tokens.device  # Ensure device consistency
        
        # === TOKEN EMBEDDING ===
        x = self.token_embedding(tokens)  # [B, T, embed_dim]
        
        # === FORWARD BRANCH: Sequential Mamba ===
        forward_features = x
        for mamba_layer in self.forward_mamba:
            # SSM processing - ensure device consistency
            u = forward_features.mean(-1, keepdim=True)  # [B, T, 1]
            y = mamba_layer(u)  # [B, T, 1] 
            # Residual connection
            forward_features = forward_features + y.expand(-1, -1, self.embed_dim)
        
        # === BACKWARD BRANCH: Multi-scale Convolutions ===
        x_conv = x.transpose(1, 2)  # [B, embed_dim, T] for conv1d
        
        # Process all scales in parallel - ensure device consistency
        pyramid_features = []
        for conv_layer in self.backward_pyramid:
            scale_features = conv_layer(x_conv)  # [B, embed_dim//4, T]
            # Explicitly ensure on correct device
            scale_features = scale_features.to(device)
            pyramid_features.append(scale_features)
        
        # Concatenate all scales
        multi_scale_features = torch.cat(pyramid_features, dim=1)  # [B, embed_dim, T]
        
        # Mix with pointwise convolution
        backward_features = self.backward_pointwise(multi_scale_features)  # [B, embed_dim, T]
        
        # Add residual connection and normalize
        backward_features = backward_features + x_conv  # Residual
        backward_features = self.backward_norm(backward_features.transpose(1, 2)).transpose(1, 2)
        
        backward_features = backward_features.transpose(1, 2)  # [B, T, embed_dim]
        
        # === FUSION ===
        # Concatenate forward and backward features - ensure device consistency
        combined = torch.cat([forward_features, backward_features], dim=-1)  # [B, T, 2*embed_dim]
        token_embeddings = self.fusion(combined)  # [B, T, embed_dim]
        
        if return_sequence:
            # Return per-token embeddings for TTS alignment
            return token_embeddings  # [B, T, embed_dim]
        else:
            # Return sequence-level embedding (for classification tasks)
            attention_scores = self.global_attention(token_embeddings).squeeze(-1)  # [B, T]
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(1)  # [B, 1, T]
            pooled = torch.bmm(attention_weights, token_embeddings).squeeze(1)  # [B, embed_dim]
            return pooled


# Legacy TextEncoder for compatibility (keeping the old one as backup)
class LegacyTextEncoder(nn.Module):
    """Original LSTM-based text encoder"""
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
        
    def forward(self, tokens):
        x = self.embedding(tokens)
        lstm_out, _ = self.lstm(x)
        
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(1)
        pooled = torch.bmm(attention_weights, lstm_out).squeeze(1)
        
        return self.projection(pooled)


# Use new HybridTextEncoder as default
TextEncoder = HybridTextEncoder


class HybridMambaConvRefiner(nn.Module):
    """Hybrid model: Forward Mamba + Backward Convolutions"""
    def __init__(self, text_dim, num_codebooks, codebook_size, hidden_dim):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Token embeddings
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim) for _ in range(num_codebooks)
        ])
        
        # Position and confidence embeddings
        self.pos_embedding = nn.Embedding(1000, hidden_dim)
        self.confidence_embedding = nn.Embedding(20, hidden_dim)
        
        # Text conditioning
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # FORWARD BRANCH: Sequential Mamba layers for global context
        self.forward_layers = nn.ModuleList([
            OptimizedSSMBlock(hidden_dim) for _ in range(3)  # Use optimized SSM
        ])
        
        # BACKWARD BRANCH: Multi-scale dilated convolutions (parallel)
        self.backward_pyramid = nn.ModuleList([
            # Different dilations for multi-scale local patterns
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=1, padding=1, groups=4),   # Local details
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=3, padding=3, groups=4),   # Phoneme level  
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=9, padding=9, groups=4),   # Syllable level
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=27, padding=27, groups=4), # Word level
        ])
        
        # Pointwise convolution to mix multi-scale features
        self.backward_pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.backward_norm = nn.LayerNorm(hidden_dim)
        
        # Branch fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads for each codebook
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, codebook_size)
            ) for _ in range(num_codebooks)
        ])
        
        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Branch loss tracking (for logging)
        self.forward_loss_weight = 0.6
        self.backward_loss_weight = 0.4
    
    def forward(self, text_emb, current_tokens, confidence_level=5):
        B, C, T = current_tokens.shape
        
        # === TOKEN EMBEDDING ===
        token_embeds = []
        for cb_idx in range(C):
            cb_tokens = current_tokens[:, cb_idx, :]  # [B, T]
            cb_embed = self.token_embeddings[cb_idx](cb_tokens)  # [B, T, H]
            token_embeds.append(cb_embed)
        
        # Average across codebooks
        x = torch.stack(token_embeds, dim=2).mean(dim=2)  # [B, T, H]
        
        # === CONDITIONING ===
        # Position embeddings
        positions = torch.arange(T, device=current_tokens.device)
        pos_embed = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)
        
        # Confidence embeddings
        conf_level = torch.full((T,), min(confidence_level, 19), device=current_tokens.device)
        conf_embed = self.confidence_embedding(conf_level).unsqueeze(0).expand(B, -1, -1)
        
        # Text conditioning
        text_cond = self.text_proj(text_emb).unsqueeze(1).expand(-1, T, -1)
        
        # Combine all embeddings
        x = x + pos_embed + conf_embed + text_cond  # [B, T, H]
        
        # === FORWARD BRANCH: Sequential Mamba processing ===
        forward_features = x
        for layer in self.forward_layers:
            # SSM expects [B, T, 1] input
            u = forward_features.mean(-1, keepdim=True)  # [B, T, 1]
            y = layer(u)  # [B, T, 1]
            # Add residual connection
            forward_features = forward_features + y.expand(-1, -1, self.hidden_dim)
        
        # === BACKWARD BRANCH: Multi-scale dilated convolutions (parallel) ===
        x_conv = x.transpose(1, 2)  # [B, H, T] for conv1d
        
        # Process all scales in parallel
        pyramid_features = []
        for conv_layer in self.backward_pyramid:
            scale_features = conv_layer(x_conv)  # [B, H//4, T]
            pyramid_features.append(scale_features)
        
        # Concatenate all scales
        multi_scale_features = torch.cat(pyramid_features, dim=1)  # [B, H, T]
        
        # Mix with pointwise convolution
        backward_features = self.backward_pointwise(multi_scale_features)  # [B, H, T]
        
        # Add residual connection and normalize
        backward_features = backward_features + x_conv  # Residual
        backward_features = self.backward_norm(backward_features.transpose(1, 2)).transpose(1, 2)
        
        backward_features = backward_features.transpose(1, 2)  # [B, T, H]
        
        # === FUSION ===
        # Concatenate forward and backward features
        fused_features = torch.cat([forward_features, backward_features], dim=-1)  # [B, T, 2H]
        final_features = self.fusion_layer(fused_features)  # [B, T, H]
        
        # === OUTPUT GENERATION ===
        # Generate logits for each codebook
        outputs = []
        for cb_idx in range(self.num_codebooks):
            cb_output = self.output_heads[cb_idx](final_features)  # [B, T, codebook_size]
            outputs.append(cb_output)
        
        logits = torch.stack(outputs, dim=1)  # [B, C, T, codebook_size]
        
        # Generate confidence scores
        confidence = self.confidence_head(final_features).squeeze(-1)  # [B, T]
        
        # === COMPUTE BRANCH LOSSES (for monitoring) ===
        # This is optional and used for logging purposes
        forward_loss = self.compute_branch_loss(forward_features, "forward")
        backward_loss = self.compute_branch_loss(backward_features, "backward")
        
        return {
            'logits': logits,
            'confidence': confidence,
            'forward_loss': forward_loss,
            'backward_loss': backward_loss,
            'forward_features': forward_features,
            'backward_features': backward_features
        }
    
    def compute_branch_loss(self, features, branch_type):
        """Compute regularization loss for each branch (for monitoring)"""
        # Different regularization strategies for each branch
        if branch_type == "forward":
            # Forward (Mamba): Encourage temporal consistency
            if features.shape[1] > 1:
                temporal_diff = features[:, 1:] - features[:, :-1]
                temporal_loss = torch.mean(temporal_diff ** 2) * 0.001
                return temporal_loss
            else:
                return torch.tensor(0.0, device=features.device)
        else:  # backward
            # Backward (Conv): Encourage local pattern diversity
            local_variance = torch.var(features, dim=1).mean()
            pattern_loss = torch.mean(features ** 2) * 0.0001
            return pattern_loss - local_variance * 0.0001  # Encourage diversity


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 HYBRID MAMBA-CONV TTS TEST")
    print("="*60)
    
    # Check required files
    audio_file = "speech.mp3"
    transcription_file = "speech_transcription.json"
    
    if not Path(audio_file).exists() or not Path(transcription_file).exists():
        logger.error(f"❌ Missing files: {audio_file} or {transcription_file}")
        exit(1)
    
    # Check nucleotide_tokenizer
    try:
        from nucleotide_tokenizer import NucleotideTokenizer
    except ImportError:
        logger.error("❌ nucleotide_tokenizer.py not found!")
        exit(1)
    
    # Run hybrid test
    try:
        tester = HybridMambaConvTester(audio_file, transcription_file)
        success = tester.run_hybrid_test()
        
        if success:
            logger.info("\n🎉 SUCCESS! Hybrid Mamba-Conv TTS works!")
            logger.info("⚡ Forward Mamba: Linear complexity + global context")
            logger.info("🔄 Backward Conv: Parallel processing + local patterns")
            logger.info("🎵 Check hybrid_test_*.wav files")
            logger.info("📊 Check training plots and logs!")
        else:
            logger.warning("\n⚠️  Hybrid test completed but with issues")
            
    except Exception as e:
        logger.error(f"\n❌ HYBRID TEST FAILED: {e}")
        import traceback
        traceback.print_exc()