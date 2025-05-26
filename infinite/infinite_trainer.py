#!/usr/bin/env python3
"""
Infinite Trainer - Continuous Training z Virtual Checkpoints
==========================================================
Training system dla infinite continuous processing
Key features:
- Trenuje całe minutowe batches naraz
- Virtual checkpoints co 10s dla error tracking
- Fresh state tylko między batches
- No internal chunking - true continuous learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import infinite modules
try:
    from infinite_modules import (
        InfiniteMambaTextEncoder,
        InfiniteMambaAudioProcessor, 
        InfiniteDurationRegulator,
        InfiniteAudioStyleExtractor
    )
    from infinite_data_loader import InfiniteDataLoader
    from nucleotide_tokenizer import NucleotideTokenizer
    logger.info("✅ Infinite modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    exit(1)


class InfiniteTTSModel(nn.Module):
    """
    Complete TTS model for infinite continuous processing
    Processes entire minutowe batches without internal breaks
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_codebooks=4, codebook_size=1024, state_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.state_size = state_size
        
        # Infinite continuous components
        self.text_encoder = InfiniteMambaTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=4,
            state_size=state_size
        )
        
        self.duration_regulator = InfiniteDurationRegulator(
            text_dim=embed_dim,
            style_dim=64,
            hidden_dim=128,
            tokens_per_second=75.0,
            state_size=state_size
        )
        
        self.audio_processor = InfiniteMambaAudioProcessor(
            hidden_dim=hidden_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            num_layers=3,
            state_size=state_size
        )
        
        self.style_extractor = InfiniteAudioStyleExtractor(
            audio_dim=hidden_dim,
            style_dim=64
        )
        
        # Projections
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        self.default_style = nn.Parameter(torch.randn(64) * 0.01)
        
        logger.info(f"🎯 InfiniteTTSModel: {sum(p.numel() for p in self.parameters()):,} parameters")
        logger.info(f"   🔄 Infinite continuous processing")
        logger.info(f"   📍 Virtual checkpoints for error tracking")
    
    def reset_infinite_states(self, batch_size=1):
        """Reset all infinite states for new batch - ONLY place state resets!"""
        self.text_encoder.reset_infinite_state(batch_size)
        self.duration_regulator.reset_infinite_state(batch_size)
        self.audio_processor.reset_infinite_state(batch_size)
        logger.debug(f"🔄 All infinite states reset for batch_size={batch_size}")
        
    
    def forward(self, text_tokens, audio_tokens=None, reset_states=False):
        """NAPRAWIONY forward pass"""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Reset infinite states if requested
        if reset_states:
            self.reset_infinite_states(batch_size)
        
        # NAPRAWKA: Text encoding - bez return_sequence
        text_features = self.text_encoder(text_tokens, reset_state=reset_states)
        
        # NAPRAWKA: Global text context - użyj pooling
        if text_features.dim() == 3:  # [B, T, D]
            text_context = torch.mean(text_features, dim=1)  # [B, D]
        else:  # [B, D]
            text_context = text_features
        
        text_context = self.text_proj(text_context)
        
        # Style extraction
        if audio_tokens is not None:
            with torch.no_grad():
                B, C, T = audio_tokens.shape
                audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
                pseudo_audio = audio_mean.expand(B, self.hidden_dim, min(T, 200))
                style_embedding = self.style_extractor(pseudo_audio)
        else:
            style_embedding = self.default_style.unsqueeze(0).expand(batch_size, -1)
        
        # Duration regulation
        regulated_features, predicted_durations, duration_tokens, duration_confidence = \
            self.duration_regulator(text_features, style_embedding, reset_state=reset_states)
        
        # Audio processing
        if audio_tokens is not None:
            if regulated_features.dim() == 3:
                regulated_context = torch.mean(regulated_features, dim=1)
            else:
                regulated_context = regulated_features
            
            regulated_context = self.text_proj(regulated_context)
            audio_logits = self.audio_processor(audio_tokens, regulated_context, reset_state=reset_states)
        else:
            audio_logits = None
        
        return {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'duration_tokens': duration_tokens,
            'duration_confidence': duration_confidence,
            'text_features': text_features,
            'regulated_features': regulated_features,
            'style_embedding': style_embedding,
            'infinite_processing': True
        }


class InfiniteTrainer:
    """
    Trainer for infinite continuous processing with virtual checkpoints
    """
    def __init__(self, model, tokenizer, data_loader):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        
        logger.info(f"🎯 InfiniteTrainer initialized")
        logger.info(f"   Data: {len(data_loader)} continuous batches")
        logger.info(f"   🔄 Infinite continuous processing")
        logger.info(f"   📍 Virtual checkpoint training")
    
    def compute_virtual_checkpoint_losses(self, model_output, batch_data):
        """
        Compute losses at virtual checkpoints bez breaking continuous flow
        """
        logits = model_output.get('logits')
        predicted_durations = model_output.get('predicted_durations')
        duration_confidence = model_output.get('duration_confidence')
        
        text_tokens = batch_data['text_tokens']
        audio_codes = batch_data['audio_codes']
        virtual_checkpoints = batch_data.get('virtual_checkpoints', [])
        
        if not virtual_checkpoints or logits is None:
            # Fallback to standard loss computation
            from losses import compute_combined_loss
            return compute_combined_loss(model_output, batch_data, text_tokens, self.device)
        
        # Virtual checkpoint loss computation
        checkpoint_losses = []
        total_loss = 0
        total_accuracy = 0
        total_duration_accuracy = 0
        
        B, C, T_logits, V = logits.shape
        _, C_audio, T_audio = audio_codes.shape
        
        for i, checkpoint in enumerate(virtual_checkpoints):
            try:
                start_token = checkpoint['start_token']
                end_token = checkpoint['end_token']
                
                # Ensure valid range
                start_token = max(0, min(start_token, T_logits - 1))
                end_token = max(start_token + 1, min(end_token, T_logits))
                
                if end_token <= start_token:
                    continue
                
                # Extract checkpoint window
                checkpoint_logits = logits[:, :, start_token:end_token, :]  # [B, C, T_window, V]
                
                # Audio targets for this checkpoint
                audio_start = min(start_token, T_audio - 1)
                audio_end = min(end_token, T_audio)
                
                if audio_end <= audio_start:
                    continue
                
                checkpoint_audio = audio_codes[:, :, audio_start:audio_end]  # [B, C, T_window]
                
                # Compute checkpoint loss
                checkpoint_loss = self._compute_checkpoint_loss(
                    checkpoint_logits, checkpoint_audio, checkpoint
                )
                
                if checkpoint_loss is not None:
                    checkpoint_losses.append({
                        'checkpoint_idx': i,
                        'time_start': checkpoint['time_start'],
                        'time_end': checkpoint['time_end'],
                        'loss': checkpoint_loss['loss'],
                        'accuracy': checkpoint_loss['accuracy'],
                        'token_count': end_token - start_token
                    })
                    
                    total_loss += checkpoint_loss['loss']
                    total_accuracy += checkpoint_loss['accuracy']
                    
            except Exception as e:
                logger.debug(f"Checkpoint {i} computation failed: {e}")
                continue
        
        # Duration loss (całego batch)
        duration_loss = self._compute_duration_loss(
            predicted_durations, batch_data, text_tokens
        )
        
        # Confidence loss
        confidence_loss = self._compute_confidence_loss(
            duration_confidence, predicted_durations, batch_data
        )
        
        # Combine losses
        if len(checkpoint_losses) > 0:
            avg_checkpoint_loss = total_loss / len(checkpoint_losses)
            avg_accuracy = total_accuracy / len(checkpoint_losses)
            
            # Duration accuracy
            duration_accuracy = self._compute_duration_accuracy(
                predicted_durations, batch_data
            )
            
            combined_loss = (
                1.0 * avg_checkpoint_loss +    # Audio loss from checkpoints
                8.0 * duration_loss +          # Duration loss
                1.0 * confidence_loss          # Confidence loss
            )
            
            return {
                'total_loss': combined_loss,
                'checkpoint_loss': avg_checkpoint_loss,
                'duration_loss': duration_loss,
                'confidence_loss': confidence_loss,
                'accuracy': avg_accuracy,
                'duration_accuracy': duration_accuracy,
                'checkpoint_losses': checkpoint_losses,
                'num_checkpoints': len(checkpoint_losses),
                'virtual_checkpoint_training': True
            }
        else:
            # Fallback if no valid checkpoints
            logger.warning("⚠️  No valid virtual checkpoints, using fallback loss")
            from losses import compute_combined_loss
            return compute_combined_loss(model_output, batch_data, text_tokens, self.device)
    
    def _compute_checkpoint_loss(self, checkpoint_logits, checkpoint_audio, checkpoint_info):
        """Compute loss for single virtual checkpoint"""
        try:
            B, C, T_window, V = checkpoint_logits.shape
            _, C_audio, T_audio = checkpoint_audio.shape
            
            if T_audio <= 1:
                return None
            
            # Teacher forcing: predict next token
            input_tokens = checkpoint_audio[:, :, :-1]   # [B, C, T-1]
            target_tokens = checkpoint_audio[:, :, 1:]   # [B, C, T-1]
            
            min_C = min(C, C_audio)
            min_T = min(T_window, input_tokens.shape[-1], target_tokens.shape[-1])
            
            if min_T <= 0:
                return None
            
            total_loss = 0
            total_accuracy = 0
            valid_codebooks = 0
            
            for cb_idx in range(min_C):
                try:
                    pred_logits = checkpoint_logits[:, cb_idx, :min_T, :]  # [B, min_T, V]
                    target_cb = target_tokens[:, cb_idx, :min_T]           # [B, min_T]
                    
                    # Clamp targets
                    target_cb = torch.clamp(target_cb, 0, V - 1)
                    
                    # Cross entropy loss
                    cb_loss = F.cross_entropy(
                        pred_logits.reshape(-1, V),
                        target_cb.reshape(-1),
                        reduction='mean'
                    )
                    
                    # Accuracy
                    pred_tokens = torch.argmax(pred_logits, dim=-1)
                    cb_accuracy = (pred_tokens == target_cb).float().mean().item()
                    
                    if not (torch.isnan(cb_loss) or torch.isinf(cb_loss)):
                        total_loss += cb_loss
                        total_accuracy += cb_accuracy
                        valid_codebooks += 1
                
                except Exception as e:
                    logger.debug(f"Codebook {cb_idx} checkpoint loss failed: {e}")
                    continue
            
            if valid_codebooks > 0:
                return {
                    'loss': total_loss / valid_codebooks,
                    'accuracy': total_accuracy / valid_codebooks
                }
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Checkpoint loss computation failed: {e}")
            return None
    
    def _compute_duration_loss(self, predicted_durations, batch_data, text_tokens):
        """Compute duration loss for entire batch"""
        try:
            if predicted_durations is None:
                return torch.tensor(0.5, device=self.device, requires_grad=True)
            
            # Target duration based on batch info
            batch_duration = batch_data.get('duration', 60.0)
            text_length = text_tokens.shape[1]
            
            if text_length <= 0:
                return torch.tensor(0.5, device=self.device, requires_grad=True)
            
            # Create duration targets
            avg_duration_per_token = batch_duration / text_length
            avg_duration_per_token = max(0.05, min(0.3, avg_duration_per_token))
            
            target_durations = torch.full_like(predicted_durations, avg_duration_per_token)
            
            # L1 loss for duration
            duration_loss = F.l1_loss(predicted_durations, target_durations) * 5.0
            
            # Regularization
            duration_reg = (
                torch.mean(torch.relu(predicted_durations - 0.3)) +  # Penalty for > 0.3s
                torch.mean(torch.relu(0.05 - predicted_durations))   # Penalty for < 0.05s
            )
            
            total_duration_loss = duration_loss + duration_reg
            
            if torch.isnan(total_duration_loss) or torch.isinf(total_duration_loss):
                return torch.tensor(0.5, device=self.device, requires_grad=True)
            
            return total_duration_loss
            
        except Exception as e:
            logger.debug(f"Duration loss computation failed: {e}")
            return torch.tensor(0.5, device=self.device, requires_grad=True)
    
    def _compute_confidence_loss(self, confidence, predicted_durations, batch_data):
        """Compute confidence loss"""
        try:
            if confidence is None or predicted_durations is None:
                return torch.tensor(0.1, device=self.device, requires_grad=True)
            
            # High confidence when predictions are reasonable
            reasonable_mask = (predicted_durations >= 0.05) & (predicted_durations <= 0.3)
            confidence_target = reasonable_mask.float() * 0.8 + (~reasonable_mask).float() * 0.2
            
            conf_loss = F.mse_loss(confidence, confidence_target)
            
            if torch.isnan(conf_loss) or torch.isinf(conf_loss):
                return torch.tensor(0.1, device=self.device, requires_grad=True)
            
            return conf_loss
            
        except Exception as e:
            logger.debug(f"Confidence loss computation failed: {e}")
            return torch.tensor(0.1, device=self.device, requires_grad=True)
    
    def _compute_duration_accuracy(self, predicted_durations, batch_data):
        """Compute duration accuracy"""
        try:
            if predicted_durations is None:
                return 0.0
            
            batch_duration = batch_data.get('duration', 60.0)
            predicted_total = predicted_durations.sum().item()
            
            if batch_duration <= 0:
                return 0.0
            
            # Accuracy if within 50% of target
            ratio = predicted_total / batch_duration
            accuracy = 1.0 if 0.5 <= ratio <= 1.5 else max(0.0, 1.0 - abs(ratio - 1.0))
            
            return accuracy
            
        except Exception as e:
            logger.debug(f"Duration accuracy computation failed: {e}")
            return 0.0
    
    def train_infinite_batch(self, batch_data, optimizer):
        """
        Train on single infinite continuous batch with virtual checkpoints
        """
        try:
            # Prepare batch for training
            training_batch = self.data_loader.prepare_batch_for_training(batch_data)
            
            text_tokens = training_batch['text_tokens']
            audio_codes = training_batch['audio_codes']
            duration = training_batch['duration']
            num_checkpoints = training_batch['num_checkpoints']
            
            logger.debug(f"🔄 Training infinite batch: {duration:.1f}s, {num_checkpoints} checkpoints")
            
            # Forward pass with fresh state
            optimizer.zero_grad()
            output = self.model(text_tokens, audio_codes, reset_states=True)
            
            # Compute virtual checkpoint losses
            loss_dict = self.compute_virtual_checkpoint_losses(output, training_batch)
            
            if loss_dict is not None:
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Add batch info
                loss_dict['batch_info'] = {
                    'duration': duration,
                    'audio_tokens': training_batch['audio_tokens'],
                    'text_tokens': training_batch['text_token_count'],
                    'checkpoints': num_checkpoints,
                    'batch_dir': training_batch.get('batch_dir', 'unknown'),
                    'infinite_processing': True
                }
                
                return loss_dict
            else:
                optimizer.zero_grad()
                return None
                
        except Exception as e:
            logger.debug(f"Infinite batch training failed: {e}")
            optimizer.zero_grad()
            return None
    
    def train(self, steps=2000, learning_rate=1e-3):
        """
        Main infinite training loop with virtual checkpoints
        """
        logger.info(f"🚀 Starting INFINITE training for {steps} steps")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   🔄 Fresh states between batches only")
        logger.info(f"   📍 Virtual checkpoint training every ~10s")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-8
        )
        
        # Metrics tracking
        successful_steps = 0
        failed_steps = 0
        losses = []
        accuracies = []
        duration_accuracies = []
        checkpoint_counts = []
        best_accuracy = 0.0
        best_duration_accuracy = 0.0
        
        # Training loop
        logger.info(f"⏱️  Starting infinite continuous training...")
        
        for step in range(steps):
            try:
                # Get random continuous batch
                batch_data = self.data_loader.get_random_continuous_batch()
                
                if batch_data is None:
                    failed_steps += 1
                    continue
                
                # Train on infinite batch
                loss_dict = self.train_infinite_batch(batch_data, optimizer)
                
                if loss_dict is not None:
                    # Track metrics
                    total_loss = loss_dict['total_loss'].item()
                    current_accuracy = loss_dict['accuracy']
                    current_duration_accuracy = loss_dict['duration_accuracy']
                    num_checkpoints = loss_dict.get('num_checkpoints', 0)
                    
                    losses.append(total_loss)
                    accuracies.append(current_accuracy)
                    duration_accuracies.append(current_duration_accuracy)
                    checkpoint_counts.append(num_checkpoints)
                    
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    if current_duration_accuracy > best_duration_accuracy:
                        best_duration_accuracy = current_duration_accuracy
                    
                    successful_steps += 1
                    
                    # Enhanced logging
                    if (step % 50 == 0 or current_accuracy > 0.15 or 
                        current_duration_accuracy > 0.4 or step < 10):
                        
                        logger.info(f"Step {step:4d}: Loss={total_loss:.4f}, "
                                  f"Acc={current_accuracy:.4f}, DurAcc={current_duration_accuracy:.4f}")
                        
                        batch_info = loss_dict['batch_info']
                        logger.info(f"         Batch: {batch_info['duration']:.1f}s, "
                                  f"{batch_info['audio_tokens']:,} tokens, "
                                  f"{batch_info['checkpoints']} checkpoints")
                        
                        # Show checkpoint details
                        if 'checkpoint_losses' in loss_dict:
                            checkpoint_losses = loss_dict['checkpoint_losses'][:3]  # Show first 3
                            for cp in checkpoint_losses:
                                logger.info(f"         CP {cp['checkpoint_idx']}: "
                                          f"{cp['time_start']:.1f}-{cp['time_end']:.1f}s, "
                                          f"Loss={cp['loss'].item():.4f}, "
                                          f"Acc={cp['accuracy']:.4f}")
                    
                    # Success detection
                    if current_accuracy > 0.6:
                        logger.info(f"🎉 EXCELLENT AUDIO PROGRESS! Accuracy {current_accuracy:.4f}")
                    if current_duration_accuracy > 0.7:
                        logger.info(f"🎉 EXCELLENT DURATION PROGRESS! Duration Accuracy {current_duration_accuracy:.4f}")
                    
                    # Early success check
                    if (best_accuracy > 0.4 and best_duration_accuracy > 0.6 and 
                        step > 1000):
                        logger.info(f"🎉 INFINITE TRAINING SUCCESS!")
                        logger.info(f"   Audio accuracy: {best_accuracy:.4f}")
                        logger.info(f"   Duration accuracy: {best_duration_accuracy:.4f}")
                        break
                        
                else:
                    failed_steps += 1
                    
                # Memory cleanup
                if step % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.debug(f"Step {step} failed: {e}")
                failed_steps += 1
                continue
        
        # Final results
        success_rate = successful_steps / (successful_steps + failed_steps) * 100 if (successful_steps + failed_steps) > 0 else 0
        
        final_loss = losses[-1] if losses else 999.0
        final_acc = accuracies[-1] if accuracies else 0.0
        final_dur_acc = duration_accuracies[-1] if duration_accuracies else 0.0
        avg_checkpoints = np.mean(checkpoint_counts) if checkpoint_counts else 0
        
        logger.info(f"\n🎉 INFINITE training completed!")
        logger.info(f"   Successful steps: {successful_steps}/{steps} ({success_rate:.1f}%)")
        logger.info(f"   Best audio accuracy: {best_accuracy:.4f}")
        logger.info(f"   Best duration accuracy: {best_duration_accuracy:.4f}")
        logger.info(f"   Final - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}, DurAcc: {final_dur_acc:.4f}")
        logger.info(f"   Average checkpoints per batch: {avg_checkpoints:.1f}")
        logger.info(f"   🔄 Infinite continuous processing")
        logger.info(f"   📍 Virtual checkpoint training")
        
        # Save model if successful
        if best_accuracy > 0.2 or best_duration_accuracy > 0.4:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'final_loss': final_loss,
                'final_accuracy': final_acc,
                'final_duration_accuracy': final_dur_acc,
                'best_accuracy': best_accuracy,
                'best_duration_accuracy': best_duration_accuracy,
                'successful_steps': successful_steps,
                'vocab_size': self.tokenizer.get_vocab_size(),
                'infinite_training': True,
                'continuous_processing': True,
                'virtual_checkpoints': True,
                'avg_checkpoints_per_batch': avg_checkpoints
            }, 'infinite_model.pt')
            
            logger.info("💾 INFINITE model saved as 'infinite_model.pt'")
            return True
        else:
            logger.warning("⚠️  Training not successful enough")
            return False


def main():
    """Main function for infinite training"""
    logger.info("🎯 INFINITE TTS Training - Continuous Processing")
    logger.info("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check continuous data  
    if not Path("continuous_data").exists():
        logger.error("❌ continuous_data directory not found!")
        logger.error("   Please run continuous_audio_preprocessor.py first")
        return
    
    try:
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        data_loader = InfiniteDataLoader("continuous_data", device)
        
        if len(data_loader) == 0:
            logger.error("❌ No continuous batches loaded!")
            return
        
        # Show data stats
        stats = data_loader.get_stats()
        logger.info(f"\n📊 Infinite Data Statistics:")
        logger.info(f"   Total batches: {stats['total_batches']}")
        logger.info(f"   Total duration: {stats['total_duration']:.1f}s ({stats['total_minutes']:.1f} min)")
        logger.info(f"   Total tokens: {stats['total_tokens']:,}")
        logger.info(f"   Average batch duration: {stats['avg_duration']:.1f}s")
        logger.info(f"   Total checkpoints: {stats['total_checkpoints']}")
        logger.info(f"   ✅ Infinite ready: {stats['infinite_ready']}")
        
        # Create infinite model
        model = InfiniteTTSModel(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_codebooks=4,
            codebook_size=1024,
            state_size=64
        ).to(device)
        
        # Create infinite trainer
        trainer = InfiniteTrainer(model, tokenizer, data_loader)
        
        logger.info(f"\n🚀 Starting INFINITE training...")
        logger.info(f"   🔄 Fresh states between batches only")
        logger.info(f"   📍 Virtual checkpoints every ~10s")
        logger.info(f"   🎯 Continuous processing - no internal chunking")
        
        # Train with infinite system
        success = trainer.train(steps=2000, learning_rate=1e-3)
        
        if success:
            logger.info("✅ INFINITE training successful!")
            logger.info("🎵 Ready for infinite continuous speech generation!")
        else:
            logger.warning("⚠️  May need more steps or adjustments")
    
    except Exception as e:
        logger.error(f"❌ INFINITE training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()