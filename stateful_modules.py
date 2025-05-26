#!/usr/bin/env python3
"""
No-Overlap Training System - Clean Sequential Learning
=====================================================
Training system designed for NO-OVERLAP clean chunks
Key features:
- Works with no_overlap_data/ structure
- Uses Stateful Modules for clean transitions
- Batch-aware training (sequential chunks)
- No masking needed - clean boundaries
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import warnings
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import our new stateful modules
from stateful_modules import (
    StatefulMambaTextEncoder,
    StatefulMambaAudioProcessor, 
    StatefulDurationRegulator,
    AudioStyleExtractor,
    BatchStateManager,
    StreamingInferenceManager
)
from nucleotide_tokenizer import NucleotideTokenizer


class NoOverlapDataLoader:
    """
    Data loader specifically for NO-OVERLAP clean chunks
    """
    def __init__(self, data_dir="no_overlap_data", device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = device
        self.batches = []
        self.chunks = []
        
        logger.info(f"🔍 Loading NO-OVERLAP data from {data_dir}")
        self._load_all_batches()
        
    def _load_all_batches(self):
        """Load all clean batches and chunks"""
        if not self.data_dir.exists():
            logger.error(f"❌ NO-OVERLAP data directory not found: {self.data_dir}")
            return
            
        batch_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('clean_batch_')]
        batch_dirs.sort()
        
        logger.info(f"📁 Found {len(batch_dirs)} clean batches")
        
        for batch_dir in batch_dirs:
            try:
                # Load batch metadata
                meta_path = batch_dir / "batch_meta.json"
                if not meta_path.exists():
                    logger.warning(f"⚠️  No metadata for {batch_dir.name}")
                    continue
                    
                with open(meta_path, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)
                
                # Verify this is a NO-OVERLAP batch
                if not batch_meta.get('no_overlaps', False):
                    logger.warning(f"⚠️  Batch {batch_dir.name} not marked as no-overlap")
                    continue
                
                # Load all chunks in this batch
                batch_chunks = []
                for chunk_file in batch_meta['chunk_files']:
                    chunk_path = batch_dir / chunk_file
                    if chunk_path.exists():
                        try:
                            chunk_data = torch.load(chunk_path, map_location=self.device)
                            
                            # Verify chunk is clean
                            if not chunk_data.get('clean_chunk', False):
                                logger.warning(f"⚠️  Chunk {chunk_file} not marked as clean")
                                continue
                                
                            # Add batch info to chunk
                            chunk_data['batch_dir'] = batch_dir.name
                            chunk_data['batch_meta'] = batch_meta
                            
                            batch_chunks.append(chunk_data)
                            self.chunks.append(chunk_data)
                            
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to load {chunk_file}: {e}")
                            continue
                
                if len(batch_chunks) > 0:
                    batch_info = {
                        'batch_idx': batch_meta['batch_idx'],
                        'batch_dir': batch_dir.name,
                        'chunks': batch_chunks,
                        'metadata': batch_meta,
                        'clean_sequential': True,
                        'no_overlaps': True
                    }
                    self.batches.append(batch_info)
                    
                    logger.info(f"   ✅ Loaded {batch_dir.name}: {len(batch_chunks)} clean chunks")
                else:
                    logger.warning(f"⚠️  No valid chunks in {batch_dir.name}")
                    
            except Exception as e:
                logger.warning(f"⚠️  Failed to load batch {batch_dir.name}: {e}")
                continue
        
        logger.info(f"📊 NO-OVERLAP Data loaded:")
        logger.info(f"   Total batches: {len(self.batches)}")
        logger.info(f"   Total chunks: {len(self.chunks)}")
        logger.info(f"   All chunks are CLEAN (no overlaps) ✅")
        
        if len(self.chunks) == 0:
            logger.error("❌ No valid chunks loaded!")
    
    def get_random_chunk(self):
        """Get random clean chunk"""
        if len(self.chunks) == 0:
            return None
        return self.chunks[np.random.randint(0, len(self.chunks))]
    
    def get_sequential_batch(self, batch_idx=None):
        """Get sequential batch of clean chunks"""
        if len(self.batches) == 0:
            return None
            
        if batch_idx is None:
            batch_idx = np.random.randint(0, len(self.batches))
        elif batch_idx >= len(self.batches):
            return None
            
        return self.batches[batch_idx]
    
    def get_stats(self):
        """Get data statistics"""
        if len(self.chunks) == 0:
            return {'total_chunks': 0, 'total_batches': 0}
            
        total_duration = sum(chunk.get('duration', 0) for chunk in self.chunks)
        total_words = sum(chunk.get('word_count', 0) for chunk in self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'total_batches': len(self.batches),
            'total_duration': total_duration,
            'total_words': total_words,
            'avg_chunk_duration': total_duration / len(self.chunks),
            'no_overlaps': True,
            'clean_boundaries': True
        }


class StatefulTTSModel(nn.Module):
    """
    TTS Model with Stateful Processing for NO-OVERLAP chunks
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
        
        # Stateful components for clean chunk processing
        self.text_encoder = StatefulMambaTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            state_size=state_size,
            num_layers=3
        )
        
        self.duration_regulator = StatefulDurationRegulator(
            text_dim=embed_dim,
            style_dim=64,
            hidden_dim=128,
            tokens_per_second=75.0,  # Proven value
            state_size=state_size
        )
        
        self.audio_processor = StatefulMambaAudioProcessor(
            hidden_dim=hidden_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            state_size=state_size,
            num_layers=2
        )
        
        self.style_extractor = AudioStyleExtractor(
            audio_dim=hidden_dim,
            style_dim=64
        )
        
        # Projections
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        self.default_style = nn.Parameter(torch.randn(64) * 0.01)
        
        logger.info(f"🧠 StatefulTTSModel: {sum(p.numel() for p in self.parameters()):,} parameters")
        logger.info(f"   🔄 Stateful processing for clean chunks")
        logger.info(f"   🎯 NO-OVERLAP optimized")
    
    def forward(self, text_tokens, audio_tokens=None, text_states=None, 
                audio_states=None, duration_state=None, return_states=False):
        """Forward pass with state management"""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Stateful text encoding
        if return_states:
            text_features, new_text_states = self.text_encoder(
                text_tokens, 
                initial_states=text_states,
                return_states=True,
                return_sequence=True
            )
        else:
            text_features = self.text_encoder(
                text_tokens,
                initial_states=text_states,
                return_sequence=True
            )
            new_text_states = None
        
        # Global text context
        text_context = self.text_encoder(
            text_tokens,
            initial_states=text_states,
            return_sequence=False
        )
        text_context = self.text_proj(text_context)
        
        # Style extraction
        if audio_tokens is not None:
            with torch.no_grad():
                B, C, T = audio_tokens.shape
                # Create pseudo audio features for style extraction
                audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
                pseudo_audio = audio_mean.expand(B, self.hidden_dim, min(T, 100))
                style_embedding = self.style_extractor(pseudo_audio)
        else:
            style_embedding = self.default_style.unsqueeze(0).expand(batch_size, -1)
        
        # Stateful duration regulation
        if return_states:
            regulated_features, predicted_durations, duration_tokens, duration_confidence, new_duration_state = \
                self.duration_regulator(
                    text_features,
                    style_embedding,
                    initial_state=duration_state,
                    return_state=True
                )
        else:
            regulated_features, predicted_durations, duration_tokens, duration_confidence = \
                self.duration_regulator(
                    text_features,
                    style_embedding,
                    initial_state=duration_state
                )
            new_duration_state = None
        
        # Stateful audio processing
        if audio_tokens is not None:
            regulated_context = torch.mean(self.text_proj(regulated_features), dim=1)
            
            if return_states:
                audio_logits, new_audio_states = self.audio_processor(
                    audio_tokens,
                    regulated_context,
                    initial_states=audio_states,
                    return_states=True
                )
            else:
                audio_logits = self.audio_processor(
                    audio_tokens,
                    regulated_context,
                    initial_states=audio_states
                )
                new_audio_states = None
        else:
            audio_logits = None
            new_audio_states = None
        
        result = {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'duration_tokens': duration_tokens,
            'duration_confidence': duration_confidence,
            'text_features': text_features,
            'regulated_features': regulated_features,
            'style_embedding': style_embedding
        }
        
        if return_states:
            result['states'] = {
                'text_states': new_text_states,
                'audio_states': new_audio_states,
                'duration_state': new_duration_state
            }
        
        return result


class NoOverlapTrainer:
    """
    Trainer specifically designed for NO-OVERLAP sequential chunks
    """
    def __init__(self, model, tokenizer, data_loader):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        
        # State management for sequential training
        self.state_manager = BatchStateManager(max_batch_size=32)
        
        logger.info(f"🎯 NoOverlapTrainer initialized")
        logger.info(f"   Data: {data_loader.get_stats()['total_chunks']} NO-OVERLAP chunks")
        logger.info(f"   🔄 Stateful processing enabled")
        logger.info(f"   🧠 State management active")
    
    def train_single_chunk(self, chunk_data, states=None):
        """Train on single clean chunk with optional state continuity"""
        try:
            # Prepare data
            text_tokens = chunk_data['text_tokens']
            if text_tokens.dim() == 1:
                text_tokens = text_tokens.unsqueeze(0)
                
            audio_codes = chunk_data['audio_codes']
            if audio_codes.dim() == 2:
                audio_codes = audio_codes.unsqueeze(0)
            
            chunk_duration = chunk_data.get('duration', 4.0)
            
            # Extract states if provided
            text_states = states.get('text_states') if states else None
            audio_states = states.get('audio_states') if states else None
            duration_state = states.get('duration_state') if states else None
            
            # Forward pass with state management
            output = self.model(
                text_tokens, 
                audio_codes,
                text_states=text_states,
                audio_states=audio_states,
                duration_state=duration_state,
                return_states=True
            )
            
            # Compute losses (using fixed loss functions)
            from losses import compute_combined_loss
            loss_dict = compute_combined_loss(output, chunk_data, text_tokens, self.device)
            
            # Add state information
            loss_dict['chunk_info'] = {
                'text': chunk_data['text'][:50] + "..." if len(chunk_data['text']) > 50 else chunk_data['text'],
                'duration': chunk_duration,
                'batch_dir': chunk_data.get('batch_dir', 'unknown'),
                'chunk_idx': chunk_data.get('chunk_idx_in_batch', 0),
                'clean_chunk': chunk_data.get('clean_chunk', False),
                'has_overlap': chunk_data.get('has_overlap', True)  # Should be False for clean chunks
            }
            
            # Return loss dict and new states
            return loss_dict, output.get('states')
            
        except Exception as e:
            logger.debug(f"Single chunk training failed: {e}")
            return None, None
    
    def train_sequential_batch(self, batch_info, optimizer):
        """Train on sequential batch of clean chunks (with state continuity)"""
        chunks = batch_info['chunks']
        batch_idx = batch_info['batch_idx']
        
        logger.debug(f"🔄 Training sequential batch {batch_idx} ({len(chunks)} clean chunks)")
        
        batch_losses = []
        batch_accuracies = []
        batch_duration_accuracies = []
        
        # Initialize states for this batch
        current_states = None
        
        for chunk_idx, chunk_data in enumerate(chunks):
            try:
                # Train on this chunk with state continuity
                loss_dict, new_states = self.train_single_chunk(chunk_data, current_states)
                
                if loss_dict is not None:
                    # Backward pass
                    total_loss = loss_dict['total_loss']
                    total_loss.backward()
                    
                    # Accumulate metrics
                    batch_losses.append(total_loss.item())
                    batch_accuracies.append(loss_dict['accuracy'])
                    batch_duration_accuracies.append(loss_dict['duration_accuracy'])
                    
                    # Update states for next chunk
                    current_states = new_states
                    
                    logger.debug(f"   Chunk {chunk_idx}: Loss={total_loss.item():.4f}, "
                               f"Acc={loss_dict['accuracy']:.4f}, DurAcc={loss_dict['duration_accuracy']:.4f}")
                else:
                    logger.debug(f"   Chunk {chunk_idx}: FAILED")
                    current_states = None  # Reset states on failure
                    
            except Exception as e:
                logger.debug(f"   Chunk {chunk_idx} failed: {e}")
                current_states = None  # Reset states on failure
                continue
        
        # Step optimizer after processing entire batch
        if len(batch_losses) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Return batch statistics
            return {
                'batch_loss': np.mean(batch_losses),
                'batch_accuracy': np.mean(batch_accuracies),
                'batch_duration_accuracy': np.mean(batch_duration_accuracies),
                'successful_chunks': len(batch_losses),
                'total_chunks': len(chunks)
            }
        else:
            optimizer.zero_grad()  # Clear gradients even on failure
            return None
    
    def train(self, steps=3000, learning_rate=2e-3, use_sequential_batches=True):
        """Main training loop for NO-OVERLAP system"""
        logger.info(f"🚀 Starting NO-OVERLAP training for {steps} steps")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Sequential batches: {use_sequential_batches}")
        logger.info(f"   Clean boundaries: ✅ No masking needed")
        
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
        best_accuracy = 0.0
        best_duration_accuracy = 0.0
        
        # Training loop
        logger.info(f"⏱️  Training for {steps} steps...")
        
        step = 0
        while step < steps:
            try:
                if use_sequential_batches and step % 4 == 0:  # Every 4 steps, try a sequential batch
                    # Sequential batch training (with state continuity)
                    batch_info = self.data_loader.get_sequential_batch()
                    
                    if batch_info is not None:
                        batch_result = self.train_sequential_batch(batch_info, optimizer)
                        
                        if batch_result is not None:
                            # Track metrics from sequential batch
                            batch_loss = batch_result['batch_loss']
                            batch_acc = batch_result['batch_accuracy']
                            batch_dur_acc = batch_result['batch_duration_accuracy']
                            
                            losses.append(batch_loss)
                            accuracies.append(batch_acc)
                            duration_accuracies.append(batch_dur_acc)
                            
                            if batch_acc > best_accuracy:
                                best_accuracy = batch_acc
                            if batch_dur_acc > best_duration_accuracy:
                                best_duration_accuracy = batch_dur_acc
                            
                            successful_steps += 1
                            
                            logger.info(f"Step {step:4d} [BATCH]: Loss={batch_loss:.4f}, "
                                      f"Acc={batch_acc:.4f}, DurAcc={batch_dur_acc:.4f}")
                            logger.info(f"         Sequential chunks: {batch_result['successful_chunks']}/{batch_result['total_chunks']}")
                        else:
                            failed_steps += 1
                    else:
                        failed_steps += 1
                else:
                    # Single chunk training (fallback)
                    chunk_data = self.data_loader.get_random_chunk()
                    
                    if chunk_data is not None:
                        optimizer.zero_grad()
                        loss_dict, _ = self.train_single_chunk(chunk_data)
                        
                        if loss_dict is not None:
                            total_loss = loss_dict['total_loss']
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()
                            
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
                            
                            if step % 100 == 0:
                                logger.info(f"Step {step:4d}: Loss={total_loss.item():.4f}, "
                                          f"Acc={current_accuracy:.4f}, DurAcc={current_duration_accuracy:.4f}")
                        else:
                            failed_steps += 1
                    else:
                        failed_steps += 1
                
                step += 1
                
                # Success detection
                if best_accuracy > 0.5 and best_duration_accuracy > 0.5:
                    logger.info(f"🎉 EXCELLENT PROGRESS! Both metrics > 50%")
                    if step > 1000:
                        logger.info(f"🎉 NO-OVERLAP TRAINING SUCCESS!")
                        break
                
                # Memory cleanup
                if step % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.debug(f"Step {step} failed: {e}")
                failed_steps += 1
                step += 1
                continue
        
        # Final results
        success_rate = successful_steps / (successful_steps + failed_steps) * 100 if (successful_steps + failed_steps) > 0 else 0
        
        final_loss = losses[-1] if losses else 999.0
        final_acc = accuracies[-1] if accuracies else 0.0
        final_dur_acc = duration_accuracies[-1] if duration_accuracies else 0.0
        
        logger.info(f"\n🎉 NO-OVERLAP training completed!")
        logger.info(f"   Successful steps: {successful_steps}/{steps} ({success_rate:.1f}%)")
        logger.info(f"   Best audio accuracy: {best_accuracy:.4f}")
        logger.info(f"   Best duration accuracy: {best_duration_accuracy:.4f}")
        logger.info(f"   Final - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}, DurAcc: {final_dur_acc:.4f}")
        logger.info(f"   🔄 Stateful processing: ENABLED")
        logger.info(f"   🎯 Clean boundaries: NO masking needed")
        
        # Save model if successful
        if best_accuracy > 0.1 or best_duration_accuracy > 0.3:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'final_loss': final_loss,
                'final_accuracy': final_acc,
                'final_duration_accuracy': final_dur_acc,
                'best_accuracy': best_accuracy,
                'best_duration_accuracy': best_duration_accuracy,
                'successful_steps': successful_steps,
                'vocab_size': self.tokenizer.get_vocab_size(),
                'no_overlap_training': True,
                'stateful_processing': True,
                'clean_boundaries': True
            }, 'no_overlap_model.pt')
            
            logger.info("💾 NO-OVERLAP model saved as 'no_overlap_model.pt'")
            return True
        else:
            logger.warning("⚠️  Training not successful enough")
            return False


def main():
    """Main function for NO-OVERLAP training"""
    logger.info("🎯 NO-OVERLAP TTS Training - Stateful Clean Chunks")
    logger.info("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check NO-OVERLAP data
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        logger.error("❌ no_overlap_data directory not found!")
        logger.error("   Please run audio_preprocessor.py first to create NO-OVERLAP data")
        logger.info(f"   Looking for: {data_path.absolute()}")
        
        # Check if there are any similar directories
        current_dir = Path(".")
        similar_dirs = [d for d in current_dir.iterdir() if d.is_dir() and ("data" in d.name.lower() or "overlap" in d.name.lower())]
        if similar_dirs:
            logger.info("   Found similar directories:")
            for d in similar_dirs:
                logger.info(f"     - {d.name}")
        return
    
    try:
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        data_loader = NoOverlapDataLoader(
            data_dir="no_overlap_data",
            device=device
        )
        
        # Show data stats
        stats = data_loader.get_stats()
        if stats['total_chunks'] == 0:
            logger.error("❌ No chunks found in no_overlap_data!")
            logger.info("   Checking directory structure...")
            
            # Debug directory structure
            data_dir = Path("no_overlap_data")
            if data_dir.exists():
                subdirs = list(data_dir.iterdir())
                logger.info(f"   Found {len(subdirs)} items in no_overlap_data/:")
                for item in subdirs[:10]:  # Show first 10 items
                    logger.info(f"     - {item.name} ({'dir' if item.is_dir() else 'file'})")
                if len(subdirs) > 10:
                    logger.info(f"     ... and {len(subdirs) - 10} more items")
            return
        
        logger.info(f"\n📊 NO-OVERLAP Data Statistics:")
        logger.info(f"   Total chunks: {stats['total_chunks']}")
        logger.info(f"   Total batches: {stats['total_batches']}")
        logger.info(f"   Total duration: {stats['total_duration']:.1f}s")
        logger.info(f"   Avg chunk duration: {stats['avg_chunk_duration']:.2f}s")
        logger.info(f"   ✅ NO overlaps: {stats['no_overlaps']}")
        logger.info(f"   ✅ Clean boundaries: {stats['clean_boundaries']}")
        
        # Create stateful model
        model = StatefulTTSModel(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_codebooks=4,
            codebook_size=1024,
            state_size=64
        ).to(device)
        
        # Create NO-OVERLAP trainer
        trainer = NoOverlapTrainer(model, tokenizer, data_loader)
        
        logger.info(f"\n🚀 Starting NO-OVERLAP training...")
        logger.info(f"   🔄 Stateful processing: Will learn clean transitions")
        logger.info(f"   🎯 No masking: Clean chunk boundaries")
        logger.info(f"   📈 Sequential batches: Better context learning")
        
        # Train with NO-OVERLAP system
        success = trainer.train(
            steps=3000, 
            learning_rate=2e-3, 
            use_sequential_batches=True
        )
        
        if success:
            logger.info("✅ NO-OVERLAP training successful!")
            logger.info("🎵 Ready for stateful speech generation!")
        else:
            logger.warning("⚠️  May need more steps or adjustments")
    
    except Exception as e:
        logger.error(f"❌ NO-OVERLAP training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()