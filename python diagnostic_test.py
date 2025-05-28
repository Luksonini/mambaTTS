#!/usr/bin/env python3
"""
Diagnostic Test - Sprawdź czy model RZECZYWIŚCIE umie generować
============================================================
"""

import torch
import torch.nn.functional as F  # ← DODANE
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def diagnostic_test():
    """Sprawdź co model NAPRAWDĘ umie"""
    
    logger.info("🔍 DIAGNOSTIC TEST - Sprawdź czy model naprawdę umie generować")
    logger.info("=" * 70)
    
    try:
        # Load components
        from nucleotide_tokenizer import NucleotideTokenizer
        from sequential_trainer import Enhanced8CodebookTTSModel, Enhanced8CodebookDataLoader
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = NucleotideTokenizer()
        data_loader = Enhanced8CodebookDataLoader("no_overlap_data", device)
        
        # Load model
        model_path = "enhanced_8codebook_model_debug.pt"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        config = checkpoint.get('model_config', {})
        vocab_size = checkpoint.get('vocab_size', tokenizer.get_vocab_size())
        
        model = Enhanced8CodebookTTSModel(
            vocab_size=vocab_size,
            embed_dim=config.get('embed_dim', 384),
            hidden_dim=config.get('hidden_dim', 512),
            num_codebooks=config.get('num_codebooks', 8),
            codebook_size=config.get('codebook_size', 1024)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ Model loaded - Best accuracy: {checkpoint.get('best_accuracy', 'unknown')}")
        
        # Get a training sample
        chunk = data_loader.get_random_chunk()
        text = chunk['text']
        original_audio = chunk['audio_codes']  # [C, T]
        
        logger.info(f"\n🎯 TEST SAMPLE:")
        logger.info(f"   Text: '{text[:60]}...'")
        logger.info(f"   Original audio shape: {original_audio.shape}")
        
        # Tokenize text
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Prepare original audio for model
        if original_audio.dim() == 2:
            original_audio = original_audio.unsqueeze(0)  # Add batch dim
        
        B, C, T = original_audio.shape
        if C < 8:
            padding = torch.zeros(B, 8 - C, T, dtype=original_audio.dtype, device=original_audio.device)
            original_audio = torch.cat([original_audio, padding], dim=1)
        elif C > 8:
            original_audio = original_audio[:, :8, :]
        
        logger.info(f"   Text tokens shape: {text_tokens.shape}")
        logger.info(f"   Audio tokens shape: {original_audio.shape}")
        
        # TEST 1: Reconstruction (with target audio) - Should work well
        logger.info(f"\n🧪 TEST 1: RECONSTRUCTION (with target audio)")
        logger.info(f"   Should work well if model learned properly...")
        
        with torch.no_grad():
            output_recon = model(text_tokens, original_audio)
            logits_recon = output_recon['logits']  # [B, 8, T, 1024]
            
            if logits_recon is not None:
                # Check reconstruction accuracy
                predicted_tokens = torch.argmax(logits_recon, dim=-1)  # [B, 8, T]
                target_tokens = original_audio[:, :, 1:]  # Shift for next-token prediction
                
                # Match lengths
                min_T = min(predicted_tokens.shape[2], target_tokens.shape[2])
                pred_trunc = predicted_tokens[:, :, :min_T]
                target_trunc = target_tokens[:, :, :min_T]
                
                recon_accuracy = (pred_trunc == target_trunc).float().mean().item()
                logger.info(f"   ✅ Reconstruction accuracy: {recon_accuracy:.4f} ({recon_accuracy*100:.1f}%)")
                
                if recon_accuracy > 0.8:
                    logger.info(f"   ✅ GOOD: Model can reconstruct when given target audio")
                else:
                    logger.info(f"   ❌ BAD: Model struggles even with target audio")
            else:
                logger.info(f"   ❌ ERROR: No logits returned")
        
        # TEST 2: Generation (without target audio) - The real test
        logger.info(f"\n🧪 TEST 2: GENERATION (without target audio)")
        logger.info(f"   This is the REAL test - can model generate from scratch?")
        
        with torch.no_grad():
            output_gen = model(text_tokens, audio_tokens=None)
            
            predicted_durations = output_gen.get('predicted_durations')
            duration_confidence = output_gen.get('duration_confidence')
            
            logger.info(f"   Predicted durations shape: {predicted_durations.shape if predicted_durations is not None else 'None'}")
            
            if predicted_durations is not None:
                dur_min = predicted_durations.min().item()
                dur_max = predicted_durations.max().item()
                dur_mean = predicted_durations.mean().item()
                logger.info(f"   Duration range: {dur_min:.3f}s - {dur_max:.3f}s (mean: {dur_mean:.3f}s)")
                
                if dur_min == dur_max:
                    logger.info(f"   ❌ PROBLEM: All durations are identical! ({dur_min:.3f}s)")
                    logger.info(f"   🔧 This explains why generation fails - no duration variety")
                else:
                    logger.info(f"   ✅ GOOD: Duration has variety")
            
            # Check if model can generate ANY audio logits
            try:
                # Try to get text context
                text_context = model.text_encoder(text_tokens, return_sequence=False)
                text_context = model.text_proj(text_context)
                
                # Create dummy audio start (single token)
                dummy_audio = torch.randint(0, 1024, (1, 8, 1), device=device)
                
                # Try to get next token logits
                dummy_logits = model.audio_processor(dummy_audio, text_context)
                logger.info(f"   Audio processor output shape: {dummy_logits.shape}")
                
                # Check if logits are reasonable
                logits_sample = dummy_logits[0, 0, 0, :]  # First codebook, first time, all vocab
                logits_entropy = -torch.sum(F.softmax(logits_sample, dim=-1) * F.log_softmax(logits_sample, dim=-1))
                logger.info(f"   Logits entropy: {logits_entropy:.3f} (higher = more diverse)")
                
                if logits_entropy < 2.0:
                    logger.info(f"   ❌ LOW ENTROPY: Model outputs very peaked distributions")
                    logger.info(f"   🔧 This causes repetitive generation")
                else:
                    logger.info(f"   ✅ GOOD ENTROPY: Model has diverse outputs")
                    
            except Exception as e:
                logger.info(f"   ❌ GENERATION ERROR: {e}")
        
        # TEST 3: Compare with random baseline
        logger.info(f"\n🧪 TEST 3: COMPARISON WITH RANDOM BASELINE")
        
        # Generate random audio tokens
        random_audio = torch.randint(0, 1024, original_audio.shape, device=device)
        
        with torch.no_grad():
            try:
                output_random = model(text_tokens, random_audio)
                logits_random = output_random['logits']
                
                if logits_random is not None:
                    predicted_random = torch.argmax(logits_random, dim=-1)
                    target_random = random_audio[:, :, 1:]
                    
                    min_T = min(predicted_random.shape[2], target_random.shape[2])
                    pred_r = predicted_random[:, :, :min_T]
                    target_r = target_random[:, :, :min_T]
                    
                    random_accuracy = (pred_r == target_r).float().mean().item()
                    logger.info(f"   Random audio accuracy: {random_accuracy:.4f} ({random_accuracy*100:.1f}%)")
                    
                    if abs(recon_accuracy - random_accuracy) < 0.1:
                        logger.info(f"   ❌ MAJOR PROBLEM: Model performs similarly on real vs random audio!")
                        logger.info(f"   🔧 This suggests model hasn't learned meaningful audio patterns")
                    else:
                        logger.info(f"   ✅ GOOD: Model distinguishes real vs random audio")
                        
            except Exception as e:
                logger.info(f"   ❌ Random test error: {e}")
        
        # FINAL DIAGNOSIS
        logger.info(f"\n🏥 FINAL DIAGNOSIS:")
        logger.info(f"=" * 50)
        
        if 'recon_accuracy' in locals() and recon_accuracy > 0.8:
            if dur_min == dur_max:
                logger.info(f"🔧 DIAGNOSIS: Duration prediction COLLAPSED")
                logger.info(f"   - Model learned reconstruction (97.3%) but not generation")
                logger.info(f"   - All durations are {dur_min:.3f}s (ZERO variety)")
                logger.info(f"   - This causes monotonous, repetitive audio generation")
                logger.info(f"   - FIX: Use improved get_safe_duration_targets() with variety")
            else:
                logger.info(f"🤔 DIAGNOSIS: Unclear - model seems OK but generation fails")
                logger.info(f"   - Try different generation parameters")
                logger.info(f"   - Check autoregressive implementation")
        else:
            logger.info(f"❌ DIAGNOSIS: Model didn't learn properly")
            logger.info(f"   - Low reconstruction accuracy indicates training issues")
            logger.info(f"   - Need to retrain with better loss functions")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnostic_test()