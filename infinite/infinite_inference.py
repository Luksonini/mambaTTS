# #!/usr/bin/env python3
# """
# Infinite Inference - Continuous Speech Generation
# ===============================================
# Generate speech using infinite continuous processing
# Key features:
# - Load infinite trained model
# - Generate long continuous sequences
# - No chunking - true infinite generation
# - Virtual monitoring for quality control
# """

# import torch
# import torchaudio
# import json
# import logging
# from pathlib import Path
# import numpy as np

# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logger = logging.getLogger(__name__)

# def load_infinite_model():
#     """Load the trained infinite model"""
#     model_path = Path("infinite_model.pt")
    
#     if not model_path.exists():
#         logger.error("❌ Infinite model file not found: infinite_model.pt")
#         logger.info("   Please run infinite_trainer.py first")
#         return None, None
    
#     try:
#         checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
#         logger.info("📦 Loading INFINITE model...")
#         logger.info(f"   Best accuracy: {checkpoint.get('best_accuracy', 0):.4f}")
#         logger.info(f"   Best duration accuracy: {checkpoint.get('best_duration_accuracy', 0):.4f}")
#         logger.info(f"   Infinite training: {checkpoint.get('infinite_training', False)}")
#         logger.info(f"   Continuous processing: {checkpoint.get('continuous_processing', False)}")
#         logger.info(f"   Virtual checkpoints: {checkpoint.get('virtual_checkpoints', False)}")
        
#         from nucleotide_tokenizer import NucleotideTokenizer
#         from infinite_trainer import InfiniteTTSModel
        
#         tokenizer = NucleotideTokenizer()
#         vocab_size = tokenizer.get_vocab_size()
        
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         model = InfiniteTTSModel(
#             vocab_size=vocab_size,
#             embed_dim=128,
#             hidden_dim=256,
#             num_codebooks=4,
#             codebook_size=1024,
#             state_size=64
#         ).to(device)
        
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
        
#         logger.info(f"✅ INFINITE model loaded successfully on {device}")
#         return model, tokenizer
        
#     except Exception as e:
#         logger.error(f"❌ Failed to load infinite model: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None


# def load_infinite_batch(batch_name):
#     """Load a continuous batch from continuous_data/"""
#     data_dir = Path("continuous_data")
    
#     if not data_dir.exists():
#         logger.error("❌ continuous_data directory not found")
#         return None
    
#     batch_dir = data_dir / batch_name
#     if not batch_dir.exists():
#         logger.error(f"❌ Batch directory not found: {batch_name}")
#         # List available batches
#         available_batches = [d.name for d in data_dir.iterdir() 
#                            if d.is_dir() and d.name.startswith('continuous_batch_')]
#         available_batches.sort()
#         logger.info(f"   Available batches: {available_batches[:5]}{'...' if len(available_batches) > 5 else ''}")
#         return None
    
#     # Load batch metadata
#     meta_path = batch_dir / "batch_meta.json"
#     if not meta_path.exists():
#         logger.error(f"❌ No metadata found for {batch_name}")
#         return None
    
#     try:
#         with open(meta_path, 'r', encoding='utf-8') as f:
#             batch_meta = json.load(f)
        
#         logger.info(f"📦 Loading continuous batch: {batch_name}")
#         logger.info(f"   Metadata: {batch_meta.get('duration', 0):.1f}s, "
#                   f"continuous={batch_meta.get('continuous_batch', False)}")
        
#         # Load the continuous batch file
#         batch_file = batch_meta.get('batch_file')
#         if not batch_file:
#             logger.error(f"❌ No batch file specified")
#             return None
        
#         batch_path = batch_dir / batch_file
#         if not batch_path.exists():
#             logger.error(f"❌ Batch file not found: {batch_file}")
#             return None
        
#         # Load continuous batch data
#         batch_data = torch.load(batch_path, map_location='cpu', weights_only=False)
        
#         # Verify it's a continuous batch
#         if not batch_data.get('continuous_batch', False):
#             logger.warning(f"⚠️  {batch_name} not marked as continuous batch")
        
#         if not batch_data.get('no_internal_chunking', False):
#             logger.warning(f"⚠️  {batch_name} has internal chunking")
        
#         logger.info(f"✅ Loaded continuous batch: {batch_data['duration']:.1f}s, "
#                   f"{batch_data['audio_tokens']:,} tokens")
        
#         return batch_data
        
#     except Exception as e:
#         logger.error(f"❌ Failed to load batch {batch_name}: {e}")
#         return None


# def reconstruct_infinite_batch(model, tokenizer, batch_name):
#     """
#     Reconstruct full audio from infinite continuous batch
#     """
#     logger.info(f"🎵 Reconstructing INFINITE batch: {batch_name}")
    
#     # Load continuous batch
#     batch_data = load_infinite_batch(batch_name)
#     if batch_data is None:
#         return None
    
#     logger.info(f"📦 Loaded continuous batch: {batch_data['duration']:.1f}s")
#     logger.info(f"   🔄 Continuous processing - no internal chunking")
#     logger.info(f"   📍 Virtual checkpoints: {batch_data.get('num_checkpoints', 0)}")
    
#     # Get continuous audio codes
#     continuous_audio_codes = batch_data['continuous_audio_codes']  # [C, T]
    
#     logger.info(f"🎵 Audio codes shape: {continuous_audio_codes.shape}")
#     logger.info(f"   Total tokens: {continuous_audio_codes.shape[1]:,}")
#     logger.info(f"   Expected duration: {continuous_audio_codes.shape[1]/75:.1f}s")
    
#     # Decode to audio using EnCodec
#     try:
#         from encodec import EncodecModel
        
#         codec = EncodecModel.encodec_model_24khz()
#         codec.set_target_bandwidth(3.0)
#         codec.eval()
        
#         device = next(model.parameters()).device
#         if device.type == 'cuda':
#             codec = codec.cuda()
#             continuous_audio_codes = continuous_audio_codes.to(device)
        
#         # Add batch dimension for decoding
#         decode_tokens = continuous_audio_codes.unsqueeze(0)  # [1, C, T]
        
#         logger.info(f"🔊 Decoding infinite sequence: {decode_tokens.shape}")
        
#         with torch.no_grad():
#             decoded_audio = codec.decode([(decode_tokens, None)])
            
#             if len(decoded_audio) > 0:
#                 waveform = decoded_audio[0].squeeze().cpu()
                
#                 actual_duration = len(waveform) / 24000
#                 expected_duration = continuous_audio_codes.shape[1] / 75
#                 original_duration = batch_data['duration']
#                 rms = torch.sqrt(torch.mean(waveform**2)).item()
#                 max_amp = torch.max(torch.abs(waveform)).item()
                
#                 logger.info(f"✅ INFINITE batch reconstructed!")
#                 logger.info(f"   Waveform: {waveform.shape} samples")
#                 logger.info(f"   Actual duration: {actual_duration:.2f}s")
#                 logger.info(f"   Expected duration: {expected_duration:.2f}s")
#                 logger.info(f"   Original duration: {original_duration:.2f}s")
#                 logger.info(f"   RMS: {rms:.4f}")
#                 logger.info(f"   Max amplitude: {max_amp:.4f}")
                
#                 # Calculate ratios
#                 expected_ratio = actual_duration / expected_duration if expected_duration > 0 else 0
#                 original_ratio = actual_duration / original_duration if original_duration > 0 else 0
                
#                 logger.info(f"📊 Duration Analysis:")
#                 logger.info(f"   vs Expected: {expected_ratio:.2f}x")
#                 logger.info(f"   vs Original: {original_ratio:.2f}x")
#                 logger.info(f"   🎯 Infinite continuous - no gaps!")
                
#                 # Save reconstructed audio
#                 output_path = f"infinite_batch_{batch_name}_{actual_duration:.1f}s.wav"
#                 torchaudio.save(output_path, waveform.unsqueeze(0), 24000)
                
#                 logger.info(f"💾 INFINITE batch saved: {output_path}")
                
#                 return {
#                     'success': True,
#                     'audio_file': output_path,
#                     'actual_duration': actual_duration,
#                     'expected_duration': expected_duration,
#                     'original_duration': original_duration,
#                     'waveform_shape': waveform.shape,
#                     'rms': rms,
#                     'max_amplitude': max_amp,
#                     'expected_ratio': expected_ratio,
#                     'original_ratio': original_ratio,
#                     'infinite_processing': True
#                 }
#             else:
#                 logger.error("❌ EnCodec returned empty audio")
#                 return None
                
#     except Exception as e:
#         logger.error(f"❌ Infinite reconstruction failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# def generate_infinite_speech(model, tokenizer, text_input, max_length=300, 
#                            virtual_checkpoints=True):
#     """
#     Generate speech from text using infinite continuous processing
#     """
#     logger.info(f"🎤 Generating INFINITE speech from text:")
#     logger.info(f"   Text: '{text_input}'")
#     logger.info(f"   Max length: {max_length} tokens")
#     logger.info(f"   Virtual checkpoints: {virtual_checkpoints}")
    
#     device = next(model.parameters()).device
    
#     try:
#         # Tokenize text
#         text_tokens = tokenizer.encode(text_input, add_special_tokens=True)
#         text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
#         logger.info(f"   Text tokens: {text_tokens.shape}")
        
#         with torch.no_grad():
#             # Reset infinite states for new generation
#             model.reset_infinite_states(1)
            
#             # Get text features and duration prediction
#             output = model(text_tokens, audio_tokens=None, reset_states=True)
            
#             predicted_durations = output.get('predicted_durations')
#             regulated_features = output.get('regulated_features')
            
#             if predicted_durations is not None:
#                 pred_total_duration = predicted_durations.sum().item()
#                 required_tokens = int(pred_total_duration * 75)  # 75 tokens per second
#                 required_tokens = min(required_tokens, max_length)  # Limit length
                
#                 logger.info(f"🧠 Infinite ML Predictions:")
#                 logger.info(f"   Predicted duration: {pred_total_duration:.2f}s")
#                 logger.info(f"   Required tokens: {required_tokens}")
                
#                 if required_tokens > 10:
#                     # Prepare for infinite generation
#                     text_context = torch.mean(regulated_features, dim=1)
#                     text_context = model.text_proj(text_context)
                    
#                     # Generate audio tokens with infinite continuous processing
#                     generated_codes = torch.zeros(1, 4, required_tokens, dtype=torch.long, device=device)
                    
#                     logger.info(f"🎵 Generating {required_tokens} tokens with INFINITE processing...")
                    
#                     # Virtual checkpoint positions for monitoring
#                     checkpoint_interval = 75  # ~1 second
#                     checkpoints = list(range(0, required_tokens, checkpoint_interval))
                    
#                     # Generate tokens with infinite continuous processing
#                     generation_steps = min(required_tokens - 1, 200)  # Limit for demo
                    
#                     for t in range(generation_steps):
#                         try:
#                             current_codes = generated_codes[:, :, :t+1]
                            
#                             # Get predictions for next token using infinite processing
#                             gen_logits = model.audio_processor(
#                                 current_codes, 
#                                 text_context,
#                                 reset_state=(t == 0)  # Only reset at start
#                             )
                            
#                             if gen_logits.shape[2] > t:
#                                 next_logits = gen_logits[:, :, t, :]  # [1, 4, vocab_size]
                                
#                                 # Add temperature for natural speech
#                                 temperature = 0.8 + 0.2 * np.random.random()  # Variable temperature
#                                 next_logits = next_logits / temperature
#                                 probs = torch.softmax(next_logits, dim=-1)
                                
#                                 # Sample next tokens
#                                 next_tokens = torch.multinomial(probs.view(-1, 1024), 1).view(1, 4)
                                
#                                 if t + 1 < required_tokens:
#                                     generated_codes[:, :, t + 1] = next_tokens
                                
#                                 # Virtual checkpoint monitoring
#                                 if virtual_checkpoints and t in checkpoints:
#                                     checkpoint_time = t / 75.0
#                                     logger.info(f"     Checkpoint at {checkpoint_time:.1f}s (token {t})")
#                             else:
#                                 break
                                
#                         except Exception as e:
#                             logger.warning(f"     Generation failed at token {t}: {e}")
#                             break
                    
#                     # Get final generated sequence
#                     final_codes = generated_codes[:, :, :generation_steps+1]  # [1, 4, T]
                    
#                     logger.info(f"✅ Generated {final_codes.shape[2]} tokens with infinite processing")
                    
#                     # Decode to audio
#                     from encodec import EncodecModel
                    
#                     codec = EncodecModel.encodec_model_24khz()
#                     codec.set_target_bandwidth(3.0)
#                     codec.eval()
#                     if device.type == 'cuda':
#                         codec = codec.cuda()
                    
#                     logger.info(f"🔊 Decoding infinite generated sequence...")
                    
#                     decoded_audio = codec.decode([(final_codes, None)])
                    
#                     if len(decoded_audio) > 0:
#                         waveform = decoded_audio[0].squeeze().cpu()
                        
#                         actual_duration = len(waveform) / 24000
#                         rms = torch.sqrt(torch.mean(waveform**2)).item()
                        
#                         logger.info(f"✅ INFINITE speech generated successfully!")
#                         logger.info(f"   Duration: {actual_duration:.2f}s")
#                         logger.info(f"   Predicted: {pred_total_duration:.2f}s")
#                         logger.info(f"   Ratio: {actual_duration/pred_total_duration:.2f}x")
#                         logger.info(f"   RMS: {rms:.4f}")
#                         logger.info(f"   Generated tokens: {final_codes.shape[2]}")
                        
#                         # Save generated speech
#                         text_preview = text_input[:30].replace(' ', '_').replace('.', '').replace(',', '')
#                         output_path = f"infinite_speech_{text_preview}_{actual_duration:.1f}s.wav"
#                         torchaudio.save(output_path, waveform.unsqueeze(0), 24000)
                        
#                         logger.info(f"💾 INFINITE generated speech saved: {output_path}")
                        
#                         return {
#                             'success': True,
#                             'audio_file': output_path,
#                             'text': text_input,
#                             'actual_duration': actual_duration,
#                             'predicted_duration': pred_total_duration,
#                             'generated_tokens': final_codes.shape[2],
#                             'rms': rms,
#                             'infinite_processing': True,
#                             'virtual_checkpoints_used': virtual_checkpoints
#                         }
#                     else:
#                         logger.error("❌ No decoded audio from infinite generation")
#                         return None
#                 else:
#                     logger.warning("⚠️  Predicted duration too short for infinite generation")
#                     return None
#             else:
#                 logger.error("❌ No duration predictions from infinite model")
#                 return None
                
#     except Exception as e:
#         logger.error(f"❌ Infinite speech generation failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# def generate_long_infinite_speech(model, tokenizer, text_segments, 
#                                 max_total_duration=120.0):
#     """
#     Generate very long speech by processing multiple text segments continuously
#     """
#     logger.info(f"🎤 Generating LONG INFINITE speech:")
#     logger.info(f"   Text segments: {len(text_segments)}")
#     logger.info(f"   Max total duration: {max_total_duration:.1f}s")
    
#     device = next(model.parameters()).device
#     all_generated_codes = []
#     total_predicted_duration = 0
    
#     try:
#         # Reset infinite states once at the beginning
#         model.reset_infinite_states(1)
        
#         with torch.no_grad():
#             for i, text_segment in enumerate(text_segments):
#                 logger.info(f"   Processing segment {i+1}/{len(text_segments)}: '{text_segment[:50]}...'")
                
#                 # Tokenize segment
#                 text_tokens = tokenizer.encode(text_segment, add_special_tokens=True)
#                 text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device).unsqueeze(0)
                
#                 # Process with continuous state (no reset between segments!)
#                 output = model(text_tokens, audio_tokens=None, reset_states=(i == 0))
                
#                 predicted_durations = output.get('predicted_durations')
#                 regulated_features = output.get('regulated_features')
                
#                 if predicted_durations is not None:
#                     segment_duration = predicted_durations.sum().item()
#                     total_predicted_duration += segment_duration
                    
#                     # Check if we're approaching max duration
#                     if total_predicted_duration > max_total_duration:
#                         logger.info(f"   Stopping at segment {i+1} - max duration reached")
#                         break
                    
#                     required_tokens = int(segment_duration * 75)
#                     required_tokens = min(required_tokens, 150)  # Limit per segment
                    
#                     logger.info(f"     Segment duration: {segment_duration:.2f}s, tokens: {required_tokens}")
                    
#                     # Generate for this segment with continuous processing
#                     text_context = torch.mean(regulated_features, dim=1)
#                     text_context = model.text_proj(text_context)
                    
#                     segment_codes = torch.zeros(1, 4, required_tokens, dtype=torch.long, device=device)
                    
#                     # Generate with continuous state
#                     for t in range(min(required_tokens - 1, 100)):  # Limit for speed
#                         try:
#                             current_codes = segment_codes[:, :, :t+1]
#                             gen_logits = model.audio_processor(
#                                 current_codes, 
#                                 text_context,
#                                 reset_state=False  # NEVER reset - continuous!
#                             )
                            
#                             if gen_logits.shape[2] > t:
#                                 next_logits = gen_logits[:, :, t, :]
#                                 temperature = 0.8
#                                 next_logits = next_logits / temperature
#                                 probs = torch.softmax(next_logits, dim=-1)
#                                 next_tokens = torch.multinomial(probs.view(-1, 1024), 1).view(1, 4)
                                
#                                 if t + 1 < required_tokens:
#                                     segment_codes[:, :, t + 1] = next_tokens
#                         except:
#                             break
                    
#                     # Add to overall sequence
#                     segment_final = segment_codes[:, :, :min(required_tokens, 101)]
#                     all_generated_codes.append(segment_final.squeeze(0))  # [4, T]
                    
#                     logger.info(f"     Generated {segment_final.shape[2]} tokens for segment {i+1}")
        
#         if len(all_generated_codes) == 0:
#             logger.error("❌ No segments generated")
#             return None
        
#         # Concatenate all segments into one long sequence
#         final_long_sequence = torch.cat(all_generated_codes, dim=1)  # [4, T_total]
        
#         logger.info(f"✅ LONG INFINITE sequence generated:")
#         logger.info(f"   Total segments processed: {len(all_generated_codes)}")
#         logger.info(f"   Total tokens: {final_long_sequence.shape[1]:,}")
#         logger.info(f"   Expected duration: {final_long_sequence.shape[1]/75:.1f}s")
#         logger.info(f"   Predicted duration: {total_predicted_duration:.1f}s")
        
#         # Decode long sequence
#         from encodec import EncodecModel
        
#         codec = EncodecModel.encodec_model_24khz()
#         codec.set_target_bandwidth(3.0)
#         codec.eval()
#         if device.type == 'cuda':
#             codec = codec.cuda()
        
#         decode_tokens = final_long_sequence.unsqueeze(0).to(device)  # [1, 4, T_total]
        
#         logger.info(f"🔊 Decoding LONG infinite sequence: {decode_tokens.shape}")
        
#         decoded_audio = codec.decode([(decode_tokens, None)])
        
#         if len(decoded_audio) > 0:
#             waveform = decoded_audio[0].squeeze().cpu()
            
#             actual_duration = len(waveform) / 24000
#             rms = torch.sqrt(torch.mean(waveform**2)).item()
            
#             logger.info(f"✅ LONG INFINITE speech generated!")
#             logger.info(f"   Actual duration: {actual_duration:.1f}s ({actual_duration/60:.1f} minutes)")
#             logger.info(f"   Predicted duration: {total_predicted_duration:.1f}s")
#             logger.info(f"   Ratio: {actual_duration/total_predicted_duration:.2f}x")
#             logger.info(f"   RMS: {rms:.4f}")
            
#             # Save long generated speech
#             output_path = f"infinite_long_speech_{actual_duration:.0f}s.wav"
#             torchaudio.save(output_path, waveform.unsqueeze(0), 24000)
            
#             logger.info(f"💾 LONG INFINITE speech saved: {output_path}")
            
#             return {
#                 'success': True,
#                 'audio_file': output_path,
#                 'segments_processed': len(all_generated_codes),
#                 'actual_duration': actual_duration,
#                 'predicted_duration': total_predicted_duration,
#                 'total_tokens': final_long_sequence.shape[1],
#                 'rms': rms,
#                 'infinite_continuous_processing': True
#             }
#         else:
#             logger.error("❌ No decoded audio from long sequence")
#             return None
            
#     except Exception as e:
#         logger.error(f"❌ Long infinite generation failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# def main():
#     """Test infinite inference capabilities"""
#     logger.info("🎵 INFINITE Inference - Continuous Speech Generation")
#     logger.info("=" * 60)
    
#     # Load infinite model
#     model, tokenizer = load_infinite_model()
#     if model is None:
#         return
    
#     # Test 1: Batch reconstruction
#     logger.info(f"\n📼 Test 1: INFINITE Batch Reconstruction")
#     logger.info("="*40)
    
#     # Get available continuous batches
#     data_dir = Path("continuous_data")
#     if data_dir.exists():
#         batch_dirs = [d.name for d in data_dir.iterdir() 
#                      if d.is_dir() and d.name.startswith('continuous_batch_')]
#         batch_dirs.sort()
        
#         # Test first few batches
#         test_batches = batch_dirs[:2] if len(batch_dirs) >= 2 else batch_dirs
        
#         for batch_name in test_batches:
#             logger.info(f"\n🧪 Testing infinite batch: {batch_name}")
            
#             result = reconstruct_infinite_batch(model, tokenizer, batch_name)
            
#             if result and result['success']:
#                 logger.info(f"✅ Infinite reconstruction successful!")
#                 logger.info(f"   File: {result['audio_file']}")
#                 logger.info(f"   Duration: {result['actual_duration']:.2f}s")
#                 logger.info(f"   Processing: {result.get('infinite_processing', False)}")
    
#     # Test 2: Single text generation
#     logger.info(f"\n🎤 Test 2: INFINITE Text-to-Speech Generation")
#     logger.info("="*40)
    
#     test_texts = [
#         "Dzisiaj będziemy testować nieskończone przetwarzanie mowy.",
#         "Sztuczna inteligencja umożliwia generowanie naturalnej mowy w języku polskim.",
#         "Ten system używa ciągłego przetwarzania bez wewnętrznego dzielenia na fragmenty."
#     ]
    
#     for i, text in enumerate(test_texts):
#         logger.info(f"\n🎤 Test text {i+1}: '{text}'")
        
#         result = generate_infinite_speech(model, tokenizer, text, max_length=200)
        
#         if result and result['success']:
#             logger.info(f"✅ Infinite generation successful!")
#             logger.info(f"   File: {result['audio_file']}")
#             logger.info(f"   Duration: {result['actual_duration']:.2f}s")
#             logger.info(f"   Infinite processing: {result.get('infinite_processing', False)}")
    
#     # Test 3: Long continuous generation
#     logger.info(f"\n🎤 Test 3: LONG INFINITE Continuous Generation")
#     logger.info("="*40)
    
#     long_text_segments = [
#         "Pierwsza część długiego tekstu do wygenerowania.",
#         "Druga część kontynuuje bezpośrednio po pierwszej.",
#         "Trzecia część kończy długą sekwencję mowy.",
#         "Wszystko jest przetwarzane w sposób ciągły bez przerw."
#     ]
    
#     result = generate_long_infinite_speech(model, tokenizer, long_text_segments, 
#                                          max_total_duration=30.0)
    
#     if result and result['success']:
#         logger.info(f"✅ Long infinite generation successful!")
#         logger.info(f"   File: {result['audio_file']}")
#         logger.info(f"   Duration: {result['actual_duration']:.1f}s")
#         logger.info(f"   Segments: {result['segments_processed']}")
#         logger.info(f"   Continuous processing: {result.get('infinite_continuous_processing', False)}")
    
#     logger.info(f"\n🎉 INFINITE inference testing complete!")
#     logger.info(f"📁 Check files: infinite_batch_*.wav, infinite_speech_*.wav, infinite_long_speech_*.wav")
#     logger.info(f"🎯 Infinite continuous processing - no chunking, true continuous flow!")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Infinite Trainer - Simplified Working Version
============================================
Simplified version that focuses on getting training to work first
Then we can add virtual checkpoints complexity later
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

# Import modules - same as before
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


class SimplifiedInfiniteTTSModel(nn.Module):
    """
    Simplified TTS model that definitely works
    Same architecture but simpler forward pass
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_codebooks=4, codebook_size=1024, state_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Use standard modules instead of infinite ones for now
        try:
            from modules import (
                MambaConvTextEncoder, 
                MambaConvAudioProcessor, 
                DurationRegulator,
                AudioStyleExtractor
            )
            self.text_encoder = MambaConvTextEncoder(vocab_size, embed_dim)
            self.duration_regulator = DurationRegulator(
                text_dim=embed_dim, style_dim=64, hidden_dim=64, tokens_per_second=75.0
            )
            self.audio_processor = MambaConvAudioProcessor(hidden_dim, num_codebooks, codebook_size)
            self.style_extractor = AudioStyleExtractor(hidden_dim, 64)
            logger.info("✅ Using standard modules for simplified version")
            
        except ImportError:
            # Fallback to infinite modules but use them simply
            self.text_encoder = InfiniteMambaTextEncoder(vocab_size, embed_dim, num_layers=3, state_size=state_size)
            self.duration_regulator = InfiniteDurationRegulator(
                text_dim=embed_dim, style_dim=64, hidden_dim=128, 
                tokens_per_second=75.0, state_size=state_size
            )
            self.audio_processor = InfiniteMambaAudioProcessor(
                hidden_dim, num_codebooks, codebook_size, num_layers=2, state_size=state_size
            )
            self.style_extractor = InfiniteAudioStyleExtractor(hidden_dim, 64)
            logger.info("✅ Using infinite modules in simplified mode")
        
        # Projections
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        self.default_style = nn.Parameter(torch.randn(64) * 0.01)
        
        logger.info(f"🎯 SimplifiedInfiniteTTSModel: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, text_tokens, audio_tokens=None):
        """
        Simplified forward pass - no complex state management for now
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        try:
            # Text encoding
            text_features = self.text_encoder(text_tokens, return_sequence=True)
            text_context = self.text_encoder(text_tokens, return_sequence=False)
            text_context = self.text_proj(text_context)
            
            # Style extraction
            if audio_tokens is not None:
                with torch.no_grad():
                    B, C, T = audio_tokens.shape
                    audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
                    pseudo_audio = audio_mean.expand(B, self.hidden_dim, min(T, 100))
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
            
        except Exception as e:
            logger.error(f"❌ Model forward pass failed: {e}")
            return None


class SimplifiedInfiniteTrainer:
    """
    Simplified trainer that definitely works
    Focus on basic training first, then add complexity
    """
    def __init__(self, model, tokenizer, data_loader):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        
        logger.info(f"🎯 SimplifiedInfiniteTrainer initialized")
        logger.info(f"   Data: {len(data_loader)} continuous batches")
    
    def simple_loss_computation(self, model_output, batch_data):
        """
        Simple loss computation that definitely works
        Use existing proven loss function
        """
        try:
            # Prepare data in expected format
            text_tokens = batch_data['text_tokens']
            
            # Create chunk_data format expected by existing loss function
            chunk_data = {
                'audio_codes': batch_data['audio_codes'],
                'duration': batch_data['duration'],
                'text': batch_data['full_text']
            }
            
            # Use existing proven loss function
            from losses import compute_combined_loss
            loss_dict = compute_combined_loss(model_output, chunk_data, text_tokens, self.device)
            
            return loss_dict
            
        except Exception as e:
            logger.error(f"❌ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_simple_batch(self, batch_data, optimizer):
        """
        Simple batch training that definitely works
        """
        try:
            # Prepare batch for training
            training_batch = self.data_loader.prepare_batch_for_training(batch_data)
            
            text_tokens = training_batch['text_tokens']
            audio_codes = training_batch['audio_codes']
            duration = training_batch['duration']
            
            logger.debug(f"🔄 Training batch: {duration:.1f}s")
            logger.debug(f"   Text tokens: {text_tokens.shape}")
            logger.debug(f"   Audio codes: {audio_codes.shape}")
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(text_tokens, audio_codes)
            
            if output is None:
                logger.error("❌ Model returned None")
                return None
            
            # Compute loss
            loss_dict = self.simple_loss_computation(output, training_batch)
            
            if loss_dict is not None:
                total_loss = loss_dict['total_loss']
                
                # Check if loss is valid
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning("⚠️  Invalid loss detected")
                    return None
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Add batch info
                loss_dict['batch_info'] = {
                    'duration': duration,
                    'audio_tokens': training_batch['audio_tokens'],
                    'text_tokens': training_batch['text_token_count'],
                    'batch_dir': training_batch.get('batch_dir', 'unknown')
                }
                
                return loss_dict
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Batch training failed: {e}")
            import traceback
            traceback.print_exc()
            optimizer.zero_grad()
            return None
    
    def train(self, steps=2000, learning_rate=2e-3):
        """
        Simple training loop that definitely works
        """
        logger.info(f"🚀 Starting SIMPLIFIED INFINITE training for {steps} steps")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Focus: Get basic training working first")
        
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
        logger.info(f"⏱️  Starting simplified training...")
        
        for step in range(steps):
            try:
                # Get random continuous batch
                batch_data = self.data_loader.get_random_continuous_batch()
                
                if batch_data is None:
                    logger.warning(f"⚠️  Step {step}: No batch data")
                    failed_steps += 1
                    continue
                
                # Train on batch
                loss_dict = self.train_simple_batch(batch_data, optimizer)
                
                if loss_dict is not None:
                    # Track metrics
                    total_loss = loss_dict['total_loss'].item()
                    current_accuracy = loss_dict['accuracy']
                    current_duration_accuracy = loss_dict['duration_accuracy']
                    
                    losses.append(total_loss)
                    accuracies.append(current_accuracy)
                    duration_accuracies.append(current_duration_accuracy)
                    
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                    if current_duration_accuracy > best_duration_accuracy:
                        best_duration_accuracy = current_duration_accuracy
                    
                    successful_steps += 1
                    
                    # Enhanced logging
                    if (step % 100 == 0 or current_accuracy > 0.1 or 
                        current_duration_accuracy > 0.3 or step < 10):
                        
                        logger.info(f"Step {step:4d}: Loss={total_loss:.4f}, "
                                  f"Acc={current_accuracy:.4f}, DurAcc={current_duration_accuracy:.4f}")
                        
                        batch_info = loss_dict['batch_info']
                        logger.info(f"         Batch: {batch_info['duration']:.1f}s, "
                                  f"{batch_info['audio_tokens']:,} tokens")
                        logger.info(f"         Dir: {batch_info['batch_dir']}")
                    
                    # Success detection
                    if current_accuracy > 0.5:
                        logger.info(f"🎉 GREAT PROGRESS! Accuracy {current_accuracy:.4f}")
                    if current_duration_accuracy > 0.5:
                        logger.info(f"🎉 GREAT DURATION! Duration Accuracy {current_duration_accuracy:.4f}")
                    
                    # Early success check
                    if (best_accuracy > 0.3 and best_duration_accuracy > 0.4 and 
                        step > 1000):
                        logger.info(f"🎉 SIMPLIFIED TRAINING SUCCESS!")
                        break
                        
                else:
                    failed_steps += 1
                    if step < 10:
                        logger.warning(f"⚠️  Step {step}: Training failed")
                    
                # Memory cleanup
                if step % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.error(f"❌ Step {step} failed: {e}")
                if step < 10:
                    import traceback
                    traceback.print_exc()
                failed_steps += 1
                continue
        
        # Final results
        success_rate = successful_steps / (successful_steps + failed_steps) * 100 if (successful_steps + failed_steps) > 0 else 0
        
        final_loss = losses[-1] if losses else 999.0
        final_acc = accuracies[-1] if accuracies else 0.0
        final_dur_acc = duration_accuracies[-1] if duration_accuracies else 0.0
        
        logger.info(f"\n🎉 SIMPLIFIED training completed!")
        logger.info(f"   Successful steps: {successful_steps}/{steps} ({success_rate:.1f}%)")
        logger.info(f"   Best audio accuracy: {best_accuracy:.4f}")
        logger.info(f"   Best duration accuracy: {best_duration_accuracy:.4f}")
        logger.info(f"   Final - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}, DurAcc: {final_dur_acc:.4f}")
        
        # Save model if any success
        if successful_steps > 0 and (best_accuracy > 0.05 or best_duration_accuracy > 0.2):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'final_loss': final_loss,
                'final_accuracy': final_acc,
                'final_duration_accuracy': final_dur_acc,
                'best_accuracy': best_accuracy,
                'best_duration_accuracy': best_duration_accuracy,
                'successful_steps': successful_steps,
                'vocab_size': self.tokenizer.get_vocab_size(),
                'simplified_infinite_training': True,
                'success_rate': success_rate
            }, 'simplified_infinite_model.pt')
            
            logger.info("💾 SIMPLIFIED model saved as 'simplified_infinite_model.pt'")
            return True
        else:
            logger.warning("⚠️  No successful training steps")
            return False


def debug_single_batch():
    """Debug function to test single batch processing"""
    logger.info("🔍 Debugging single batch processing...")
    
    try:
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        data_loader = InfiniteDataLoader("continuous_data", device)
        
        if len(data_loader) == 0:
            logger.error("❌ No data loaded")
            return
        
        # Get one batch
        batch_data = data_loader.get_random_continuous_batch()
        if batch_data is None:
            logger.error("❌ No batch data")
            return
        
        logger.info(f"✅ Got batch: {batch_data['duration']:.1f}s")
        
        # Prepare for training
        training_batch = data_loader.prepare_batch_for_training(batch_data)
        logger.info(f"✅ Prepared training batch:")
        logger.info(f"   Text tokens: {training_batch['text_tokens'].shape}")
        logger.info(f"   Audio codes: {training_batch['audio_codes'].shape}")
        
        # Create simple model
        model = SimplifiedInfiniteTTSModel(vocab_size).to(device)
        logger.info(f"✅ Created model")
        
        # Test forward pass
        with torch.no_grad():
            output = model(training_batch['text_tokens'], training_batch['audio_codes'])
        
        if output is not None:
            logger.info(f"✅ Forward pass successful!")
            for key, value in output.items():
                if torch.is_tensor(value):
                    logger.info(f"   {key}: {value.shape}")
                else:
                    logger.info(f"   {key}: {type(value)}")
        else:
            logger.error("❌ Forward pass failed")
            return
        
        # Test loss computation
        trainer = SimplifiedInfiniteTrainer(model, tokenizer, data_loader)
        loss_dict = trainer.simple_loss_computation(output, training_batch)
        
        if loss_dict is not None:
            logger.info(f"✅ Loss computation successful!")
            logger.info(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
            logger.info(f"   Accuracy: {loss_dict['accuracy']:.4f}")
            logger.info(f"   Duration accuracy: {loss_dict['duration_accuracy']:.4f}")
        else:
            logger.error("❌ Loss computation failed")
            return
        
        logger.info("✅ Single batch debug successful - training should work!")
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function for simplified infinite training"""
    logger.info("🎯 SIMPLIFIED INFINITE TTS Training")
    logger.info("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check continuous data  
    if not Path("continuous_data").exists():
        logger.error("❌ continuous_data directory not found!")
        return
    
    # Debug single batch first
    debug_single_batch()
    
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
        logger.info(f"\n📊 Data Statistics:")
        logger.info(f"   Total batches: {stats['total_batches']}")
        logger.info(f"   Total duration: {stats['total_duration']:.1f}s")
        logger.info(f"   Average batch duration: {stats['avg_duration']:.1f}s")
        
        # Create simplified model
        model = SimplifiedInfiniteTTSModel(vocab_size).to(device)
        
        # Create simplified trainer
        trainer = SimplifiedInfiniteTrainer(model, tokenizer, data_loader)
        
        logger.info(f"\n🚀 Starting SIMPLIFIED training...")
        
        # Train with simplified system
        success = trainer.train(steps=2000, learning_rate=2e-3)
        
        if success:
            logger.info("✅ SIMPLIFIED training successful!")
        else:
            logger.warning("⚠️  Training needs investigation")
    
    except Exception as e:
        logger.error(f"❌ SIMPLIFIED training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()