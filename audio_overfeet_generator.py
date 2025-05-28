#!/usr/bin/env python3
"""
Audio Generator for Overfitting Check
====================================
Generates new audio from training texts to check for overfitting
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import existing components
try:
    from nucleotide_tokenizer import NucleotideTokenizer
    from sequential_trainer import Enhanced8CodebookTTSModel, Enhanced8CodebookDataLoader
    logger.info("✅ Imported required components")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    exit(1)


class AudioGenerator:
    """Generator to create audio from text using trained model"""
    
    def __init__(self, model_path="enhanced_8codebook_model.pt", data_dir="no_overlap_data"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = NucleotideTokenizer()
        self.data_loader = Enhanced8CodebookDataLoader(data_dir, self.device)
        
        # Load trained model
        self.model = self._load_model(model_path)
        if self.model is None:
            logger.error("❌ Failed to load model")
            return
            
        self.model.eval()  # Set to evaluation mode
        logger.info(f"🎵 AudioGenerator ready on {self.device}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            if not Path(model_path).exists():
                logger.error(f"❌ Model file not found: {model_path}")
                return None
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Get model config
            config = checkpoint.get('model_config', {})
            vocab_size = checkpoint.get('vocab_size', self.tokenizer.get_vocab_size())
            
            # Create model with same architecture
            model = Enhanced8CodebookTTSModel(
                vocab_size=vocab_size,
                embed_dim=config.get('embed_dim', 384),
                hidden_dim=config.get('hidden_dim', 512),
                num_codebooks=config.get('num_codebooks', 8),
                codebook_size=config.get('codebook_size', 1024)
            ).to(self.device)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"✅ Model loaded from {model_path}")
            logger.info(f"   Best accuracy: {checkpoint.get('best_accuracy', 'unknown')}")
            logger.info(f"   Best duration accuracy: {checkpoint.get('best_duration_accuracy', 'unknown')}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return None
    
    def generate_audio_tokens(self, text: str, max_length: int = 2000,  # INCREASED
                            temperature: float = 0.8, top_k: int = 50) -> Optional[torch.Tensor]:
        """Generate audio tokens from text using trained model - FIXED VERSION"""
        try:
            if self.model is None:
                logger.error("❌ No model loaded")
                return None
            
            # Tokenize text
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if text_tokens is None or len(text_tokens) == 0:
                logger.warning(f"⚠️ Failed to tokenize: '{text}'")
                return None
            
            # Prepare input
            text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            logger.info(f"🎯 Generating audio for: '{text[:50]}...'")
            logger.info(f"   Text tokens shape: {text_tokens.shape}")
            
            with torch.no_grad():
                # Get text features and predicted durations
                output = self.model(text_tokens, audio_tokens=None)
                
                predicted_durations = output['predicted_durations']  # [B, L]
                duration_tokens = output['duration_tokens']        # [B, L]
                text_features = output['text_features']            # [B, L, D]
                
                logger.info(f"   Predicted durations: {predicted_durations.shape}")
                logger.info(f"   Duration range: {predicted_durations.min():.3f}s - {predicted_durations.max():.3f}s")
                
                # 🔧 FIX: Use ACTUAL predicted durations with intelligent scaling
                if predicted_durations is not None:
                    # Calculate total expected length from predicted durations
                    total_predicted_duration = predicted_durations.sum().item()  # Total seconds
                    
                    # INTELLIGENT SCALING: If predicted duration is too long, scale it down
                    text_length = text_tokens.shape[1]
                    reasonable_duration_per_token = 0.08  # ~80ms per token (reasonable for Polish)
                    reasonable_total = text_length * reasonable_duration_per_token
                    
                    if total_predicted_duration > reasonable_total * 2:  # If >2x reasonable
                        scale_factor = reasonable_total * 1.5 / total_predicted_duration  # Scale to 1.5x reasonable
                        total_predicted_duration = total_predicted_duration * scale_factor
                        logger.info(f"   🔧 SCALED: Duration scaled by {scale_factor:.3f} to {total_predicted_duration:.2f}s")
                    
                    expected_length = int(total_predicted_duration * 75)  # 75 tokens per second
                    expected_length = min(expected_length, max_length)  # Safety limit
                    
                    logger.info(f"   🎯 FIXED: Using predicted total duration: {total_predicted_duration:.2f}s")
                    logger.info(f"   Expected audio length: {expected_length} tokens ({expected_length/75:.2f}s)")
                else:
                    # Fallback to old method if no predictions
                    text_length = text_tokens.shape[1]
                    expected_length = min(text_length * 6, max_length)
                    logger.info(f"   ⚠️ Fallback: Using fixed multiplier method")
                    logger.info(f"   Expected audio length: {expected_length} tokens ({expected_length/75:.2f}s)")
                
                # Safety check
                if expected_length <= 0:
                    expected_length = 300  # 4 second fallback
                    logger.warning(f"   ⚠️ Expected length was 0, using fallback: {expected_length}")
                
                # Initialize audio generation
                batch_size = text_tokens.shape[0]
                generated_tokens = torch.zeros(batch_size, 8, expected_length, 
                                            dtype=torch.long, device=self.device)
                
                # Get text context for audio processor
                text_context = self.model.text_encoder(text_tokens, return_sequence=False)
                text_context = self.model.text_proj(text_context)
                
                # 🎯 IMPROVED: Generate using predicted durations for timing
                current_pos = 0
                text_length = text_tokens.shape[1]
                
                # Map each text token to audio positions using predicted durations
                for t_idx in range(text_length):
                    if current_pos >= expected_length:
                        break
                    
                    # Get predicted duration for this text token with scaling
                    if predicted_durations is not None:
                        token_duration = predicted_durations[0, t_idx].item()  # seconds
                        
                        # Apply same scaling as above if needed
                        if 'scale_factor' in locals():
                            token_duration = token_duration * scale_factor
                        
                        dur_tokens = max(1, int(token_duration * 75))  # Convert to tokens
                    else:
                        # Fallback
                        dur_tokens = duration_tokens[0, t_idx].item() if duration_tokens is not None else 6
                    
                    # Limit to remaining space
                    dur_tokens = min(dur_tokens, expected_length - current_pos)
                    
                    if dur_tokens <= 0:
                        continue
                    
                    # Generate tokens for this duration
                    for pos in range(current_pos, current_pos + dur_tokens):
                        if pos >= expected_length:
                            break
                        
                        # Create partial sequence for context
                        if pos > 0:
                            partial_audio = generated_tokens[:, :, :pos]
                        else:
                            # Start with random tokens for first position
                            for c in range(8):
                                generated_tokens[0, c, 0] = torch.randint(0, 1024, (1,), device=self.device)
                            continue
                        
                        # Get logits from model
                        try:
                            logits = self.model.audio_processor(partial_audio, text_context)
                            # logits shape: [B, 8, T, codebook_size]
                            
                            if logits.shape[2] > pos:
                                current_logits = logits[0, :, pos, :]  # [8, codebook_size]
                            else:
                                current_logits = logits[0, :, -1, :]   # Use last position
                            
                            # Sample from logits for each codebook with improved diversity
                            for c in range(8):
                                logits_c = current_logits[c] / temperature
                                
                                # Top-k sampling with better diversity
                                if top_k > 0:
                                    top_k_logits, top_k_indices = torch.topk(logits_c, min(top_k, logits_c.size(-1)))
                                    
                                    # Add small noise for more diversity
                                    noise = torch.randn_like(top_k_logits) * 0.1
                                    top_k_logits = top_k_logits + noise
                                    
                                    probs = F.softmax(top_k_logits, dim=-1)
                                    sampled_idx = torch.multinomial(probs, 1)
                                    token = top_k_indices[sampled_idx].item()
                                else:
                                    probs = F.softmax(logits_c, dim=-1)
                                    token = torch.multinomial(probs, 1).item()
                                
                                generated_tokens[0, c, pos] = token
                        
                        except Exception as e:
                            logger.debug(f"Generation error at pos {pos}: {e}")
                            # Fallback: diversified random tokens
                            for c in range(8):
                                # Use different ranges for different codebooks for diversity
                                start_range = c * 128
                                end_range = min(start_range + 128, 1024)
                                token = torch.randint(start_range, end_range, (1,), device=self.device).item()
                                generated_tokens[0, c, pos] = token
                    
                    current_pos += dur_tokens
                
                # Fill remaining positions if any (shouldn't happen with good duration prediction)
                if current_pos < expected_length:
                    logger.info(f"   📝 Filling remaining {expected_length - current_pos} positions")
                    for pos in range(current_pos, expected_length):
                        for c in range(8):
                            # Use diverse random tokens
                            start_range = c * 128
                            end_range = min(start_range + 128, 1024)
                            token = torch.randint(start_range, end_range, (1,), device=self.device).item()
                            generated_tokens[0, c, pos] = token
                
                # Return generated tokens [8, T]
                final_tokens = generated_tokens[0, :, :expected_length]
                logger.info(f"✅ Generated audio tokens shape: {final_tokens.shape}")
                logger.info(f"   🎵 Final duration: {final_tokens.shape[1]/75:.2f}s")
                
                return final_tokens
                
        except Exception as e:
            logger.error(f"❌ Audio generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_with_original(self, chunk_data: Dict) -> Dict:
        """Compare generated audio with original chunk audio"""
        try:
            text = chunk_data['text']
            original_audio = chunk_data['audio_codes']  # [C, T]
            
            # Generate new audio from text
            generated_audio = self.generate_audio_tokens(text)
            
            if generated_audio is None:
                return {'error': 'Failed to generate audio'}
            
            # Ensure same number of codebooks
            if original_audio.shape[0] != 8:
                if original_audio.shape[0] < 8:
                    padding = torch.zeros(8 - original_audio.shape[0], original_audio.shape[1], 
                                        dtype=original_audio.dtype, device=original_audio.device)
                    original_audio = torch.cat([original_audio, padding], dim=0)
                else:
                    original_audio = original_audio[:8, :]
            
            # Compare statistics
            comparison = {
                'text': text,
                'original_shape': original_audio.shape,
                'generated_shape': generated_audio.shape,
                'original_duration': original_audio.shape[1] / 75.0,
                'generated_duration': generated_audio.shape[1] / 75.0,
                'chunk_duration': chunk_data.get('duration', 0),
            }
            
            # Token statistics per codebook
            original_stats = {}
            generated_stats = {}
            
            for c in range(8):
                orig_tokens = original_audio[c, :]
                gen_tokens = generated_audio[c, :]
                
                original_stats[f'codebook_{c}'] = {
                    'min': orig_tokens.min().item(),
                    'max': orig_tokens.max().item(),
                    'mean': orig_tokens.float().mean().item(),
                    'unique': len(torch.unique(orig_tokens)),
                }
                
                generated_stats[f'codebook_{c}'] = {
                    'min': gen_tokens.min().item(),
                    'max': gen_tokens.max().item(),
                    'mean': gen_tokens.float().mean().item(),
                    'unique': len(torch.unique(gen_tokens)),
                }
            
            comparison['original_stats'] = original_stats
            comparison['generated_stats'] = generated_stats
            
            # Similarity metrics (simple)
            min_length = min(original_audio.shape[1], generated_audio.shape[1])
            if min_length > 0:
                orig_sample = original_audio[:, :min_length]
                gen_sample = generated_audio[:, :min_length]
                
                # Token exact match rate
                exact_matches = (orig_sample == gen_sample).float().mean().item()
                comparison['exact_match_rate'] = exact_matches
                
                # Distribution similarity (per codebook)
                distribution_similarity = []
                for c in range(8):
                    orig_hist = torch.bincount(orig_sample[c], minlength=1024)
                    gen_hist = torch.bincount(gen_sample[c], minlength=1024)
                    
                    # Cosine similarity of histograms
                    orig_hist = orig_hist.float()
                    gen_hist = gen_hist.float()
                    
                    if orig_hist.norm() > 0 and gen_hist.norm() > 0:
                        cos_sim = F.cosine_similarity(orig_hist.unsqueeze(0), gen_hist.unsqueeze(0)).item()
                    else:
                        cos_sim = 0.0
                    
                    distribution_similarity.append(cos_sim)
                
                comparison['distribution_similarity'] = {
                    'per_codebook': distribution_similarity,
                    'average': np.mean(distribution_similarity)
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Comparison failed: {e}")
            return {'error': str(e)}
    
    def test_overfitting(self, num_samples: int = 10, save_results: bool = True) -> Dict:
        """Test for overfitting by comparing generated vs original audio"""
        logger.info(f"🔍 Testing overfitting with {num_samples} samples...")
        
        if not self.data_loader.chunks:
            logger.error("❌ No chunks available for testing")
            return {'error': 'No chunks available'}
        
        results = {
            'num_samples': num_samples,
            'successful_generations': 0,
            'failed_generations': 0,
            'comparisons': [],
            'summary': {}
        }
        
        # Get random chunks for testing
        test_chunks = random.sample(self.data_loader.chunks, min(num_samples, len(self.data_loader.chunks)))
        
        exact_match_rates = []
        distribution_similarities = []
        duration_accuracies = []
        
        for i, chunk_data in enumerate(test_chunks):
            logger.info(f"📊 Testing sample {i+1}/{num_samples}")
            logger.info(f"   Text: '{chunk_data['text'][:60]}...'")
            
            comparison = self.compare_with_original(chunk_data)
            
            if 'error' in comparison:
                results['failed_generations'] += 1
                logger.warning(f"   ❌ Failed: {comparison['error']}")
                continue
            
            results['successful_generations'] += 1
            results['comparisons'].append(comparison)
            
            # Collect metrics
            if 'exact_match_rate' in comparison:
                exact_match_rates.append(comparison['exact_match_rate'])
                logger.info(f"   Exact match rate: {comparison['exact_match_rate']:.4f}")
            
            if 'distribution_similarity' in comparison:
                avg_dist_sim = comparison['distribution_similarity']['average']
                distribution_similarities.append(avg_dist_sim)
                logger.info(f"   Distribution similarity: {avg_dist_sim:.4f}")
            
            # Duration accuracy
            orig_dur = comparison['original_duration']
            gen_dur = comparison['generated_duration']
            chunk_dur = comparison['chunk_duration']
            
            if chunk_dur > 0:
                orig_accuracy = 1.0 - abs(orig_dur - chunk_dur) / chunk_dur
                gen_accuracy = 1.0 - abs(gen_dur - chunk_dur) / chunk_dur
                duration_accuracies.append({
                    'original': max(0, orig_accuracy),
                    'generated': max(0, gen_accuracy)
                })
                logger.info(f"   Duration - Original: {orig_dur:.2f}s, Generated: {gen_dur:.2f}s, Expected: {chunk_dur:.2f}s")
        
        # Calculate summary statistics
        if exact_match_rates:
            results['summary']['exact_match'] = {
                'mean': np.mean(exact_match_rates),
                'std': np.std(exact_match_rates),
                'min': np.min(exact_match_rates),
                'max': np.max(exact_match_rates)
            }
        
        if distribution_similarities:
            results['summary']['distribution_similarity'] = {
                'mean': np.mean(distribution_similarities),
                'std': np.std(distribution_similarities),
                'min': np.min(distribution_similarities),
                'max': np.max(distribution_similarities)
            }
        
        if duration_accuracies:
            orig_acc = [d['original'] for d in duration_accuracies]
            gen_acc = [d['generated'] for d in duration_accuracies]
            
            results['summary']['duration_accuracy'] = {
                'original': {
                    'mean': np.mean(orig_acc),
                    'std': np.std(orig_acc)
                },
                'generated': {
                    'mean': np.mean(gen_acc),
                    'std': np.std(gen_acc)
                }
            }
        
        # Overfitting assessment
        overfitting_score = 0.0
        overfitting_indicators = []
        
        if exact_match_rates:
            mean_exact_match = np.mean(exact_match_rates)
            if mean_exact_match > 0.8:
                overfitting_score += 0.4
                overfitting_indicators.append(f"High exact match rate: {mean_exact_match:.3f}")
            elif mean_exact_match > 0.5:
                overfitting_score += 0.2
                overfitting_indicators.append(f"Moderate exact match rate: {mean_exact_match:.3f}")
        
        if distribution_similarities:
            mean_dist_sim = np.mean(distribution_similarities)
            if mean_dist_sim > 0.9:
                overfitting_score += 0.3
                overfitting_indicators.append(f"Very high distribution similarity: {mean_dist_sim:.3f}")
            elif mean_dist_sim > 0.7:
                overfitting_score += 0.1
                overfitting_indicators.append(f"High distribution similarity: {mean_dist_sim:.3f}")
        
        results['overfitting_assessment'] = {
            'score': overfitting_score,
            'level': 'High' if overfitting_score > 0.5 else 'Moderate' if overfitting_score > 0.2 else 'Low',
            'indicators': overfitting_indicators
        }
        
        # Save results
        if save_results:
            with open('overfitting_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("💾 Results saved to 'overfitting_test_results.json'")
        
        # Print summary
        logger.info(f"\n📊 Overfitting Test Results:")
        logger.info(f"   Successful generations: {results['successful_generations']}/{num_samples}")
        
        if 'exact_match' in results['summary']:
            em = results['summary']['exact_match']
            logger.info(f"   Exact match rate: {em['mean']:.4f} ± {em['std']:.4f}")
        
        if 'distribution_similarity' in results['summary']:
            ds = results['summary']['distribution_similarity']
            logger.info(f"   Distribution similarity: {ds['mean']:.4f} ± {ds['std']:.4f}")
        
        assessment = results['overfitting_assessment']
        logger.info(f"   🎯 Overfitting level: {assessment['level']} (score: {assessment['score']:.2f})")
        
        for indicator in assessment['indicators']:
            logger.info(f"      • {indicator}")
        
        return results
    
    def generate_sample_audios(self, num_samples: int = 5) -> None:
        """Generate sample audios and save them for inspection"""
        logger.info(f"🎵 Generating {num_samples} sample audios...")
        
        if not self.data_loader.chunks:
            logger.error("❌ No chunks available")
            return
        
        sample_chunks = random.sample(self.data_loader.chunks, min(num_samples, len(self.data_loader.chunks)))
        
        generated_audios = []
        
        for i, chunk_data in enumerate(sample_chunks):
            text = chunk_data['text']
            logger.info(f"🎯 Sample {i+1}: '{text[:50]}...'")
            
            # Generate audio
            generated_tokens = self.generate_audio_tokens(text)
            
            if generated_tokens is not None:
                generated_audios.append({
                    'text': text,
                    'audio_tokens': generated_tokens.cpu(),
                    'original_audio_tokens': chunk_data['audio_codes'].cpu() if 'audio_codes' in chunk_data else None,
                    'duration': generated_tokens.shape[1] / 75.0,
                    'chunk_info': {
                        'batch_dir': chunk_data.get('batch_dir', 'unknown'),
                        'original_duration': chunk_data.get('duration', 0)
                    }
                })
                logger.info(f"✅ Generated audio: {generated_tokens.shape} ({generated_tokens.shape[1]/75:.2f}s)")
            else:
                logger.warning(f"❌ Failed to generate audio for sample {i+1}")
        
        if generated_audios:
            # Save generated audios
            output_data = {
                'generated_samples': generated_audios,
                'generation_info': {
                    'model': 'Enhanced8CodebookTTSModel',
                    'num_samples': len(generated_audios),
                    'codebook_info': {
                        'num_codebooks': 8,
                        'codebook_size': 1024,
                        'sample_rate': 24000
                    }
                }
            }
            
            torch.save(output_data, 'generated_sample_audios.pt')
            logger.info(f"💾 Generated {len(generated_audios)} sample audios saved to 'generated_sample_audios.pt'")
            
            # Save info file
            with open('generated_sample_info.txt', 'w', encoding='utf-8') as f:
                f.write("Generated Sample Audios Information\n")
                f.write("=" * 50 + "\n\n")
                
                for i, sample in enumerate(generated_audios):
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"  Text: {sample['text']}\n")
                    f.write(f"  Generated duration: {sample['duration']:.2f}s\n")
                    f.write(f"  Original duration: {sample['chunk_info']['original_duration']:.2f}s\n")
                    f.write(f"  Audio shape: {sample['audio_tokens'].shape}\n")
                    f.write(f"  Batch: {sample['chunk_info']['batch_dir']}\n\n")
            
            logger.info("📄 Sample info saved to 'generated_sample_info.txt'")
        else:
            logger.warning("⚠️ No samples were successfully generated")


def main():
    """Main function for overfitting testing"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Audio Generator - Overfitting Test')
    parser.add_argument('--model', '-m', type=str, default="enhanced_8codebook_model.pt",
                       help='Path to the trained model file (default: enhanced_8codebook_model.pt)')
    parser.add_argument('--data', '-d', type=str, default="no_overlap_data",
                       help='Path to the data directory (default: no_overlap_data)')
    parser.add_argument('--samples', '-s', type=int, default=10,
                       help='Number of samples for overfitting test (default: 10)')
    parser.add_argument('--audio-samples', '-a', type=int, default=5,
                       help='Number of audio samples to generate (default: 5)')
    
    args = parser.parse_args()
    
    logger.info("🔍 Audio Generator - Overfitting Test")
    logger.info("=" * 50)
    logger.info(f"📁 Model: {args.model}")
    logger.info(f"📁 Data: {args.data}")
    logger.info(f"🔢 Test samples: {args.samples}")
    logger.info(f"🎵 Audio samples: {args.audio_samples}")
    
    # Check if model exists
    model_path = args.model
    if not Path(model_path).exists():
        logger.error(f"❌ Model file not found: {model_path}")
        logger.error("   Please train the model first using sequential_trainer.py")
        return
    
    # Check data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"❌ Data directory not found: {data_path}")
        return
    
    try:
        # Create generator
        generator = AudioGenerator(model_path, args.data)
        
        if generator.model is None:
            logger.error("❌ Failed to initialize generator")
            return
        
        # Test overfitting
        logger.info("\n🔍 Starting overfitting test...")
        results = generator.test_overfitting(num_samples=args.samples)
        
        # Generate sample audios
        logger.info("\n🎵 Generating sample audios...")
        generator.generate_sample_audios(num_samples=args.audio_samples)
        
        logger.info("\n✅ Overfitting test completed!")
        logger.info("📁 Check the following files:")
        logger.info("   • overfitting_test_results.json - Detailed test results")
        logger.info("   • generated_sample_audios.pt - Generated audio samples")
        logger.info("   • generated_sample_info.txt - Sample information")
        
        # Final assessment
        assessment = results.get('overfitting_assessment', {})
        overfitting_level = assessment.get('level', 'Unknown')
        
        if overfitting_level == 'High':
            logger.warning("⚠️  HIGH OVERFITTING detected!")
            logger.warning("   Model may be memorizing training data")
            logger.warning("   Consider: more data, regularization, or different architecture")
        elif overfitting_level == 'Moderate':
            logger.info("📊 MODERATE overfitting detected")
            logger.info("   Model shows some memorization but may still generalize")
        else:
            logger.info("✅ LOW overfitting - Model appears to generalize well")
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()