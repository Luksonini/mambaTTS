#!/usr/bin/env python3
"""
Debug Embedding Sizes - Find the Index Out of Range Issue
========================================================
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_embedding_sizes():
    """Check all embedding table sizes"""
    logger.info("🔍 Checking embedding sizes in modules...")
    
    try:
        from modules import (
            MambaConvTextEncoder, 
            MambaConvAudioProcessor, 
            DurationRegulator,
            AudioStyleExtractor
        )
        from nucleotide_tokenizer import NucleotideTokenizer
        
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        logger.info(f"📖 Vocab size: {vocab_size}")
        
        # Check text encoder
        logger.info("🔍 Analyzing MambaConvTextEncoder...")
        text_encoder = MambaConvTextEncoder(vocab_size, 128)
        
        # Check token embedding
        if hasattr(text_encoder, 'token_embedding'):
            logger.info(f"   Token embedding: {text_encoder.token_embedding.num_embeddings} embeddings")
        
        # Check position embedding  
        if hasattr(text_encoder, 'pos_embedding'):
            logger.info(f"   Position embedding: {text_encoder.pos_embedding.num_embeddings} positions")
        
        # Check audio processor
        logger.info("🔍 Analyzing MambaConvAudioProcessor...")
        audio_processor = MambaConvAudioProcessor(256, 4, 1024)
        
        # Check token embeddings for each codebook
        if hasattr(audio_processor, 'token_embeddings'):
            for i, emb in enumerate(audio_processor.token_embeddings):
                logger.info(f"   Codebook {i} embedding: {emb.num_embeddings} tokens")
        
        # Check position embedding
        if hasattr(audio_processor, 'pos_embedding'):
            logger.info(f"   Audio position embedding: {audio_processor.pos_embedding.num_embeddings} positions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to check embedding sizes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_safe_sizes():
    """Test with known safe sizes"""
    logger.info("🧪 Testing with safe sequence sizes...")
    
    try:
        from infinite_data_loader import InfiniteDataLoader
        from nucleotide_tokenizer import NucleotideTokenizer
        from modules import (
            MambaConvTextEncoder, 
            MambaConvAudioProcessor, 
            DurationRegulator,
            AudioStyleExtractor
        )
        import torch.nn as nn
        
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        # Load data
        data_loader = InfiniteDataLoader("continuous_data", 'cpu')
        batch = data_loader.get_random_continuous_batch()
        training_batch = data_loader.prepare_batch_for_training(batch)
        
        # Get original sizes
        orig_text_len = training_batch['text_tokens'].shape[1]
        orig_audio_len = training_batch['audio_codes'].shape[2]
        
        logger.info(f"📏 Original sizes:")
        logger.info(f"   Text: {orig_text_len} tokens")
        logger.info(f"   Audio: {orig_audio_len} tokens")
        
        # Test with progressively smaller sizes
        test_sizes = [
            (100, 500),   # Very safe
            (200, 1000),  # Safe
            (300, 1500),  # Medium
            (400, 2000),  # Large
            (500, 2500),  # Very large
        ]
        
        class SafeTestModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.text_encoder = MambaConvTextEncoder(vocab_size, 128)
                self.audio_processor = MambaConvAudioProcessor(256, 4, 1024)
                self.duration_regulator = DurationRegulator(128, 64, 64, 75.0)
                self.style_extractor = AudioStyleExtractor(256, 64)
                self.text_proj = nn.Linear(128, 256)
                self.default_style = nn.Parameter(torch.randn(64) * 0.01)
            
            def forward(self, text_tokens, audio_tokens):
                try:
                    B = text_tokens.shape[0]
                    
                    # Test text encoder first
                    logger.info(f"   Testing text encoder with {text_tokens.shape}")
                    text_features = self.text_encoder(text_tokens, return_sequence=True)
                    text_context = self.text_encoder(text_tokens, return_sequence=False)
                    logger.info(f"   ✅ Text encoder OK")
                    
                    # Test style extractor
                    logger.info(f"   Testing style extractor with audio {audio_tokens.shape}")
                    B, C, T = audio_tokens.shape
                    audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
                    pseudo_audio = audio_mean.expand(B, 256, min(T, 100))
                    style_embedding = self.style_extractor(pseudo_audio)
                    logger.info(f"   ✅ Style extractor OK")
                    
                    # Test duration regulator
                    logger.info(f"   Testing duration regulator")
                    regulated_features, predicted_durations, duration_tokens, duration_confidence = \
                        self.duration_regulator(text_features, style_embedding)
                    logger.info(f"   ✅ Duration regulator OK")
                    
                    # Test audio processor
                    logger.info(f"   Testing audio processor with {audio_tokens.shape}")
                    regulated_context = torch.mean(self.text_proj(text_context), dim=1)
                    audio_logits = self.audio_processor(audio_tokens, regulated_context)
                    logger.info(f"   ✅ Audio processor OK")
                    
                    return {
                        'logits': audio_logits,
                        'predicted_durations': predicted_durations,
                        'duration_tokens': duration_tokens,
                        'duration_confidence': duration_confidence
                    }
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed at component: {e}")
                    raise e
        
        model = SafeTestModel(vocab_size)
        
        for text_len, audio_len in test_sizes:
            logger.info(f"\n🧪 Testing with text={text_len}, audio={audio_len}")
            
            # Create truncated tensors
            text_truncated = training_batch['text_tokens'][:, :min(text_len, orig_text_len)]
            audio_truncated = training_batch['audio_codes'][:, :, :min(audio_len, orig_audio_len)]
            
            logger.info(f"   Actual sizes: text={text_truncated.shape}, audio={audio_truncated.shape}")
            
            try:
                with torch.no_grad():
                    output = model(text_truncated, audio_truncated)
                
                if output is not None:
                    logger.info(f"   ✅ SUCCESS with text={text_len}, audio={audio_len}")
                    
                    # Try to find the maximum working size
                    if text_len >= 400 and audio_len >= 2000:
                        logger.info(f"🎉 Found good working size: text={text_len}, audio={audio_len}")
                        return text_len, audio_len
                else:
                    logger.warning(f"   ⚠️  Model returned None")
                    
            except Exception as e:
                logger.error(f"   ❌ FAILED with text={text_len}, audio={audio_len}: {e}")
                continue
        
        logger.warning("⚠️  All test sizes failed")
        return None
        
    except Exception as e:
        logger.error(f"❌ Safe size testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debug function"""
    logger.info("🔍 DEBUG: Finding Index Out of Range Issue")
    logger.info("=" * 50)
    
    # Step 1: Check embedding sizes
    if check_embedding_sizes():
        logger.info("✅ Embedding size check passed")
    else:
        logger.error("❌ Embedding size check failed")
        return
    
    # Step 2: Test with safe sizes
    result = test_with_safe_sizes()
    
    if result:
        text_len, audio_len = result
        logger.info(f"\n🎉 SOLUTION FOUND!")
        logger.info(f"   Maximum working sizes: text={text_len}, audio={audio_len}")
        logger.info(f"   You need to truncate sequences in continuous preprocessor")
        logger.info(f"   Or increase embedding table sizes in modules")
    else:
        logger.error("❌ No working size found")

if __name__ == "__main__":
    main()