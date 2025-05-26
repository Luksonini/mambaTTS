#!/usr/bin/env python3
"""
Real EnCodec Audio Decoder
=========================
Decode 8-codebook tokens to actual audio using EnCodec
"""

import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def install_encodec():
    """Install EnCodec if not available"""
    try:
        import encodec
        logger.info("✅ EnCodec already installed")
        return True
    except ImportError:
        logger.info("📦 Installing EnCodec...")
        import subprocess
        import sys
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "encodec"])
            logger.info("✅ EnCodec installed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to install EnCodec: {e}")
            return False

def load_encodec_model():
    """Load pre-trained EnCodec model"""
    try:
        # Use the same import as your preprocessor
        from encodec import EncodecModel
        
        logger.info("🔄 Loading EnCodec 24kHz model (same as preprocessor)...")
        
        # Load the pre-trained model - same as your audio_processor_sequential.py
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("🚀 Model moved to GPU")
        
        logger.info("✅ EnCodec model loaded successfully")
        logger.info(f"   Sample rate: 24000Hz")
        logger.info(f"   Bandwidth: 6.0kbps")
        logger.info(f"   Same model as used in preprocessing ✅")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load EnCodec model: {e}")
        
        # Try alternative import
        try:
            logger.info("🔄 Trying alternative EnCodec import...")
            import encodec
            model = encodec.model.EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(6.0)
            model.eval()
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            logger.info("✅ EnCodec model loaded with alternative import")
            return model
            
        except Exception as e2:
            logger.error(f"❌ Alternative import also failed: {e2}")
            return None

def decode_tokens_with_encodec(tokens, model):
    """Decode 8-codebook tokens using EnCodec"""
    try:
        logger.info("🎵 Decoding tokens with real EnCodec...")
        
        # Check token shape
        if tokens.dim() != 2:
            logger.error(f"❌ Expected 2D tokens [codebooks, time], got {tokens.shape}")
            return None
        
        num_codebooks, seq_len = tokens.shape
        logger.info(f"   Input shape: [{num_codebooks}, {seq_len}]")
        logger.info(f"   Token range: {tokens.min()} - {tokens.max()}")
        
        # EnCodec expects tokens as [batch, codebooks, time]
        # Add batch dimension
        tokens_batch = tokens.unsqueeze(0)  # [1, codebooks, time]
        
        logger.info(f"   Batch shape: {tokens_batch.shape}")
        
        # Move to same device as model
        device = next(model.parameters()).device
        tokens_batch = tokens_batch.to(device)
        
        # Decode with EnCodec (same method as used in preprocessing)
        with torch.no_grad():
            logger.info("🔄 Running EnCodec decoder...")
            
            try:
                # Method 1: Direct decode using the model's decode method
                # This should match the encode/decode cycle from preprocessing
                encoded_frames = [(tokens_batch.long(), None)]  # Format: [(codes, scale)]
                audio = model.decode(encoded_frames)
                
                logger.info(f"✅ Direct decoded audio shape: {audio.shape}")
                return audio.squeeze(0).cpu()  # Remove batch dimension, move to CPU
                
            except Exception as e1:
                logger.warning(f"⚠️  Direct decode failed: {e1}")
                
                # Method 2: Try manual decode through quantizer and decoder
                try:
                    logger.info("🔄 Trying manual decode...")
                    
                    # Use the quantizer to decode tokens to embeddings
                    quantized = model.quantizer.decode(tokens_batch.long())
                    
                    # Use the decoder to convert embeddings to audio
                    audio = model.decoder(quantized)
                    
                    logger.info(f"✅ Manual decoded audio shape: {audio.shape}")
                    return audio.squeeze(0).cpu()  # Remove batch dimension, move to CPU
                    
                except Exception as e2:
                    logger.error(f"❌ Manual decode also failed: {e2}")
                    
                    # Method 3: Try with different format
                    try:
                        logger.info("🔄 Trying alternative format...")
                        
                        # Some versions expect different input format
                        codes_dict = {"codes": tokens_batch.long()}
                        quantized = model.quantizer.decode(codes_dict)
                        audio = model.decoder(quantized)
                        
                        logger.info(f"✅ Alternative format decoded audio shape: {audio.shape}")
                        return audio.squeeze(0).cpu()
                        
                    except Exception as e3:
                        logger.error(f"❌ All decode methods failed")
                        logger.error(f"   Method 1: {e1}")
                        logger.error(f"   Method 2: {e2}")
                        logger.error(f"   Method 3: {e3}")
                        return None
        
    except Exception as e:
        logger.error(f"❌ EnCodec decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_audio(audio, sample_rate=24000, filename="real_encodec_audio.wav"):
    """Save decoded audio as WAV file"""
    try:
        logger.info(f"💾 Saving audio as {filename}...")
        
        # Ensure audio is 2D [channels, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        elif audio.dim() == 3:
            audio = audio.squeeze(0)  # Remove batch dimension
        
        logger.info(f"   Audio shape: {audio.shape}")
        logger.info(f"   Duration: {audio.shape[-1] / sample_rate:.2f}s")
        logger.info(f"   Sample rate: {sample_rate}Hz")
        logger.info(f"   Amplitude range: {audio.min():.3f} - {audio.max():.3f}")
        
        # Save using torchaudio
        torchaudio.save(filename, audio.cpu(), sample_rate)
        
        logger.info(f"✅ Audio saved successfully!")
        
        # Also save as numpy for analysis
        np_audio = audio.cpu().numpy()
        np.save(filename.replace('.wav', '.npy'), np_audio)
        logger.info(f"📊 Also saved as {filename.replace('.wav', '.npy')} for analysis")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save audio: {e}")
        return False

def compare_with_original():
    """Compare generated tokens with original training data"""
    try:
        logger.info("🔍 Comparing with original training data...")
        
        # Try to load some original chunks for comparison
        data_dir = Path("no_overlap_data")
        if not data_dir.exists():
            logger.warning("⚠️  no_overlap_data not found, skipping comparison")
            return
        
        # Find first available chunk
        for batch_dir in data_dir.iterdir():
            if batch_dir.is_dir() and batch_dir.name.startswith('clean_batch_'):
                chunk_files = list(batch_dir.glob("*.pt"))
                if chunk_files:
                    chunk_file = chunk_files[0]
                    
                    logger.info(f"📊 Loading original chunk: {chunk_file.name}")
                    original_chunk = torch.load(chunk_file, map_location='cpu')
                    
                    if 'audio_codes' in original_chunk:
                        orig_codes = original_chunk['audio_codes']
                        logger.info(f"   Original shape: {orig_codes.shape}")
                        logger.info(f"   Original range: {orig_codes.min()} - {orig_codes.max()}")
                        
                        # Compare token statistics
                        for i in range(min(8, orig_codes.shape[0])):
                            orig_unique = len(torch.unique(orig_codes[i, :]))
                            logger.info(f"   Original codebook {i}: {orig_unique} unique tokens")
                        
                        logger.info(f"   Original text: '{original_chunk.get('text', 'N/A')[:50]}...'")
                        return orig_codes
                    break
        
        logger.warning("⚠️  No original chunks found for comparison")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️  Comparison failed: {e}")
        return None

def main():
    """Main function"""
    logger.info("🎵 Real EnCodec Audio Decoder")
    logger.info("=" * 50)
    
    # Install and load EnCodec
    if not install_encodec():
        logger.error("❌ Cannot proceed without EnCodec")
        return
    
    model = load_encodec_model()
    if model is None:
        logger.error("❌ Cannot proceed without EnCodec model")
        return
    
    # Load generated tokens
    try:
        logger.info("📂 Loading generated 8-codebook tokens...")
        data = torch.load('enhanced_8codebook_audio_tokens.pt', map_location='cpu')
        audio_tokens = data['audio_tokens']  # [8, T]
        
        logger.info(f"✅ Loaded tokens: {audio_tokens.shape}")
        logger.info(f"   Duration: {audio_tokens.shape[1] / 75.0:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Failed to load tokens: {e}")
        return
    
    # Compare with original data
    original_tokens = compare_with_original()
    
    # Decode with real EnCodec
    logger.info(f"\n🎵 Decoding with real EnCodec...")
    decoded_audio = decode_tokens_with_encodec(audio_tokens, model)
    
    if decoded_audio is not None:
        # Save the real audio
        success = save_audio(decoded_audio, model.sample_rate, "real_encodec_generated.wav")
        
        if success:
            logger.info(f"\n🎉 Real EnCodec decoding complete!")
            logger.info(f"   🎵 Generated file: real_encodec_generated.wav")
            logger.info(f"   📊 Analysis file: real_encodec_generated.npy")
            logger.info(f"   🔊 This is REAL TTS audio from your 8-codebook model!")
            
            # Analyze audio quality
            logger.info(f"\n📈 Audio Quality Analysis:")
            audio_flat = decoded_audio.flatten()
            logger.info(f"   RMS level: {torch.sqrt(torch.mean(audio_flat**2)):.4f}")
            logger.info(f"   Peak level: {torch.max(torch.abs(audio_flat)):.4f}")
            logger.info(f"   Dynamic range: {torch.max(audio_flat) - torch.min(audio_flat):.4f}")
            
            # Check for silence or noise
            rms_level = torch.sqrt(torch.mean(audio_flat**2))
            if rms_level < 0.001:
                logger.warning("⚠️  Audio seems very quiet - possible silence")
            elif rms_level > 0.5:
                logger.warning("⚠️  Audio seems very loud - possible noise")
            else:
                logger.info("✅ Audio levels seem reasonable")
            
        else:
            logger.error("❌ Failed to save audio")
    else:
        logger.error("❌ Decoding failed")
        
        # Try decoding original tokens for comparison
        if original_tokens is not None:
            logger.info("🔄 Trying to decode original tokens for comparison...")
            orig_audio = decode_tokens_with_encodec(original_tokens, model)
            if orig_audio is not None:
                save_audio(orig_audio, model.sample_rate, "original_encodec_audio.wav")
                logger.info("✅ Original audio decoded for comparison")

if __name__ == "__main__":
    main()