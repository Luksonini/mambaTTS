#!/usr/bin/env python3
"""
Real EnCodec Audio Decoder - IMPROVED VERSION
==============================================
Decode 8-codebook tokens to actual audio using EnCodec
- Supports TTS generated tokens
- Auto-finds latest token files
- Better error handling
- Multiple input formats
"""

import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
import argparse
import glob

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
        from encodec import EncodecModel
        
        logger.info("🔄 Loading EnCodec 24kHz model...")
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("🚀 Model moved to GPU")
        
        logger.info("✅ EnCodec model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load EnCodec model: {e}")
        return None

def find_token_files():
    """Find all available token files"""
    token_files = []
    
    # Look for TTS generated tokens
    tts_patterns = [
        "*_tokens.pt",
        "test_*_tokens.pt", 
        "generated_*_tokens.pt",
        "*_audio_tokens.pt"
    ]
    
    for pattern in tts_patterns:
        matches = list(Path('.').glob(pattern))
        token_files.extend(matches)
    
    # Look for original training tokens
    original_patterns = [
        "full_mamba_audio_tokens.pt",
        "mamba_audio_tokens.pt",
        "audio_tokens.pt"
    ]
    
    for pattern in original_patterns:
        matches = list(Path('.').glob(pattern))
        token_files.extend(matches)
    
    # Remove duplicates and sort by modification time (newest first)
    token_files = list(set(token_files))
    token_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return token_files

def load_tokens(file_path):
    """Load tokens from various file formats"""
    try:
        logger.info(f"📂 Loading tokens from: {file_path}")
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Handle different data formats
        audio_tokens = None
        metadata = {}
        
        if isinstance(data, dict):
            # TTS generated format
            if 'audio_tokens' in data:
                audio_tokens = data['audio_tokens']
                metadata = {k: v for k, v in data.items() if k != 'audio_tokens'}
                logger.info("✅ Loaded TTS generated tokens")
                
            # Training data format
            elif 'tokens' in data:
                audio_tokens = data['tokens']
                metadata = {k: v for k, v in data.items() if k != 'tokens'}
                logger.info("✅ Loaded training tokens")
                
            # Direct tensor in dict
            else:
                # Try to find tensor-like values
                for key, value in data.items():
                    if torch.is_tensor(value) and value.dim() >= 2:
                        audio_tokens = value
                        metadata[key] = 'used_as_tokens'
                        logger.info(f"✅ Loaded tokens from key: {key}")
                        break
        
        elif torch.is_tensor(data):
            # Direct tensor
            audio_tokens = data
            logger.info("✅ Loaded direct tensor")
        
        else:
            logger.error(f"❌ Unknown data format: {type(data)}")
            return None, {}
        
        if audio_tokens is None:
            logger.error("❌ No audio tokens found in file")
            return None, {}
        
        # Ensure correct format [codebooks, time]
        if audio_tokens.dim() == 3 and audio_tokens.shape[0] == 1:
            audio_tokens = audio_tokens.squeeze(0)  # Remove batch dimension
        
        logger.info(f"   Token shape: {audio_tokens.shape}")
        logger.info(f"   Token range: {audio_tokens.min()} - {audio_tokens.max()}")
        logger.info(f"   Metadata: {list(metadata.keys())}")
        
        # Log text if available
        if 'text' in metadata:
            text = str(metadata['text'])
            logger.info(f"   Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        return audio_tokens, metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to load tokens from {file_path}: {e}")
        return None, {}

def decode_tokens_with_encodec(tokens, model):
    """Decode 8-codebook tokens using EnCodec"""
    try:
        logger.info("🎵 Decoding tokens with real EnCodec...")
        
        # Validate token shape
        if tokens.dim() != 2:
            logger.error(f"❌ Expected 2D tokens [codebooks, time], got {tokens.shape}")
            return None
        
        num_codebooks, seq_len = tokens.shape
        logger.info(f"   Input shape: [{num_codebooks}, {seq_len}]")
        
        # Ensure we have 8 codebooks
        if num_codebooks < 8:
            logger.info(f"⚠️  Padding from {num_codebooks} to 8 codebooks")
            padding = torch.zeros(8 - num_codebooks, seq_len, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding], dim=0)
        elif num_codebooks > 8:
            logger.info(f"⚠️  Truncating from {num_codebooks} to 8 codebooks")
            tokens = tokens[:8, :]
        
        # Add batch dimension [1, 8, seq_len]
        tokens_batch = tokens.unsqueeze(0)
        
        # Move to same device as model
        device = next(model.parameters()).device
        tokens_batch = tokens_batch.to(device).long()
        
        logger.info(f"   Batch shape: {tokens_batch.shape}")
        logger.info(f"   Device: {device}")
        
        # Decode with multiple fallback methods
        with torch.no_grad():
            audio = None
            
            # Method 1: Direct decode (preferred)
            try:
                logger.info("🔄 Method 1: Direct decode...")
                encoded_frames = [(tokens_batch, None)]
                audio = model.decode(encoded_frames)
                logger.info(f"✅ Direct decode success: {audio.shape}")
                
            except Exception as e1:
                logger.warning(f"⚠️  Direct decode failed: {e1}")
                
                # Method 2: Manual decode through quantizer
                try:
                    logger.info("🔄 Method 2: Manual decode...")
                    quantized = model.quantizer.decode(tokens_batch)
                    audio = model.decoder(quantized)
                    logger.info(f"✅ Manual decode success: {audio.shape}")
                    
                except Exception as e2:
                    logger.warning(f"⚠️  Manual decode failed: {e2}")
                    
                    # Method 3: Alternative format
                    try:
                        logger.info("🔄 Method 3: Alternative format...")
                        # Try different input formats
                        alt_input = {"codes": tokens_batch}
                        quantized = model.quantizer.decode(alt_input)
                        audio = model.decoder(quantized)
                        logger.info(f"✅ Alternative decode success: {audio.shape}")
                        
                    except Exception as e3:
                        logger.error(f"❌ All decode methods failed:")
                        logger.error(f"   Method 1: {e1}")
                        logger.error(f"   Method 2: {e2}")
                        logger.error(f"   Method 3: {e3}")
                        return None
        
        if audio is not None:
            # Remove batch dimension and move to CPU
            audio = audio.squeeze(0).cpu()
            logger.info(f"✅ Final audio shape: {audio.shape}")
            return audio
        else:
            return None
        
    except Exception as e:
        logger.error(f"❌ EnCodec decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_audio_quality(audio, sample_rate=24000):
    """Analyze decoded audio quality"""
    try:
        logger.info("📈 Audio Quality Analysis:")
        
        audio_flat = audio.flatten()
        rms_level = torch.sqrt(torch.mean(audio_flat**2))
        peak_level = torch.max(torch.abs(audio_flat))
        
        logger.info(f"   Duration: {audio.shape[-1] / sample_rate:.2f}s")
        logger.info(f"   Channels: {1 if audio.dim() == 1 else audio.shape[0]}")
        logger.info(f"   RMS level: {rms_level:.4f}")
        logger.info(f"   Peak level: {peak_level:.4f}")
        logger.info(f"   Dynamic range: {torch.max(audio_flat) - torch.min(audio_flat):.4f}")
        
        # Quality warnings
        if rms_level < 0.001:
            logger.warning("⚠️  Audio seems very quiet - possible silence")
        elif rms_level > 0.5:
            logger.warning("⚠️  Audio seems very loud - possible noise")
        else:
            logger.info("✅ Audio levels seem reasonable")
            
        # Check for potential issues
        if peak_level > 0.95:
            logger.warning("⚠️  Audio may be clipping (peak > 0.95)")
        
        zero_samples = (audio_flat == 0).sum().item()
        if zero_samples > len(audio_flat) * 0.5:
            logger.warning(f"⚠️  High number of zero samples: {zero_samples}/{len(audio_flat)}")
        
    except Exception as e:
        logger.warning(f"⚠️  Quality analysis failed: {e}")

def save_audio(audio, sample_rate=24000, filename="decoded_audio.wav", metadata=None):
    """Save decoded audio with metadata"""
    try:
        logger.info(f"💾 Saving audio as {filename}...")
        
        # Ensure proper format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        elif audio.dim() == 3:
            audio = audio.squeeze(0)  # Remove batch dimension if present
        
        # Save audio
        torchaudio.save(filename, audio, sample_rate)
        
        # Save numpy version for analysis
        np_filename = filename.replace('.wav', '.npy')
        np.save(np_filename, audio.numpy())
        
        # Save metadata if available
        if metadata:
            meta_filename = filename.replace('.wav', '_metadata.json')
            import json
            
            # Convert tensors to serializable format
            clean_metadata = {}
            for k, v in metadata.items():
                if torch.is_tensor(v):
                    clean_metadata[k] = f"tensor_{v.shape}"
                elif isinstance(v, (str, int, float, bool, list, dict)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)
            
            with open(meta_filename, 'w') as f:
                json.dump(clean_metadata, f, indent=2)
            
            logger.info(f"📄 Metadata saved: {meta_filename}")
        
        logger.info(f"✅ Audio saved successfully!")
        logger.info(f"   🎵 Audio: {filename}")
        logger.info(f"   📊 NumPy: {np_filename}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save audio: {e}")
        return False

def main():
    """Main function with improved argument handling"""
    parser = argparse.ArgumentParser(description='Decode audio tokens with EnCodec')
    parser.add_argument('--input', '-i', type=str, help='Input token file path')
    parser.add_argument('--output', '-o', type=str, default='decoded_audio.wav', help='Output audio file path')
    parser.add_argument('--list', action='store_true', help='List available token files')
    
    args = parser.parse_args()
    
    logger.info("🎵 Real EnCodec Audio Decoder - IMPROVED")
    logger.info("=" * 60)
    
    # List available files if requested
    if args.list:
        token_files = find_token_files()
        if token_files:
            logger.info("📁 Available token files:")
            for i, file_path in enumerate(token_files):
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                logger.info(f"   {i+1}. {file_path} ({size_mb:.1f}MB)")
        else:
            logger.info("❌ No token files found")
        return
    
    # Install and load EnCodec
    if not install_encodec():
        logger.error("❌ Cannot proceed without EnCodec")
        return
    
    model = load_encodec_model()
    if model is None:
        logger.error("❌ Cannot proceed without EnCodec model")
        return
    
    # Determine input file
    input_file = None
    
    if args.input:
        input_file = Path(args.input)
        if not input_file.exists():
            logger.error(f"❌ Input file not found: {input_file}")
            return
    else:
        # Auto-find token files
        token_files = find_token_files()
        
        if not token_files:
            logger.error("❌ No token files found. Use --list to see available files")
            logger.error("   Expected files: *_tokens.pt, full_mamba_audio_tokens.pt, etc.")
            return
        
        # Use the most recent file
        input_file = token_files[0]
        logger.info(f"🎯 Auto-selected: {input_file} (most recent)")
        
        if len(token_files) > 1:
            logger.info(f"   Other files available: {[str(f) for f in token_files[1:3]]}")
            logger.info("   Use --input to specify a different file")
    
    # Load tokens
    audio_tokens, metadata = load_tokens(input_file)
    if audio_tokens is None:
        logger.error("❌ Failed to load tokens")
        return
    
    # Generate output filename based on input
    if args.output == 'decoded_audio.wav':  # Default output
        base_name = input_file.stem
        output_file = f"{base_name}_decoded.wav"
    else:
        output_file = args.output
    
    # Decode with real EnCodec
    logger.info(f"\n🎵 Decoding with real EnCodec...")
    decoded_audio = decode_tokens_with_encodec(audio_tokens, model)
    
    if decoded_audio is not None:
        # Analyze quality
        analyze_audio_quality(decoded_audio, model.sample_rate)
        
        # Save the audio
        success = save_audio(decoded_audio, model.sample_rate, output_file, metadata)
        
        if success:
            logger.info(f"\n🎉 Real EnCodec decoding complete!")
            logger.info(f"   🎵 Generated: {output_file}")
            logger.info(f"   📊 Analysis: {output_file.replace('.wav', '.npy')}")
            logger.info(f"   📄 Metadata: {output_file.replace('.wav', '_metadata.json')}")
            
            if 'text' in metadata:
                logger.info(f"   📝 Original text: '{str(metadata['text'])[:100]}...'")
            
            logger.info(f"   🔊 This is REAL TTS audio from your model!")
        else:
            logger.error("❌ Failed to save audio")
    else:
        logger.error("❌ Decoding failed")

if __name__ == "__main__":
    main()