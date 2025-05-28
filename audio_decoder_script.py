#!/usr/bin/env python3
"""
Samples Audio Decoder
====================
Dekoduje generated_sample_audios.pt z wieloma próbkami
"""

import torch
import torchaudio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_encodec():
    """Załaduj EnCodec"""
    try:
        from encodec import EncodecModel
        logger.info("🔄 Loading EnCodec...")
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load EnCodec: {e}")
        return None

def decode_samples_file(file_path="generated_sample_audios.pt"):
    """Dekoduje plik z wieloma próbkami"""
    try:
        logger.info(f"📂 Loading samples from: {file_path}")
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Sprawdź strukturę
        logger.info(f"🔍 Data keys: {list(data.keys())}")
        
        if 'generated_samples' not in data:
            logger.error("❌ No 'generated_samples' found in file")
            return False
        
        samples = data['generated_samples']
        logger.info(f"📊 Found {len(samples)} samples")
        
        # Załaduj EnCodec
        model = load_encodec()
        if model is None:
            return False
        
        # Dekoduj każdą próbkę
        for i, sample in enumerate(samples):
            try:
                logger.info(f"\n🎯 Sample {i+1}/{len(samples)}")
                
                # Pobierz dane próbki
                text = sample.get('text', 'Unknown text')
                audio_tokens = sample.get('audio_tokens')
                duration = sample.get('duration', 0)
                
                logger.info(f"   Text: '{text[:60]}...'")
                logger.info(f"   Duration: {duration:.2f}s")
                logger.info(f"   Tokens shape: {audio_tokens.shape}")
                
                # Dekoduj audio
                decoded_audio = decode_with_encodec(audio_tokens, model)
                
                if decoded_audio is not None:
                    # Zapisz audio
                    output_file = f"sample_{i+1}_decoded.wav"
                    save_audio_sample(decoded_audio, output_file, text, model.sample_rate)
                    logger.info(f"✅ Saved: {output_file}")
                else:
                    logger.warning(f"❌ Failed to decode sample {i+1}")
                    
            except Exception as e:
                logger.warning(f"❌ Error processing sample {i+1}: {e}")
                continue
        
        logger.info(f"\n🎉 Decoded {len(samples)} samples!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process samples file: {e}")
        return False

def decode_with_encodec(tokens, model):
    """Dekoduje tokeny używając EnCodec"""
    try:
        # Upewnij się, że mamy właściwy format [codebooks, time]
        if tokens.dim() == 3 and tokens.shape[0] == 1:
            tokens = tokens.squeeze(0)
        
        # Upewnij się, że mamy 8 codebooków
        if tokens.shape[0] < 8:
            padding = torch.zeros(8 - tokens.shape[0], tokens.shape[1], dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding], dim=0)
        elif tokens.shape[0] > 8:
            tokens = tokens[:8, :]
        
        # Dodaj wymiar batch [1, 8, time]
        tokens_batch = tokens.unsqueeze(0).long()
        
        # Dekoduj
        with torch.no_grad():
            encoded_frames = [(tokens_batch, None)]
            audio = model.decode(encoded_frames)
            
            # Usuń wymiar batch
            audio = audio.squeeze(0)
            return audio.cpu()
            
    except Exception as e:
        logger.error(f"❌ EnCodec decode failed: {e}")
        return None

def save_audio_sample(audio, filename, text, sample_rate=24000):
    """Zapisz próbkę audio z metadanymi"""
    try:
        # Zapisz audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(filename, audio, sample_rate)
        
        # Zapisz info o próbce
        info_file = filename.replace('.wav', '_info.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Generated Audio Sample\n")
            f.write(f"=" * 30 + "\n\n")
            f.write(f"Audio file: {filename}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Duration: {audio.shape[-1] / sample_rate:.2f}s\n")
            f.write(f"Sample rate: {sample_rate}Hz\n")
            f.write(f"Channels: {audio.shape[0]}\n")
            f.write(f"Audio shape: {audio.shape}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save audio: {e}")
        return False

def main():
    """Main function"""
    logger.info("🎵 Samples Audio Decoder")
    logger.info("=" * 30)
    
    # Sprawdź czy plik istnieje
    file_path = "generated_sample_audios.pt"
    if not Path(file_path).exists():
        logger.error(f"❌ File not found: {file_path}")
        return
    
    # Dekoduj próbki
    success = decode_samples_file(file_path)
    
    if success:
        logger.info("\n✅ All done! Check the generated files:")
        logger.info("   • sample_1_decoded.wav, sample_2_decoded.wav, ...")
        logger.info("   • sample_1_decoded_info.txt, sample_2_decoded_info.txt, ...")
        logger.info("\n🎧 You can now listen to your TTS generated audio!")
    else:
        logger.error("❌ Failed to decode samples")

if __name__ == "__main__":
    main()