#!/usr/bin/env python3
"""
Continuous Audio Preprocessor - Minutowe Batches bez Chunking
===========================================================
Tworzy ~minutowe continuous batches dla natural speech flow
Key features:
- ~1 minuta na batch (ciągnięcie po słowach)
- Brak internal chunking - jeden długi fragment na batch
- Virtual checkpoints co 10s dla error tracking
- Natural word boundaries - nie przecina słów w połowie
"""

import torch
import torchaudio
import json
import numpy as np
from pathlib import Path
from encodec import EncodecModel
from encodec.utils import convert_audio
from nucleotide_tokenizer import NucleotideTokenizer
import logging
import warnings
from typing import List, Dict, Tuple, Optional
import math

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContinuousAudioPreprocessor:
    """
    Continuous preprocessor - minutowe batches bez internal chunking
    Natural speech flow z virtual checkpoints
    """
    
    def __init__(self, 
                 target_duration_per_batch: float = 60.0,     # 1 minuta na batch
                 duration_tolerance: float = 8.0,             # ±8s tolerance  
                 virtual_checkpoint_interval: float = 10.0,   # Co 10s checkpoint
                 max_batches: int = 12,                       # ~12 minut total
                 output_dir: str = "continuous_data"):
        
        self.target_duration = target_duration_per_batch
        self.tolerance = duration_tolerance
        self.checkpoint_interval = virtual_checkpoint_interval
        self.max_batches = max_batches
        self.output_dir = Path(output_dir)
        
        # Audio processing setup
        self.sample_rate = 24000
        self.hop_size = 320
        self.frame_duration = self.hop_size / self.sample_rate
        
        # Expected tokens per second (for duration estimation)
        self.tokens_per_second = 75.0
        
        # Setup EnCodec
        logger.info("🔧 Initializing EnCodec...")
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(3.0)
        self.codec.eval()
        if torch.cuda.is_available():
            self.codec = self.codec.cuda()
        
        # Setup tokenizer
        self.tokenizer = NucleotideTokenizer()
        
        logger.info(f"🎯 ContinuousAudioPreprocessor initialized:")
        logger.info(f"   Target duration per batch: {target_duration_per_batch}s")
        logger.info(f"   Duration tolerance: ±{duration_tolerance}s")
        logger.info(f"   Virtual checkpoints every: {virtual_checkpoint_interval}s")
        logger.info(f"   Max batches: {max_batches} (~{max_batches * target_duration_per_batch / 60:.1f} minutes total)")
        logger.info(f"   🎵 CONTINUOUS processing - no internal chunking")
        logger.info(f"   🔄 Fresh state between batches only")
        logger.info(f"   Output directory: {output_dir}")
    
    def extract_words_from_whisperx(self, whisperx_data: Dict) -> List[Dict]:
        """Extract and validate words from WhisperX JSON"""
        logger.info("🔍 Extracting words from WhisperX data...")
        
        all_words = []
        segments = whisperx_data.get('segments', [])
        
        for segment_idx, segment in enumerate(segments):
            if 'words' not in segment or not segment['words']:
                continue
                
            for word_info in segment['words']:
                if not self._validate_word(word_info):
                    continue
                
                all_words.append({
                    'word': word_info['word'].strip(),
                    'start': float(word_info['start']),
                    'end': float(word_info['end']),
                    'confidence': word_info.get('score', 1.0),
                    'segment_idx': segment_idx
                })
        
        # Sort by start time
        all_words.sort(key=lambda x: x['start'])
        
        logger.info(f"✅ Extracted {len(all_words)} valid words")
        logger.info(f"   Duration: {all_words[0]['start']:.1f}s - {all_words[-1]['end']:.1f}s")
        
        return all_words
    
    def _validate_word(self, word_info: Dict) -> bool:
        """Validate individual word entry"""
        required_fields = ['word', 'start', 'end']
        
        for field in required_fields:
            if field not in word_info or word_info[field] is None:
                return False
        
        word = word_info['word'].strip()
        if not word or len(word) == 0:
            return False
        
        try:
            start = float(word_info['start'])
            end = float(word_info['end'])
            
            if start < 0 or end <= start or (end - start) > 8.0:  # Max 8s per word
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def create_continuous_batches(self, words: List[Dict]) -> List[Dict]:
        """
        Create continuous minutowe batches - ciągnięcie po słowach
        BRAK internal chunking - jeden długi fragment na batch
        """
        logger.info("🔄 Creating CONTINUOUS minutowe batches...")
        logger.info(f"   Target: {self.target_duration}s ± {self.tolerance}s per batch")
        logger.info(f"   NO internal chunking - continuous processing!")
        
        if len(words) < 50:  # Minimum reasonable number of words
            logger.error(f"❌ Not enough words for continuous batches: {len(words)}")
            return []
        
        all_batches = []
        current_word_idx = 0
        batch_idx = 0
        
        while current_word_idx < len(words) and batch_idx < self.max_batches:
            # Find batch end by duration, respecting word boundaries
            batch_words = []
            batch_start_time = words[current_word_idx]['start']
            current_duration = 0.0
            batch_end_idx = current_word_idx
            
            # Accumulate words until target duration ± tolerance
            while batch_end_idx < len(words):
                word = words[batch_end_idx]
                word_end_time = word['end']
                potential_duration = word_end_time - batch_start_time
                
                # Check if adding this word exceeds max tolerance
                if (potential_duration > self.target_duration + self.tolerance and 
                    current_duration > self.target_duration - self.tolerance):
                    # We have enough - stop here
                    break
                
                batch_words.append(word)
                current_duration = potential_duration
                batch_end_idx += 1
                
                # Safety check - don't make batches too long
                if current_duration > self.target_duration + self.tolerance * 2:
                    logger.warning(f"⚠️  Batch {batch_idx} getting very long: {current_duration:.1f}s")
                    break
            
            # Verify batch is reasonable
            if (len(batch_words) < 20 or  # Too few words
                current_duration < self.target_duration / 3):  # Too short
                logger.warning(f"⚠️  Batch {batch_idx} too small, skipping")
                current_word_idx = batch_end_idx
                continue
            
            # Calculate virtual checkpoints for this batch
            num_checkpoints = max(1, int(current_duration / self.checkpoint_interval))
            checkpoint_times = []
            for i in range(num_checkpoints):
                checkpoint_time = batch_start_time + (i + 1) * (current_duration / num_checkpoints)
                checkpoint_times.append(checkpoint_time)
            
            # Create batch metadata
            batch_info = {
                'batch_idx': batch_idx,
                'words': batch_words,
                'start_time': batch_start_time,
                'end_time': batch_words[-1]['end'],
                'duration': current_duration,
                'word_count': len(batch_words),
                'continuous_batch': True,          # NO internal chunking!
                'internal_chunks': 1,              # Just one continuous piece
                'virtual_checkpoints': checkpoint_times,
                'num_checkpoints': num_checkpoints,
                'checkpoint_interval': self.checkpoint_interval,
                'expected_tokens': int(current_duration * self.tokens_per_second),
                'natural_boundaries': True         # Cut at word boundaries
            }
            
            all_batches.append(batch_info)
            
            logger.info(f"   Batch {batch_idx}: {current_duration:.1f}s, {len(batch_words)} words, {num_checkpoints} checkpoints")
            logger.info(f"     Time: {batch_start_time:.1f}s - {batch_words[-1]['end']:.1f}s")
            logger.info(f"     🎯 CONTINUOUS - no internal breaks")
            
            # Move to next batch
            current_word_idx = batch_end_idx
            batch_idx += 1
        
        logger.info(f"✅ Created {len(all_batches)} continuous batches")
        
        # Analyze coverage
        self._analyze_continuous_coverage(all_batches)
        
        return all_batches
    
    def _analyze_continuous_coverage(self, batches: List[Dict]):
        """Analyze coverage of continuous batches"""
        logger.info("🔍 Analyzing CONTINUOUS batch coverage:")
        
        if not batches:
            return
        
        total_duration = sum(batch['duration'] for batch in batches)
        total_words = sum(batch['word_count'] for batch in batches)
        total_checkpoints = sum(batch['num_checkpoints'] for batch in batches)
        
        durations = [batch['duration'] for batch in batches]
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        logger.info(f"📊 Continuous Batch Analysis:")
        logger.info(f"   Total batches: {len(batches)}")
        logger.info(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        logger.info(f"   Total words: {total_words}")
        logger.info(f"   Total virtual checkpoints: {total_checkpoints}")
        logger.info(f"   Average batch duration: {avg_duration:.1f}s ± {std_duration:.1f}s")
        logger.info(f"   Target was: {self.target_duration:.1f}s ± {self.tolerance:.1f}s")
        
        # Check if we're hitting targets well
        in_tolerance = sum(1 for d in durations 
                          if self.target_duration - self.tolerance <= d <= self.target_duration + self.tolerance)
        tolerance_rate = in_tolerance / len(batches) * 100
        
        logger.info(f"   Within tolerance: {in_tolerance}/{len(batches)} ({tolerance_rate:.1f}%)")
        
        if tolerance_rate > 80:
            logger.info("✅ Excellent batch duration consistency!")
        elif tolerance_rate > 60:
            logger.info("✅ Good batch duration consistency")
        else:
            logger.warning("⚠️  Batch durations quite variable - consider adjusting tolerance")
        
        # Show gaps between batches
        if len(batches) > 1:
            gaps = []
            for i in range(len(batches) - 1):
                gap = batches[i+1]['start_time'] - batches[i]['end_time']
                gaps.append(gap)
            
            avg_gap = np.mean(gaps)
            logger.info(f"   Average gap between batches: {avg_gap:.2f}s")
            
            if avg_gap > 5.0:
                logger.warning("⚠️  Large gaps between batches - some audio unused")
            else:
                logger.info("✅ Small gaps - good audio utilization")
    
    def process_continuous_batch(self, batch_info: Dict, wav_data: torch.Tensor) -> Optional[Dict]:
        """Process one continuous batch - NO internal chunking"""
        batch_idx = batch_info['batch_idx']
        words = batch_info['words']
        duration = batch_info['duration']
        
        logger.info(f"🎵 Processing CONTINUOUS batch {batch_idx} ({duration:.1f}s, {len(words)} words)...")
        logger.info(f"   🔄 NO internal chunking - one continuous sequence")
        
        try:
            # Extract full text for entire batch
            full_text = ' '.join(word['word'] for word in words)
            
            if len(full_text.strip()) < 10:
                logger.warning(f"⚠️  Batch {batch_idx} text too short")
                return None
            
            # Extract continuous audio segment
            start_time = batch_info['start_time']
            end_time = batch_info['end_time']
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Bounds checking
            if start_sample >= wav_data.shape[-1] or end_sample <= start_sample:
                logger.warning(f"⚠️  Batch {batch_idx} audio bounds invalid")
                return None
            
            end_sample = min(end_sample, wav_data.shape[-1])
            continuous_audio = wav_data[:, start_sample:end_sample]
            
            if continuous_audio.shape[-1] < self.sample_rate * 5:  # At least 5 seconds
                logger.warning(f"⚠️  Batch {batch_idx} audio too short")
                return None
            
            # EnCodec encoding for ENTIRE continuous batch
            device = next(self.codec.parameters()).device
            with torch.no_grad():
                audio_tensor = continuous_audio.to(device)
                encoded = self.codec.encode(audio_tensor.unsqueeze(0))
                continuous_audio_codes = encoded[0][0].squeeze(0).cpu()  # [C, T]
            
            # Text tokenization for ENTIRE batch
            full_text_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            
            if len(full_text_tokens) < 10:
                logger.warning(f"⚠️  Batch {batch_idx} tokenization too short")
                return None
            
            # Calculate virtual checkpoint positions in tokens
            audio_tokens_total = continuous_audio_codes.shape[1]
            tokens_per_checkpoint = audio_tokens_total // batch_info['num_checkpoints']
            
            virtual_checkpoint_positions = []
            for i in range(batch_info['num_checkpoints']):
                token_pos = (i + 1) * tokens_per_checkpoint
                time_pos = token_pos / self.tokens_per_second
                virtual_checkpoint_positions.append({
                    'checkpoint_idx': i,
                    'token_position': min(token_pos, audio_tokens_total),
                    'time_position': time_pos,
                    'expected_time': batch_info['virtual_checkpoints'][i] if i < len(batch_info['virtual_checkpoints']) else time_pos
                })
            
            # Create processed continuous batch
            processed_batch = {
                # Continuous text data
                'full_text': full_text,
                'full_text_tokens': torch.tensor(full_text_tokens, dtype=torch.long),
                'words': words,
                'word_count': len(words),
                
                # Continuous audio data
                'continuous_audio_codes': continuous_audio_codes,  # [C, T] - FULL sequence
                'sample_rate': self.sample_rate,
                
                # Timing
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'continuous_duration': True,
                
                # Metadata
                'batch_idx': batch_idx,
                'continuous_batch': True,
                'internal_chunks': 1,                    # Just one piece!
                'no_internal_chunking': True,
                
                # Virtual checkpoints for error tracking
                'virtual_checkpoints': virtual_checkpoint_positions,
                'num_checkpoints': len(virtual_checkpoint_positions),
                'checkpoint_interval': self.checkpoint_interval,
                
                # Stats
                'audio_tokens': continuous_audio_codes.shape[1],
                'text_token_count': len(full_text_tokens),
                'expected_tokens': batch_info['expected_tokens'],
                'tokens_per_second_actual': continuous_audio_codes.shape[1] / duration
            }
            
            logger.info(f"   ✅ Continuous batch processed:")
            logger.info(f"     Audio tokens: {continuous_audio_codes.shape[1]} ({continuous_audio_codes.shape[1]/duration:.1f} tokens/s)")
            logger.info(f"     Text tokens: {len(full_text_tokens)}")
            logger.info(f"     Virtual checkpoints: {len(virtual_checkpoint_positions)}")
            logger.info(f"     🎯 ONE continuous sequence - no breaks!")
            
            return processed_batch
            
        except Exception as e:
            logger.warning(f"⚠️  Continuous batch {batch_idx} processing failed: {e}")
            return None
    
    def save_continuous_batch(self, batch_data: Dict) -> bool:
        """Save processed continuous batch to disk"""
        try:
            batch_idx = batch_data['batch_idx']
            duration = batch_data['duration']
            
            batch_dir = self.output_dir / f"continuous_batch_{batch_idx:02d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the continuous batch data
            batch_filename = f"continuous_{duration:.1f}s.pt"
            batch_path = batch_dir / batch_filename
            
            torch.save(batch_data, batch_path)
            
            # Save batch metadata
            batch_meta = {
                'batch_idx': batch_idx,
                'duration': batch_data['duration'],
                'word_count': batch_data['word_count'],
                'audio_tokens': batch_data['audio_tokens'],
                'text_tokens': batch_data['text_token_count'],
                'continuous_batch': batch_data['continuous_batch'],
                'no_internal_chunking': batch_data['no_internal_chunking'],
                'virtual_checkpoints': batch_data['num_checkpoints'],
                'checkpoint_interval': batch_data['checkpoint_interval'],
                'batch_file': batch_filename,
                'full_text_preview': batch_data['full_text'][:200] + "..." if len(batch_data['full_text']) > 200 else batch_data['full_text']
            }
            
            meta_path = batch_dir / "batch_meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(batch_meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Continuous batch {batch_idx} saved: {duration:.1f}s, {batch_data['audio_tokens']} tokens")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save continuous batch {batch_data['batch_idx']}: {e}")
            return False
    
    def process_full_audio_continuous(self, audio_path: str, whisperx_json_path: str) -> Dict:
        """Main processing function - creates continuous minutowe batches"""
        logger.info("🚀 CONTINUOUS AUDIO PREPROCESSING")
        logger.info("=" * 60)
        logger.info("🎯 Creating minutowe continuous batches WITHOUT internal chunking")
        
        # Load WhisperX data
        logger.info(f"📋 Loading WhisperX data: {whisperx_json_path}")
        with open(whisperx_json_path, 'r', encoding='utf-8') as f:
            whisperx_data = json.load(f)
        
        # Extract words
        words = self.extract_words_from_whisperx(whisperx_data)
        if len(words) == 0:
            logger.error("❌ No valid words extracted")
            return {'success': False, 'error': 'No valid words'}
        
        # Load audio
        logger.info(f"🎵 Loading audio: {audio_path}")
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, target_sr=self.sample_rate, target_channels=1)
        
        audio_duration = wav.shape[-1] / self.sample_rate
        logger.info(f"   Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")
        
        # Create continuous batches
        continuous_batches = self.create_continuous_batches(words)
        if len(continuous_batches) == 0:
            logger.error("❌ No continuous batches created")
            return {'success': False, 'error': 'No continuous batches created'}
        
        # Process each continuous batch
        self.output_dir.mkdir(exist_ok=True)
        processed_batches = []
        failed_batches = 0
        
        for batch_info in continuous_batches:
            batch_data = self.process_continuous_batch(batch_info, wav)
            
            if batch_data is not None:
                if self.save_continuous_batch(batch_data):
                    processed_batches.append(batch_data)
                else:
                    failed_batches += 1
            else:
                failed_batches += 1
        
        # Summary
        total_duration = sum(batch['duration'] for batch in processed_batches)
        total_tokens = sum(batch['audio_tokens'] for batch in processed_batches)
        total_checkpoints = sum(batch['num_checkpoints'] for batch in processed_batches)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 CONTINUOUS PREPROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"📊 Results:")
        logger.info(f"   Successful batches: {len(processed_batches)}")
        logger.info(f"   Failed batches: {failed_batches}")
        logger.info(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        logger.info(f"   Total audio tokens: {total_tokens:,}")
        logger.info(f"   Total virtual checkpoints: {total_checkpoints}")
        logger.info(f"   Average tokens/second: {total_tokens/total_duration:.1f}")
        logger.info(f"   🎯 CONTINUOUS batches - no internal chunking")
        logger.info(f"   🔄 Fresh state between batches only")
        logger.info(f"   📍 Virtual checkpoints every {self.checkpoint_interval}s")
        
        return {
            'success': True,
            'processed_batches': len(processed_batches),
            'failed_batches': failed_batches,
            'total_duration': total_duration,
            'total_tokens': total_tokens,
            'total_checkpoints': total_checkpoints,
            'output_dir': str(self.output_dir),
            'continuous_processing': True,
            'no_internal_chunking': True
        }


def main():
    """Main function for standalone usage"""
    audio_path = "speech.mp3"
    json_path = "speech_transcription.json"
    
    if not Path(audio_path).exists():
        logger.error(f"❌ Audio file not found: {audio_path}")
        return
    
    if not Path(json_path).exists():
        logger.error(f"❌ JSON file not found: {json_path}")
        return
    
    try:
        # Create CONTINUOUS preprocessor
        preprocessor = ContinuousAudioPreprocessor(
            target_duration_per_batch=60.0,      # 1 minuta na batch
            duration_tolerance=8.0,              # ±8s tolerance
            virtual_checkpoint_interval=10.0,     # Co 10s virtual checkpoint
            max_batches=12,                       # ~12 minut max
            output_dir="continuous_data"          # Continuous output directory
        )
        
        # Process audio
        results = preprocessor.process_full_audio_continuous(audio_path, json_path)
        
        if results['success']:
            logger.info("🚀 Ready for CONTINUOUS training!")
            logger.info("📁 Output directory: continuous_data/")
            logger.info("🎯 Features: Minutowe batches, no internal chunking, virtual checkpoints!")
            logger.info("🔄 Fresh Mamba state between batches only!")
        else:
            logger.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Continuous preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()