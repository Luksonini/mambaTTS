#!/usr/bin/env python3
"""
Batch No-Overlap Audio Preprocessor - Process all 6 samples
===========================================================
Automatically processes all 6 samples from whisper_transcription folder:
- sample1.wav + sample1_transcription.json → clean_batch_00
- sample2.wav + sample2_transcription.json → clean_batch_01
- ... etc ...

Each batch contains all chunks from one 11-minute recording (~66 chunks per batch)
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
import glob

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchNoOverlapAudioPreprocessor:
    """
    Batch processor for all 6 audio samples
    Creates clean sequential chunks WITHOUT overlaps
    One batch = one complete 11-minute recording
    """
    
    def __init__(self, 
                 words_per_chunk: int = 15,          # ~10 seconds per chunk
                 input_dir: str = "whisper_transcription",
                 output_dir: str = "no_overlap_data"):
        
        self.words_per_chunk = words_per_chunk
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Audio processing setup
        self.sample_rate = 24000
        self.hop_size = 320
        self.frame_duration = self.hop_size / self.sample_rate
        
        # Setup EnCodec
        logger.info("🔧 Initializing EnCodec...")
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(6.0)
        self.codec.eval()
        if torch.cuda.is_available():
            self.codec = self.codec.cuda()
        
        # Setup tokenizer
        self.tokenizer = NucleotideTokenizer()
        
        logger.info(f"🎯 BatchNoOverlapAudioPreprocessor initialized:")
        logger.info(f"   Words per chunk: {words_per_chunk} (~10 seconds)")
        logger.info(f"   NO OVERLAPS: Clean sequential chunks ✅")
        logger.info(f"   Input directory: {input_dir}")
        logger.info(f"   Output directory: {output_dir}")
        logger.info(f"   Expected: 6 samples → 6 clean batches (~66 chunks each)")
        logger.info(f"   🎵 EnCodec ready")
        logger.info(f"   🔤 Tokenizer ready")
    
    def find_sample_files(self) -> List[Dict]:
        """Find all sample pairs (audio + transcription)"""
        logger.info(f"🔍 Searching for sample files in {self.input_dir}")
        
        samples = []
        
        # Look for processed audio files
        audio_dir = self.input_dir / "processed_audio"
        transcription_dir = self.input_dir / "transcriptions"
        
        if not audio_dir.exists():
            logger.error(f"❌ Audio directory not found: {audio_dir}")
            return []
        
        if not transcription_dir.exists():
            logger.error(f"❌ Transcription directory not found: {transcription_dir}")
            return []
        
        # Find sample files (sample1.wav, sample2.wav, etc.)
        audio_files = sorted(audio_dir.glob("sample*.wav"))
        
        for audio_file in audio_files:
            # Extract sample number from filename (e.g., sample1.wav → 1)
            sample_name = audio_file.stem  # "sample1"
            sample_num = sample_name.replace("sample", "")
            
            # Find corresponding transcription file
            transcription_file = transcription_dir / f"{sample_name}_transcription.json"
            
            if transcription_file.exists():
                samples.append({
                    'sample_name': sample_name,
                    'sample_num': int(sample_num),
                    'audio_path': audio_file,
                    'transcription_path': transcription_file,
                    'batch_idx': int(sample_num) - 1  # 0-based batch index
                })
                logger.info(f"   ✅ Found {sample_name}: {audio_file.name} + {transcription_file.name}")
            else:
                logger.warning(f"   ⚠️  Missing transcription for {sample_name}: {transcription_file}")
        
        logger.info(f"📁 Found {len(samples)} complete sample pairs")
        return samples
    
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
        if all_words:
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
            
            if start < 0 or end <= start or (end - start) > 5.0:
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def create_chunks_for_full_recording(self, words: List[Dict], sample_name: str) -> List[Dict]:
        """
        Create all chunks for one complete 11-minute recording
        Each chunk ~10 seconds, no overlaps, sequential
        """
        logger.info(f"🔄 Creating chunks for complete recording: {sample_name}")
        
        if len(words) < self.words_per_chunk:
            logger.error(f"❌ Not enough words for chunking: {len(words)} < {self.words_per_chunk}")
            return []
        
        chunks = []
        current_word_idx = 0
        chunk_idx = 0
        
        while current_word_idx < len(words):
            chunk_start = current_word_idx
            chunk_end = current_word_idx + self.words_per_chunk
            
            # Check if we have enough words
            if chunk_end > len(words):
                # Use remaining words if we have at least half a chunk
                if len(words) - current_word_idx >= self.words_per_chunk // 2:
                    chunk_end = len(words)
                else:
                    break
            
            # Extract chunk words - NO OVERLAP!
            chunk_words = words[chunk_start:chunk_end]
            
            if len(chunk_words) < 3:  # Need at least 3 words
                break
            
            # Create chunk metadata
            chunk_info = {
                'words': chunk_words,
                'chunk_idx': chunk_idx,
                'start_time': chunk_words[0]['start'],
                'end_time': chunk_words[-1]['end'],
                'duration': chunk_words[-1]['end'] - chunk_words[0]['start'],
                'word_count': len(chunk_words),
                'has_overlap': False,  # NO OVERLAPS!
                'overlap_words': 0,    # ZERO overlaps!
                'clean_chunk': True,   # Mark as clean
                'sample_name': sample_name
            }
            
            chunks.append(chunk_info)
            current_word_idx += len(chunk_words)  # CLEAN advance - no overlap!
            chunk_idx += 1
        
        logger.info(f"✅ Created {len(chunks)} CLEAN chunks for {sample_name}")
        logger.info(f"   Total duration: {chunks[-1]['end_time'] - chunks[0]['start_time']:.1f}s")
        logger.info(f"   Average chunk duration: {sum(c['duration'] for c in chunks) / len(chunks):.1f}s")
        
        return chunks
    
    def process_single_sample(self, sample_info: Dict) -> Optional[Dict]:
        """Process one complete sample (audio + transcription)"""
        sample_name = sample_info['sample_name']
        batch_idx = sample_info['batch_idx']
        audio_path = sample_info['audio_path']
        transcription_path = sample_info['transcription_path']
        
        logger.info(f"🎵 Processing {sample_name} → clean_batch_{batch_idx:02d}")
        logger.info(f"   Audio: {audio_path}")
        logger.info(f"   Transcription: {transcription_path}")
        
        try:
            # Load WhisperX transcription
            with open(transcription_path, 'r', encoding='utf-8') as f:
                whisperx_data = json.load(f)
            
            # Extract words
            words = self.extract_words_from_whisperx(whisperx_data)
            if len(words) == 0:
                logger.error(f"❌ No valid words extracted from {sample_name}")
                return None
            
            # Load audio
            logger.info(f"🎵 Loading audio: {audio_path}")
            wav, sr = torchaudio.load(str(audio_path))
            wav = convert_audio(wav, sr, target_sr=self.sample_rate, target_channels=1)
            
            audio_duration = wav.shape[-1] / self.sample_rate
            logger.info(f"   Audio duration: {audio_duration:.1f}s")
            
            # Create chunks for entire recording
            chunks = self.create_chunks_for_full_recording(words, sample_name)
            if len(chunks) == 0:
                logger.error(f"❌ No chunks created for {sample_name}")
                return None
            
            # Process each chunk
            processed_chunks = []
            
            for chunk_info in chunks:
                processed_chunk = self._process_single_chunk(chunk_info, wav, batch_idx)
                
                if processed_chunk is not None:
                    processed_chunks.append(processed_chunk)
                else:
                    logger.warning(f"⚠️  Failed to process chunk {chunk_info['chunk_idx']} in {sample_name}")
            
            if len(processed_chunks) == 0:
                logger.error(f"❌ No chunks processed for {sample_name}")
                return None
            
            # Create batch data (one batch = one complete recording)
            batch_data = {
                'batch_idx': batch_idx,
                'sample_name': sample_name,
                'chunks': processed_chunks,
                'num_chunks': len(processed_chunks),
                'total_duration': chunks[-1]['end_time'] - chunks[0]['start_time'],
                'total_words': sum(chunk['word_count'] for chunk in chunks),
                'audio_duration': audio_duration,
                'no_overlaps': True,           # Mark as clean
                'sequential': True,            # Sequential chunks
                'clean_boundaries': True,      # Clean chunk boundaries
                'source_audio': str(audio_path),
                'source_transcription': str(transcription_path)
            }
            
            logger.info(f"✅ Processed {sample_name}: {len(processed_chunks)} chunks, {batch_data['total_duration']:.1f}s")
            
            return batch_data
            
        except Exception as e:
            logger.error(f"❌ Failed to process {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_single_chunk(self, chunk_info: Dict, wav_data: torch.Tensor, batch_idx: int) -> Optional[Dict]:
        """Process single clean chunk"""
        try:
            # Extract text
            words = chunk_info['words']
            chunk_text = ' '.join(word['word'] for word in words)
            
            if len(chunk_text.strip()) < 3:
                return None
            
            # Extract audio segment
            start_time = chunk_info['start_time']
            end_time = chunk_info['end_time']
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Bounds checking
            if start_sample >= wav_data.shape[-1] or end_sample <= start_sample:
                return None
            
            end_sample = min(end_sample, wav_data.shape[-1])
            audio_segment = wav_data[:, start_sample:end_sample]
            
            if audio_segment.shape[-1] < self.sample_rate * 0.2:  # At least 0.2 seconds
                return None
            
            # EnCodec encoding
            device = next(self.codec.parameters()).device
            with torch.no_grad():
                audio_tensor = audio_segment.to(device)
                encoded = self.codec.encode(audio_tensor.unsqueeze(0))
                audio_codes = encoded[0][0].squeeze(0).cpu()  # [C, T]
            
            # Text tokenization
            text_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True)
            
            if len(text_tokens) < 3:
                return None
            
            # Create processed chunk
            processed_chunk = {
                # Text data
                'text': chunk_text,
                'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
                'words': words,
                
                # Audio data
                'audio_codes': audio_codes,  # [C, T]
                'sample_rate': self.sample_rate,
                
                # Timing
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                
                # Metadata
                'chunk_idx': chunk_info['chunk_idx'],
                'batch_idx': batch_idx,
                'word_count': len(words),
                'sample_name': chunk_info['sample_name'],
                
                # NO-OVERLAP specific
                'has_overlap': False,         # NO overlaps!
                'overlap_words': 0,           # ZERO overlaps!
                'clean_chunk': True,          # Mark as clean
                'boundary_type': 'clean',     # Clean boundaries
                
                # Sequential processing metadata
                'sequential_chunk': True,
                'audio_tokens': audio_codes.shape[-1],
                'text_token_count': len(text_tokens)
            }
            
            return processed_chunk
            
        except Exception as e:
            logger.warning(f"⚠️  Chunk processing failed: {e}")
            return None
    
    def save_batch(self, batch_data: Dict) -> bool:
        """Save processed NO-OVERLAP batch to disk"""
        try:
            batch_idx = batch_data['batch_idx']
            sample_name = batch_data['sample_name']
            batch_dir = self.output_dir / f"clean_batch_{batch_idx:02d}"  # clean_batch_00, clean_batch_01, etc.
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each chunk individually
            for chunk in batch_data['chunks']:
                chunk_filename = f"chunk_{chunk['chunk_idx']:02d}_{chunk['start_time']:.1f}-{chunk['end_time']:.1f}s.pt"
                chunk_path = batch_dir / chunk_filename
                
                torch.save(chunk, chunk_path)
            
            # Save batch metadata
            batch_meta = {
                'batch_idx': batch_idx,
                'sample_name': sample_name,
                'num_chunks': batch_data['num_chunks'],
                'total_duration': batch_data['total_duration'],
                'total_words': batch_data['total_words'],
                'audio_duration': batch_data['audio_duration'],
                'no_overlaps': batch_data['no_overlaps'],
                'sequential': batch_data['sequential'],
                'clean_boundaries': batch_data['clean_boundaries'],
                'source_audio': batch_data['source_audio'],
                'source_transcription': batch_data['source_transcription'],
                'chunk_files': [f"chunk_{chunk['chunk_idx']:02d}_{chunk['start_time']:.1f}-{chunk['end_time']:.1f}s.pt" 
                               for chunk in batch_data['chunks']]
            }
            
            meta_path = batch_dir / "batch_meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(batch_meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Batch {batch_idx} ({sample_name}) saved: {batch_data['num_chunks']} chunks, {batch_data['total_duration']:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save batch {batch_data['batch_idx']}: {e}")
            return False
    
    def process_all_samples(self) -> Dict:
        """Main processing function - process all 6 samples"""
        logger.info("🚀 BATCH NO-OVERLAP AUDIO PREPROCESSING")
        logger.info("=" * 70)
        logger.info("Processing all samples from whisper_transcription folder")
        logger.info("Each sample becomes one clean_batch with ~66 chunks")
        logger.info("=" * 70)
        
        # Find all sample files
        samples = self.find_sample_files()
        if len(samples) == 0:
            logger.error("❌ No sample files found")
            return {'success': False, 'error': 'No sample files found'}
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Process each sample
        processed_batches = []
        failed_batches = 0
        
        for sample_info in samples:
            logger.info(f"\n{'='*50}")
            batch_data = self.process_single_sample(sample_info)
            
            if batch_data is not None:
                if self.save_batch(batch_data):
                    processed_batches.append(batch_data)
                else:
                    failed_batches += 1
            else:
                failed_batches += 1
        
        # Summary
        total_chunks = sum(batch['num_chunks'] for batch in processed_batches)
        total_duration = sum(batch['total_duration'] for batch in processed_batches)
        
        logger.info("\n" + "="*70)
        logger.info("🎉 BATCH NO-OVERLAP PREPROCESSING COMPLETE!")
        logger.info("="*70)
        logger.info(f"📊 Results:")
        logger.info(f"   Processed samples: {len(processed_batches)}")
        logger.info(f"   Failed samples: {failed_batches}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        logger.info(f"   Average chunks per sample: {total_chunks/len(processed_batches):.1f}")
        logger.info(f"   NO OVERLAPS: ✅ Clean sequential chunks")
        logger.info(f"   Clean boundaries: ✅ Let Mamba learn transitions")
        logger.info(f"   Output directory: {self.output_dir}")
        
        # Show batch structure
        logger.info(f"\n📁 Created batch structure:")
        for batch in processed_batches:
            logger.info(f"   clean_batch_{batch['batch_idx']:02d}/ - {batch['sample_name']} - {batch['num_chunks']} chunks")
        
        return {
            'success': True,
            'processed_batches': len(processed_batches),
            'failed_batches': failed_batches,
            'total_chunks': total_chunks,
            'total_duration': total_duration,
            'average_chunks_per_sample': total_chunks / len(processed_batches) if processed_batches else 0,
            'output_dir': str(self.output_dir),
            'no_overlaps': True,
            'clean_boundaries': True,
            'batch_details': [
                {
                    'batch_idx': batch['batch_idx'],
                    'sample_name': batch['sample_name'],
                    'num_chunks': batch['num_chunks'],
                    'duration': batch['total_duration']
                }
                for batch in processed_batches
            ]
        }


def main():
    """Main function for standalone usage"""
    input_dir = "whisper_transcription"
    
    if not Path(input_dir).exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        logger.info("Expected structure:")
        logger.info("whisper_transcription/")
        logger.info("├── processed_audio/")
        logger.info("│   ├── sample1.wav")
        logger.info("│   ├── sample2.wav")
        logger.info("│   └── ... sample6.wav")
        logger.info("└── transcriptions/")
        logger.info("    ├── sample1_transcription.json")
        logger.info("    ├── sample2_transcription.json")
        logger.info("    └── ... sample6_transcription.json")
        return
    
    try:
        # Create batch preprocessor
        preprocessor = BatchNoOverlapAudioPreprocessor(
            words_per_chunk=15,                    # ~10 seconds per chunk
            input_dir="whisper_transcription",     # Input folder with processed samples
            output_dir="no_overlap_data"           # Output folder with clean batches
        )
        
        # Process all samples
        results = preprocessor.process_all_samples()
        
        if results['success']:
            logger.info("🚀 Ready for NO-OVERLAP sequential training!")
            logger.info("📁 Data structure created:")
            logger.info("   6 samples → 6 clean_batches")
            logger.info("   ~396 total chunks (~66 per batch)")
            logger.info("   NO overlaps, clean boundaries")
            logger.info("   Perfect for Mamba sequential learning!")
        else:
            logger.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Batch preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()