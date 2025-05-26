#!/usr/bin/env python3
"""
No-Overlap Audio Preprocessor - Clean Sequential Chunks
======================================================
Creates clean sequential chunks WITHOUT overlaps
Key benefits:
- More training data (no duplicated tokens)
- Simpler training (no masking needed)
- Let Mamba learn natural transitions
- Easier debugging and state management
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


class NoOverlapAudioPreprocessor:
    """
    Clean sequential preprocessor WITHOUT overlaps
    Let Mamba learn natural transitions between chunks
    """
    
    def __init__(self, 
                 words_per_chunk: int = 15,          # Slightly larger chunks since no overlap
                 chunks_per_batch: int = 4,          # Sequential chunks per batch  
                 max_batches: int = 20,
                 output_dir: str = "no_overlap_data"):
        
        self.words_per_chunk = words_per_chunk
        self.chunks_per_batch = chunks_per_batch
        self.max_batches = max_batches
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
        
        logger.info(f"🎯 NoOverlapAudioPreprocessor initialized:")
        logger.info(f"   Words per chunk: {words_per_chunk}")
        logger.info(f"   NO OVERLAPS: Clean sequential chunks ✅")
        logger.info(f"   Chunks per batch: {chunks_per_batch}")
        logger.info(f"   Max batches: {max_batches}")
        logger.info(f"   Output directory: {output_dir}")
        logger.info(f"   🎵 EnCodec ready")
        logger.info(f"   🔤 Tokenizer ready")
    
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
            
            if start < 0 or end <= start or (end - start) > 5.0:
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def create_no_overlap_chunks(self, words: List[Dict]) -> List[List[Dict]]:
        """
        Create clean sequential chunks WITHOUT overlaps
        Each chunk is completely independent
        """
        logger.info("🔄 Creating NO-OVERLAP sequential chunks...")
        
        if len(words) < self.words_per_chunk:
            logger.error(f"❌ Not enough words for chunking: {len(words)} < {self.words_per_chunk}")
            return []
        
        all_batches = []
        current_word_idx = 0
        
        while current_word_idx < len(words):
            # Create one batch (sequence of clean chunks)
            batch_chunks = []
            batch_start_word_idx = current_word_idx
            
            # Create chunks_per_batch sequential chunks WITHOUT overlaps
            for chunk_in_batch in range(self.chunks_per_batch):
                chunk_start = current_word_idx
                chunk_end = current_word_idx + self.words_per_chunk
                
                # Check if we have enough words
                if chunk_end > len(words):
                    if len(batch_chunks) > 0:  # At least one chunk in batch
                        break
                    else:
                        # Not even one full chunk possible
                        break
                
                # Extract chunk words - NO OVERLAP!
                chunk_words = words[chunk_start:chunk_end]
                
                if len(chunk_words) < self.words_per_chunk // 2:  # At least half size
                    break
                
                # Create chunk metadata
                chunk_info = {
                    'words': chunk_words,
                    'chunk_idx_in_batch': chunk_in_batch,
                    'global_chunk_idx': len(all_batches) * self.chunks_per_batch + chunk_in_batch,
                    'start_time': chunk_words[0]['start'],
                    'end_time': chunk_words[-1]['end'],
                    'duration': chunk_words[-1]['end'] - chunk_words[0]['start'],
                    'word_count': len(chunk_words),
                    'has_overlap': False,  # NO OVERLAPS!
                    'overlap_words': 0,    # ZERO overlaps!
                    'clean_chunk': True    # Mark as clean
                }
                
                batch_chunks.append(chunk_info)
                current_word_idx += self.words_per_chunk  # CLEAN advance - no overlap!
            
            # Add batch if it has chunks
            if len(batch_chunks) > 0:
                batch_info = {
                    'batch_idx': len(all_batches),
                    'chunks': batch_chunks,
                    'total_duration': batch_chunks[-1]['end_time'] - batch_chunks[0]['start_time'],
                    'total_words': sum(chunk['word_count'] for chunk in batch_chunks),
                    'start_time': batch_chunks[0]['start_time'],
                    'end_time': batch_chunks[-1]['end_time'],
                    'no_overlaps': True,  # Mark batch as clean
                    'coverage_gap': batch_chunks[1]['start_time'] - batch_chunks[0]['end_time'] if len(batch_chunks) > 1 else 0.0
                }
                
                all_batches.append(batch_info)
                
                logger.info(f"   Batch {len(all_batches)}: {len(batch_chunks)} CLEAN chunks, "
                          f"{batch_info['total_duration']:.1f}s, {batch_info['total_words']} words")
                
                # Show gap between chunks (should be small)
                if len(batch_chunks) > 1:
                    avg_gap = 0
                    for i in range(len(batch_chunks) - 1):
                        gap = batch_chunks[i+1]['start_time'] - batch_chunks[i]['end_time']
                        avg_gap += gap
                    avg_gap /= (len(batch_chunks) - 1)
                    logger.info(f"     Average gap between chunks: {avg_gap:.2f}s")
            else:
                break
        
        logger.info(f"✅ Created {len(all_batches)} NO-OVERLAP sequential batches")
        
        # Analyze coverage
        self._analyze_coverage(all_batches)
        
        return all_batches
    
    def _analyze_coverage(self, batches: List[Dict]):
        """Analyze how much audio is covered without overlaps"""
        logger.info("🔍 Analyzing NO-OVERLAP coverage:")
        
        total_original_duration = 0
        total_covered_duration = 0
        total_gaps = 0
        gap_durations = []
        
        for batch in batches:
            chunks = batch['chunks']
            
            # Calculate coverage for this batch
            batch_start = chunks[0]['start_time']
            batch_end = chunks[-1]['end_time']
            batch_span = batch_end - batch_start
            
            # Calculate actual covered time (sum of chunk durations)
            covered_time = sum(chunk['duration'] for chunk in chunks)
            
            total_original_duration += batch_span
            total_covered_duration += covered_time
            
            # Calculate gaps between chunks
            for i in range(len(chunks) - 1):
                gap = chunks[i+1]['start_time'] - chunks[i]['end_time']
                if gap > 0:
                    gap_durations.append(gap)
                    total_gaps += gap
            
            logger.info(f"   Batch {batch['batch_idx']}: {covered_time:.1f}s covered out of {batch_span:.1f}s span ({covered_time/batch_span*100:.1f}%)")
        
        # Overall statistics
        coverage_ratio = total_covered_duration / total_original_duration if total_original_duration > 0 else 0
        avg_gap = sum(gap_durations) / len(gap_durations) if gap_durations else 0
        
        logger.info(f"📊 NO-OVERLAP Coverage Analysis:")
        logger.info(f"   Total covered: {total_covered_duration:.1f}s out of {total_original_duration:.1f}s ({coverage_ratio*100:.1f}%)")
        logger.info(f"   Total gaps: {total_gaps:.1f}s")
        logger.info(f"   Average gap: {avg_gap:.2f}s")
        logger.info(f"   Number of gaps: {len(gap_durations)}")
        
        if avg_gap > 2.0:
            logger.warning("⚠️  Large gaps detected - consider smaller chunk size")
        elif avg_gap < 0.5:
            logger.info("✅ Small gaps - good chunk continuity")
    
    def process_batch(self, batch_info: Dict, wav_data: torch.Tensor) -> Optional[Dict]:
        """Process one NO-OVERLAP batch"""
        batch_idx = batch_info['batch_idx']
        chunks = batch_info['chunks']
        
        logger.info(f"🎵 Processing NO-OVERLAP batch {batch_idx} ({len(chunks)} chunks)...")
        
        processed_chunks = []
        
        for chunk_info in chunks:
            processed_chunk = self._process_single_chunk(chunk_info, wav_data, batch_idx)
            
            if processed_chunk is not None:
                processed_chunks.append(processed_chunk)
            else:
                logger.warning(f"⚠️  Failed to process chunk {chunk_info['global_chunk_idx']}")
        
        if len(processed_chunks) == 0:
            logger.error(f"❌ No chunks processed in batch {batch_idx}")
            return None
        
        # Create batch-level metadata
        batch_data = {
            'batch_idx': batch_idx,
            'chunks': processed_chunks,
            'num_chunks': len(processed_chunks),
            'total_duration': batch_info['total_duration'],
            'total_words': batch_info['total_words'],
            'no_overlaps': True,           # Mark as clean
            'sequential': True,            # Still sequential
            'clean_boundaries': True       # Clean chunk boundaries
        }
        
        return batch_data
    
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
                'chunk_idx_in_batch': chunk_info['chunk_idx_in_batch'],
                'global_chunk_idx': chunk_info['global_chunk_idx'],
                'batch_idx': batch_idx,
                'word_count': len(words),
                
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
            batch_dir = self.output_dir / f"clean_batch_{batch_idx:02d}"  # Mark as clean
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each chunk individually
            for chunk in batch_data['chunks']:
                chunk_filename = f"chunk_{chunk['chunk_idx_in_batch']:02d}_{chunk['start_time']:.1f}-{chunk['end_time']:.1f}s.pt"
                chunk_path = batch_dir / chunk_filename
                
                torch.save(chunk, chunk_path)
            
            # Save batch metadata
            batch_meta = {
                'batch_idx': batch_idx,
                'num_chunks': batch_data['num_chunks'],
                'total_duration': batch_data['total_duration'],
                'total_words': batch_data['total_words'],
                'no_overlaps': batch_data['no_overlaps'],
                'sequential': batch_data['sequential'],
                'clean_boundaries': batch_data['clean_boundaries'],
                'chunk_files': [f"chunk_{chunk['chunk_idx_in_batch']:02d}_{chunk['start_time']:.1f}-{chunk['end_time']:.1f}s.pt" 
                               for chunk in batch_data['chunks']]
            }
            
            meta_path = batch_dir / "batch_meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(batch_meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 NO-OVERLAP Batch {batch_idx} saved: {batch_data['num_chunks']} chunks, {batch_data['total_duration']:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save batch {batch_data['batch_idx']}: {e}")
            return False
    
    def process_full_audio(self, audio_path: str, whisperx_json_path: str) -> Dict:
        """Main processing function - creates NO-OVERLAP sequential batches"""
        logger.info("🚀 NO-OVERLAP AUDIO PREPROCESSING")
        logger.info("=" * 60)
        
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
        logger.info(f"   Audio duration: {audio_duration:.1f}s")
        
        # Create NO-OVERLAP chunks
        batches = self.create_no_overlap_chunks(words)
        if len(batches) == 0:
            logger.error("❌ No batches created")
            return {'success': False, 'error': 'No batches created'}
        
        # Process each batch
        self.output_dir.mkdir(exist_ok=True)
        processed_batches = []
        failed_batches = 0
        
        for batch_info in batches:
            batch_data = self.process_batch(batch_info, wav)
            
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
        
        logger.info("\n" + "="*60)
        logger.info("🎉 NO-OVERLAP PREPROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"📊 Results:")
        logger.info(f"   Successful batches: {len(processed_batches)}")
        logger.info(f"   Failed batches: {failed_batches}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total duration: {total_duration:.1f}s")
        logger.info(f"   NO OVERLAPS: ✅ Clean sequential chunks")
        logger.info(f"   Clean boundaries: ✅ Let Mamba learn transitions")
        
        return {
            'success': True,
            'processed_batches': len(processed_batches),
            'failed_batches': failed_batches,
            'total_chunks': total_chunks,
            'total_duration': total_duration,
            'output_dir': str(self.output_dir),
            'no_overlaps': True,
            'clean_boundaries': True
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
        # Create NO-OVERLAP preprocessor
        preprocessor = NoOverlapAudioPreprocessor(
            words_per_chunk=15,          # Slightly larger since no overlap waste
            chunks_per_batch=4,          # Sequential chunks per batch
            max_batches=20,              # Max batches
            output_dir="no_overlap_data" # Clean output directory
        )
        
        # Process audio
        results = preprocessor.process_full_audio(audio_path, json_path)
        
        if results['success']:
            logger.info("🚀 Ready for NO-OVERLAP sequential training!")
            logger.info("📁 Output directory: no_overlap_data/")
            logger.info("🎯 Features: Clean chunks, no masking needed, let Mamba learn!")
        else:
            logger.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()