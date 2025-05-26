#!/usr/bin/env python3
"""
Sequential Batch Data Loader - Enhanced for Long Context
========================================================
Load batches sequentially with overlap awareness
"""

import torch
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class SequentialBatchDataLoader:
    """
    Enhanced data loader for sequential batch processing
    Key features:
    - Load entire batches in order
    - Track chunk overlaps
    - Maintain batch coherence
    """
    
    def __init__(self, data_dir="precomputed", vocab_size=131, codebook_size=1024, device="cuda"):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.device = device
        
        # Discover batch structure
        self.batches = self._discover_batches()
        self.batch_stats = self._analyze_batches()
        
        self.overlap_words = 3  # Expected overlap between chunks
        
        logger.info(f"📦 SequentialBatchDataLoader initialized")
        logger.info(f"   Found {len(self.batches)} batches")
        logger.info(f"   Total chunks: {sum(len(chunks) for chunks in self.batches.values())}")
        logger.info(f"   Overlap words: {self.overlap_words}")
    
    def _discover_batches(self) -> Dict[str, List[Path]]:
        """Discover all batch directories and their chunks"""
        batches = {}
        
        batch_dirs = sorted([d for d in self.data_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('batch_')])
        
        for batch_dir in batch_dirs:
            batch_name = batch_dir.name
            chunk_files = sorted(batch_dir.glob("*.pt"))
            
            if len(chunk_files) > 0:
                batches[batch_name] = chunk_files
                logger.info(f"   {batch_name}: {len(chunk_files)} chunks")
        
        return batches
    
    def _analyze_batches(self) -> Dict[str, Dict]:
        """Analyze batch statistics"""
        stats = {}
        
        for batch_name, chunk_files in self.batches.items():
            batch_duration = 0
            batch_words = 0
            chunk_info = []
            
            for chunk_file in chunk_files:
                try:
                    chunk = torch.load(chunk_file, map_location='cpu', weights_only=False)
                    
                    duration = chunk.get('duration', 0)
                    word_count = chunk.get('word_count', 0)
                    text = chunk.get('text', '')
                    
                    batch_duration += duration
                    batch_words += word_count
                    
                    # Extract timing info from filename
                    filename = chunk_file.stem
                    if '_' in filename:
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            timing_part = parts[-1]  # e.g., "0.0-5.2"
                            if '-' in timing_part:
                                start_str, end_str = timing_part.split('-')
                                try:
                                    start_time = float(start_str)
                                    end_time = float(end_str.replace('s', ''))
                                    
                                    chunk_info.append({
                                        'file': chunk_file,
                                        'start_time': start_time,
                                        'end_time': end_time,
                                        'duration': duration,
                                        'word_count': word_count,
                                        'text_preview': text[:50] + "..." if len(text) > 50 else text
                                    })
                                except ValueError:
                                    pass
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to analyze {chunk_file}: {e}")
            
            # Sort by start time for proper sequence
            chunk_info.sort(key=lambda x: x.get('start_time', 0))
            
            stats[batch_name] = {
                'total_duration': batch_duration,
                'total_words': batch_words,
                'num_chunks': len(chunk_files),
                'chunks': chunk_info,
                'avg_chunk_duration': batch_duration / len(chunk_files) if chunk_files else 0
            }
        
        return stats
    
    def get_batch_info(self) -> Dict:
        """Get detailed batch information"""
        return self.batch_stats
    
    def print_batch_analysis(self):
        """Print detailed batch analysis"""
        logger.info("\n📊 BATCH ANALYSIS:")
        logger.info("=" * 60)
        
        for batch_name, stats in self.batch_stats.items():
            logger.info(f"\n📁 {batch_name}:")
            logger.info(f"   Duration: {stats['total_duration']:.1f}s")
            logger.info(f"   Words: {stats['total_words']}")
            logger.info(f"   Chunks: {stats['num_chunks']}")
            logger.info(f"   Avg chunk: {stats['avg_chunk_duration']:.1f}s")
            
            if len(stats['chunks']) > 0:
                logger.info(f"   Sequence:")
                for i, chunk in enumerate(stats['chunks'][:3]):  # Show first 3
                    logger.info(f"     {i+1}. {chunk['start_time']:.1f}-{chunk['end_time']:.1f}s: '{chunk['text_preview']}'")
                
                if len(stats['chunks']) > 3:
                    logger.info(f"     ... and {len(stats['chunks']) - 3} more chunks")
                
                # Check for overlaps
                overlaps = []
                for i in range(len(stats['chunks']) - 1):
                    current_end = stats['chunks'][i]['end_time']
                    next_start = stats['chunks'][i + 1]['start_time']
                    
                    if current_end > next_start:
                        overlap_duration = current_end - next_start
                        overlaps.append(overlap_duration)
                
                if overlaps:
                    avg_overlap = sum(overlaps) / len(overlaps)
                    logger.info(f"   Overlaps: {len(overlaps)} found, avg {avg_overlap:.1f}s")
                else:
                    logger.info(f"   Overlaps: None detected")
    
    def load_batch_chunks(self, batch_name: str, validate: bool = True) -> List[Dict]:
        """
        Load all chunks from a specific batch in chronological order
        
        Args:
            batch_name: Name of batch (e.g., "batch_01")
            validate: Whether to validate chunk data
            
        Returns:
            List of chunk data dictionaries in chronological order
        """
        if batch_name not in self.batches:
            logger.error(f"❌ Batch not found: {batch_name}")
            return []
        
        chunks = []
        batch_stats = self.batch_stats[batch_name]
        
        # Load chunks in chronological order
        for chunk_info in batch_stats['chunks']:
            chunk_file = chunk_info['file']
            
            try:
                chunk = torch.load(chunk_file, map_location='cpu', weights_only=False)
                
                if validate and not self._validate_chunk(chunk, chunk_file.name):
                    logger.warning(f"⚠️  Invalid chunk: {chunk_file.name}")
                    continue
                
                # Add batch metadata
                chunk['batch_name'] = batch_name
                chunk['chunk_index'] = len(chunks)
                chunk['start_time'] = chunk_info['start_time']
                chunk['end_time'] = chunk_info['end_time']
                chunk['file_path'] = chunk_file
                
                # Move to device
                chunk = self._to_device(chunk)
                chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {chunk_file}: {e}")
        
        logger.info(f"📦 Loaded {len(chunks)} chunks from {batch_name}")
        
        # Analyze overlaps in loaded data
        if len(chunks) > 1:
            self._analyze_chunk_overlaps(chunks)
        
        return chunks
    
    def _analyze_chunk_overlaps(self, chunks: List[Dict]):
        """Analyze word overlaps between consecutive chunks"""
        logger.info(f"🔍 Analyzing chunk overlaps...")
        
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            current_text = current_chunk.get('text', '')
            next_text = next_chunk.get('text', '')
            
            if current_text and next_text:
                # Simple word-based overlap detection
                current_words = current_text.split()
                next_words = next_text.split()
                
                # Check for overlap at the end of current and start of next
                max_check = min(10, len(current_words), len(next_words))
                overlap_count = 0
                
                for j in range(1, max_check + 1):
                    current_suffix = current_words[-j:]
                    next_prefix = next_words[:j]
                    
                    if current_suffix == next_prefix:
                        overlap_count = j
                
                logger.info(f"   Chunks {i}-{i+1}: {overlap_count} word overlap")
                
                if overlap_count > 0:
                    overlap_words = current_words[-overlap_count:]
                    logger.info(f"     Overlap: {' '.join(overlap_words)}")
    
    def _validate_chunk(self, chunk: Dict, filename: str) -> bool:
        """Validate chunk data"""
        required = ['text', 'text_tokens', 'audio_codes', 'duration', 'word_count']
        
        for field in required:
            if field not in chunk:
                return False
        
        # Basic checks
        if len(chunk['text']) < 3:
            return False
        if chunk['duration'] <= 0:
            return False
        if chunk['word_count'] <= 0:
            return False
        
        # Fix tokens and audio codes
        from data_loader import validate_and_fix_tokens, validate_audio_codes
        
        chunk['text_tokens'] = validate_and_fix_tokens(
            chunk['text_tokens'], 
            self.vocab_size, 
            f"{filename}/text"
        )
        
        chunk['audio_codes'] = validate_audio_codes(
            chunk['audio_codes'], 
            self.codebook_size
        )
        
        return True
    
    def _to_device(self, chunk: Dict) -> Dict:
        """Move chunk tensors to device"""
        result = {}
        
        for key, value in chunk.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        
        return result
    
    def get_random_batch(self) -> str:
        """Get random batch name"""
        return random.choice(list(self.batches.keys()))
    
    def get_batch_names(self) -> List[str]:
        """Get all batch names"""
        return list(self.batches.keys())
    
    def create_sequential_context(self, chunks: List[Dict]) -> Dict:
        """
        Create sequential context from batch chunks
        
        Args:
            chunks: List of chunks from same batch
            
        Returns:
            Context information for sequential processing
        """
        if not chunks:
            return {}
        
        # Build full text sequence
        full_text_parts = []
        full_tokens = []
        chunk_boundaries = []  # (start_pos, end_pos) for each chunk
        
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            tokens = chunk.get('text_tokens', [])
            
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().tolist()
            
            # Handle overlap (skip overlapping words except for first chunk)
            if i > 0:
                # Estimate overlap tokens (simple heuristic)
                overlap_tokens = min(self.overlap_words * 2, len(tokens) // 4)  # Conservative estimate
                tokens = tokens[overlap_tokens:]
                
                # Also adjust text (rough approximation)
                words = text.split()
                if len(words) > self.overlap_words:
                    text = ' '.join(words[self.overlap_words:])
            
            # Record boundaries
            chunk_boundaries.append((current_pos, current_pos + len(tokens)))
            
            # Accumulate
            full_text_parts.append(text)
            full_tokens.extend(tokens)
            current_pos += len(tokens)
        
        # Combine full text
        full_text = ' '.join(full_text_parts)
        
        return {
            'full_text': full_text,
            'full_tokens': full_tokens,
            'chunk_boundaries': chunk_boundaries,
            'num_chunks': len(chunks),
            'total_duration': sum(chunk.get('duration', 0) for chunk in chunks),
            'context_info': {
                'batch_name': chunks[0].get('batch_name', 'unknown'),
                'start_time': chunks[0].get('start_time', 0),
                'end_time': chunks[-1].get('end_time', 0),
                'total_length': len(full_tokens)
            }
        }


def test_sequential_loader():
    """Test the sequential batch data loader"""
    logger.info("🧪 Testing Sequential Batch Data Loader")
    logger.info("=" * 50)
    
    try:
        # Initialize loader
        loader = SequentialBatchDataLoader()
        
        # Print analysis
        loader.print_batch_analysis()
        
        # Test loading a batch
        batch_names = loader.get_batch_names()
        if batch_names:
            test_batch = batch_names[0]
            logger.info(f"\n🧪 Testing batch: {test_batch}")
            
            chunks = loader.load_batch_chunks(test_batch)
            
            if chunks:
                logger.info(f"✅ Loaded {len(chunks)} chunks")
                
                # Create sequential context
                context = loader.create_sequential_context(chunks)
                
                logger.info(f"📝 Sequential context:")
                logger.info(f"   Full text length: {len(context['full_text'])} chars")
                logger.info(f"   Full tokens: {len(context['full_tokens'])} tokens") 
                logger.info(f"   Chunk boundaries: {context['chunk_boundaries']}")
                logger.info(f"   Total duration: {context['total_duration']:.1f}s")
                
                logger.info(f"\n📋 Full text preview:")
                preview = context['full_text'][:200] + "..." if len(context['full_text']) > 200 else context['full_text']
                logger.info(f"   '{preview}'")
                
                return True
            else:
                logger.error("❌ No chunks loaded")
                return False
        else:
            logger.error("❌ No batches found")
            return False
    
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_sequential_loader()