#!/usr/bin/env python3
"""
Safe Data Loader - Clean and Simple
==================================
Handles data loading with comprehensive validation
"""

import torch
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_and_fix_tokens(tokens, vocab_size, context=""):
    """
    Fix token indices to prevent CUDA indexing errors
    """
    if not torch.is_tensor(tokens):
        tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Check bounds
    invalid_mask = (tokens < 0) | (tokens >= vocab_size)
    num_invalid = invalid_mask.sum().item()
    
    if num_invalid > 0:
        logger.warning(f"⚠️  {context}: Fixed {num_invalid} invalid tokens (range: 0-{vocab_size-1})")
        tokens[invalid_mask] = 0  # Set to padding token
    
    return tokens

def validate_audio_codes(audio_codes, codebook_size=1024, max_length=1000):
    """
    Fix audio codes to prevent indexing errors
    """
    if not torch.is_tensor(audio_codes):
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)
    
    # Handle shape
    if audio_codes.dim() == 2:  # [C, T]
        audio_codes = audio_codes.unsqueeze(0)  # [1, C, T]
    
    B, C, T = audio_codes.shape
    
    # Truncate if too long
    if T > max_length:
        logger.warning(f"⚠️  Audio too long: {T} -> {max_length}")
        audio_codes = audio_codes[:, :, :max_length]
    
    # Fix invalid codes
    invalid_mask = (audio_codes < 0) | (audio_codes >= codebook_size)
    num_invalid = invalid_mask.sum().item()
    
    if num_invalid > 0:
        logger.warning(f"⚠️  Fixed {num_invalid} invalid audio codes")
        audio_codes[invalid_mask] = 0
    
    return audio_codes

class SafeDataLoader:
    """
    Clean and safe data loader for preprocessed chunks
    """
    def __init__(self, data_dir="precomputed", vocab_size=131, codebook_size=1024, device="cuda"):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.device = device
        self.chunks = []
        
        self._load_chunks()
        self.stats = self._compute_stats()
        
        logger.info(f"📦 SafeDataLoader: {len(self.chunks)} chunks loaded")
        logger.info(f"   Total duration: {self.stats['total_duration']:.1f}s")
        logger.info(f"   Avg chunk: {self.stats['avg_duration']:.1f}s, {self.stats['avg_words']:.0f} words")
    
    def _load_chunks(self):
        """Load and validate all chunks"""
        batch_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')])
        
        loaded = 0
        failed = 0
        
        for batch_dir in batch_dirs:
            for pt_file in sorted(batch_dir.glob("*.pt")):
                try:
                    chunk = torch.load(pt_file, map_location='cpu', weights_only=False)
                    
                    if self._validate_chunk(chunk, pt_file.name):
                        chunk['filename'] = pt_file.name
                        self.chunks.append(chunk)
                        loaded += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load {pt_file.name}: {e}")
                    failed += 1
        
        random.shuffle(self.chunks)
        logger.info(f"📊 Loaded {loaded} chunks, {failed} failed")
    
    def _validate_chunk(self, chunk, filename):
        """Validate single chunk"""
        try:
            # Required fields
            required = ['text', 'text_tokens', 'audio_codes', 'duration', 'word_count']
            for field in required:
                if field not in chunk:
                    logger.warning(f"⚠️  {filename}: missing {field}")
                    return False
            
            # Basic checks
            if len(chunk['text']) < 3:
                return False
            if chunk['duration'] <= 0 or chunk['duration'] > 30:
                return False
            if chunk['word_count'] <= 0:
                return False
            
            # Fix tokens
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
            
        except Exception as e:
            logger.warning(f"⚠️  {filename}: validation failed: {e}")
            return False
    
    def _compute_stats(self):
        """Compute statistics"""
        if not self.chunks:
            return {'total_duration': 0, 'avg_duration': 0, 'avg_words': 0}
        
        total_duration = sum(c['duration'] for c in self.chunks)
        total_words = sum(c['word_count'] for c in self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'total_duration': total_duration,
            'avg_duration': total_duration / len(self.chunks),
            'avg_words': total_words / len(self.chunks)
        }
    
    def get_random_chunk(self):
        """Get random chunk moved to device"""
        if not self.chunks:
            raise ValueError("No chunks available!")
        
        chunk = random.choice(self.chunks)
        return self._to_device(chunk)
    
    def get_chunk(self, idx):
        """Get specific chunk by index"""
        idx = idx % len(self.chunks)
        chunk = self.chunks[idx]
        return self._to_device(chunk)
    
    def _to_device(self, chunk):
        """Move chunk tensors to device safely"""
        result = {}
        
        for key, value in chunk.items():
            if isinstance(value, torch.Tensor):
                # Final validation before moving to device
                if key == 'text_tokens':
                    value = validate_and_fix_tokens(value, self.vocab_size, f"device/{key}")
                elif key == 'audio_codes':
                    value = validate_audio_codes(value, self.codebook_size)
                
                result[key] = value.to(self.device)
            else:
                result[key] = value
        
        return result
    
    def get_stats(self):
        """Get data statistics"""
        return self.stats