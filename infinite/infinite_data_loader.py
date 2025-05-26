#!/usr/bin/env python3
"""
Infinite Data Loader - Continuous Minutowe Batches
=================================================
Ładuje continuous minutowe batches jako infinite streams
Key features:
- Ładuje pełne ~minutowe batches (nie chunks!)
- Każdy batch = jedna continuous sekwencja
- Virtual checkpoints wbudowane w dane
- Fresh state między batches
"""

import torch
import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class InfiniteDataLoader:
    """
    Data loader dla continuous minutowych batches
    Każdy batch = infinite-like continuous stream
    """
    
    def __init__(self, data_dir="continuous_data", device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = device
        self.continuous_batches = []
        self.batch_metadata = []
        
        logger.info(f"🔍 Loading INFINITE continuous batches from {data_dir}")
        self._load_all_continuous_batches()
        
    def _load_all_continuous_batches(self):
        """Load all continuous minutowe batches"""
        if not self.data_dir.exists():
            logger.error(f"❌ Continuous data directory not found: {self.data_dir}")
            return
            
        batch_dirs = [d for d in self.data_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('continuous_batch_')]
        batch_dirs.sort()
        
        logger.info(f"📁 Found {len(batch_dirs)} continuous batch directories")
        
        for batch_dir in batch_dirs:
            try:
                # Load batch metadata
                meta_path = batch_dir / "batch_meta.json"
                if not meta_path.exists():
                    logger.warning(f"⚠️  No metadata for {batch_dir.name}")
                    continue
                    
                with open(meta_path, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)
                
                # Verify this is a continuous batch
                if not batch_meta.get('continuous_batch', False):
                    logger.warning(f"⚠️  {batch_dir.name} not marked as continuous")
                    continue
                
                # Load the continuous batch file
                batch_file = batch_meta.get('batch_file')
                if not batch_file:
                    logger.warning(f"⚠️  No batch file specified in {batch_dir.name}")
                    continue
                
                batch_path = batch_dir / batch_file
                if not batch_path.exists():
                    logger.warning(f"⚠️  Batch file not found: {batch_file}")
                    continue
                
                try:
                    # Load continuous batch data
                    continuous_batch = torch.load(batch_path, map_location=self.device, weights_only=False)
                    
                    # Verify batch is properly structured
                    if not self._validate_continuous_batch(continuous_batch, batch_dir.name):
                        continue
                    
                    # Add metadata
                    continuous_batch['batch_dir'] = batch_dir.name
                    continuous_batch['batch_meta'] = batch_meta
                    continuous_batch['batch_file'] = batch_file
                    
                    self.continuous_batches.append(continuous_batch)
                    self.batch_metadata.append(batch_meta)
                    
                    logger.info(f"   ✅ Loaded {batch_dir.name}: {continuous_batch['duration']:.1f}s, "
                              f"{continuous_batch['audio_tokens']:,} tokens, "
                              f"{continuous_batch['num_checkpoints']} checkpoints")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load {batch_file}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"⚠️  Failed to process {batch_dir.name}: {e}")
                continue
        
        logger.info(f"📊 Infinite Data Loader statistics:")
        logger.info(f"   Total continuous batches: {len(self.continuous_batches)}")
        
        if len(self.continuous_batches) > 0:
            total_duration = sum(batch['duration'] for batch in self.continuous_batches)
            total_tokens = sum(batch['audio_tokens'] for batch in self.continuous_batches)
            total_checkpoints = sum(batch['num_checkpoints'] for batch in self.continuous_batches)
            avg_duration = total_duration / len(self.continuous_batches)
            
            logger.info(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
            logger.info(f"   Total audio tokens: {total_tokens:,}")
            logger.info(f"   Total virtual checkpoints: {total_checkpoints}")
            logger.info(f"   Average batch duration: {avg_duration:.1f}s")
            logger.info(f"   🎯 INFINITE continuous streams ready!")
        else:
            logger.error("❌ No valid continuous batches loaded!")
    
    def _validate_continuous_batch(self, batch_data: Dict, batch_name: str) -> bool:
        """Validate that batch is properly structured for infinite processing"""
        required_fields = [
            'full_text', 'full_text_tokens', 'continuous_audio_codes',
            'duration', 'virtual_checkpoints', 'continuous_batch',
            'no_internal_chunking', 'audio_tokens'
        ]
        
        for field in required_fields:
            if field not in batch_data:
                logger.warning(f"⚠️  {batch_name} missing field: {field}")
                return False
        
        # Verify it's actually continuous
        if not batch_data.get('continuous_batch', False):
            logger.warning(f"⚠️  {batch_name} not marked as continuous batch")
            return False
        
        if not batch_data.get('no_internal_chunking', False):
            logger.warning(f"⚠️  {batch_name} has internal chunking - not infinite!")
            return False
        
        # Check data shapes
        try:
            text_tokens = batch_data['full_text_tokens']
            audio_codes = batch_data['continuous_audio_codes']
            
            if not torch.is_tensor(text_tokens) or not torch.is_tensor(audio_codes):
                logger.warning(f"⚠️  {batch_name} data not tensors")
                return False
            
            if text_tokens.dim() != 1 or audio_codes.dim() != 2:
                logger.warning(f"⚠️  {batch_name} unexpected tensor shapes")
                return False
            
            if len(batch_data['virtual_checkpoints']) < 1:
                logger.warning(f"⚠️  {batch_name} no virtual checkpoints")
                return False
            
            # Check duration makes sense
            duration = batch_data['duration']
            if duration < 30 or duration > 120:  # 30s - 2min reasonable range
                logger.warning(f"⚠️  {batch_name} unusual duration: {duration:.1f}s")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️  {batch_name} validation error: {e}")
            return False
        
        return True
    
    def get_random_continuous_batch(self) -> Optional[Dict]:
        """Get random continuous batch - represents infinite stream segment"""
        if len(self.continuous_batches) == 0:
            return None
        
        batch_idx = np.random.randint(0, len(self.continuous_batches))
        batch = self.continuous_batches[batch_idx].copy()
        
        # Add selection metadata
        batch['selected_batch_idx'] = batch_idx
        batch['infinite_stream_segment'] = True
        
        return batch
    
    def get_continuous_batch(self, batch_idx: int) -> Optional[Dict]:
        """Get specific continuous batch by index"""
        if batch_idx < 0 or batch_idx >= len(self.continuous_batches):
            return None
        
        batch = self.continuous_batches[batch_idx].copy()
        batch['selected_batch_idx'] = batch_idx
        batch['infinite_stream_segment'] = True
        
        return batch
    
    def get_batch_by_duration(self, min_duration: float = 50.0, max_duration: float = 70.0) -> Optional[Dict]:
        """Get batch within specific duration range"""
        suitable_batches = []
        
        for i, batch in enumerate(self.continuous_batches):
            duration = batch['duration']
            if min_duration <= duration <= max_duration:
                suitable_batches.append((i, batch))
        
        if len(suitable_batches) == 0:
            logger.warning(f"⚠️  No batches found in duration range {min_duration:.1f}s - {max_duration:.1f}s")
            return None
        
        # Select random suitable batch
        selected_idx, selected_batch = suitable_batches[np.random.randint(0, len(suitable_batches))]
        batch = selected_batch.copy()
        batch['selected_batch_idx'] = selected_idx
        batch['infinite_stream_segment'] = True
        batch['duration_filtered'] = True
        
        return batch
    
    def iterate_all_batches(self):
        """Iterator over all continuous batches - for full dataset training"""
        for i, batch in enumerate(self.continuous_batches):
            batch_copy = batch.copy()
            batch_copy['selected_batch_idx'] = i
            batch_copy['infinite_stream_segment'] = True
            batch_copy['full_dataset_iteration'] = True
            yield batch_copy
    
    def get_virtual_checkpoint_info(self, batch_data: Dict) -> List[Dict]:
        """Get detailed virtual checkpoint information for training"""
        if 'virtual_checkpoints' not in batch_data:
            return []
        
        checkpoints = batch_data['virtual_checkpoints']
        audio_tokens_total = batch_data['audio_tokens']
        duration = batch_data['duration']
        
        checkpoint_info = []
        for i, checkpoint in enumerate(checkpoints):
            # Calculate token ranges for this checkpoint
            if i == 0:
                start_token = 0
            else:
                start_token = checkpoints[i-1].get('token_position', 0)
            
            end_token = checkpoint.get('token_position', audio_tokens_total)
            
            checkpoint_info.append({
                'checkpoint_idx': i,
                'start_token': start_token,
                'end_token': end_token,
                'token_count': end_token - start_token,
                'time_start': start_token / 75.0,  # Assuming 75 tokens/s
                'time_end': end_token / 75.0,
                'duration': (end_token - start_token) / 75.0,
                'expected_time': checkpoint.get('expected_time', end_token / 75.0)
            })
        
        return checkpoint_info
    
    def prepare_batch_for_training(self, batch_data: Dict) -> Dict:
        """Prepare continuous batch for infinite-style training"""
        # Ensure tensors are on correct device
        text_tokens = batch_data['full_text_tokens'].to(self.device)
        audio_codes = batch_data['continuous_audio_codes'].to(self.device)
        
        # Add batch dimension if needed
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.unsqueeze(0)  # [1, T_text]
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(0)  # [1, C, T_audio]
        
        # Get virtual checkpoint info
        checkpoint_info = self.get_virtual_checkpoint_info(batch_data)
        
        training_batch = {
            # Core data for training
            'text_tokens': text_tokens,
            'audio_codes': audio_codes,
            'full_text': batch_data['full_text'],
            'duration': batch_data['duration'],
            
            # Infinite processing metadata
            'continuous_batch': True,
            'infinite_stream': True,
            'no_internal_chunking': True,
            'fresh_state_per_batch': True,
            
            # Virtual checkpoint information
            'virtual_checkpoints': checkpoint_info,
            'num_checkpoints': len(checkpoint_info),
            'checkpoint_interval': batch_data.get('checkpoint_interval', 10.0),
            
            # Statistics
            'audio_tokens': batch_data['audio_tokens'],
            'text_token_count': text_tokens.shape[1],
            'tokens_per_second': batch_data.get('tokens_per_second_actual', 75.0),
            
            # Metadata
            'batch_idx': batch_data.get('batch_idx', 0),
            'batch_dir': batch_data.get('batch_dir', 'unknown'),
            'selected_batch_idx': batch_data.get('selected_batch_idx', 0)
        }
        
        return training_batch
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        if len(self.continuous_batches) == 0:
            return {
                'total_batches': 0,
                'total_duration': 0,
                'total_tokens': 0,
                'infinite_ready': False
            }
        
        total_duration = sum(batch['duration'] for batch in self.continuous_batches)
        total_tokens = sum(batch['audio_tokens'] for batch in self.continuous_batches)
        total_checkpoints = sum(batch['num_checkpoints'] for batch in self.continuous_batches)
        
        durations = [batch['duration'] for batch in self.continuous_batches]
        tokens_per_batch = [batch['audio_tokens'] for batch in self.continuous_batches]
        
        return {
            'total_batches': len(self.continuous_batches),
            'total_duration': total_duration,
            'total_minutes': total_duration / 60.0,
            'total_tokens': total_tokens,
            'total_checkpoints': total_checkpoints,
            'avg_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'avg_tokens_per_batch': np.mean(tokens_per_batch),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'infinite_ready': True,
            'continuous_processing': True,
            'no_chunking': True
        }
    
    def __len__(self):
        """Return number of continuous batches available"""
        return len(self.continuous_batches)
    
    def __getitem__(self, idx):
        """Support indexing"""
        return self.get_continuous_batch(idx)


def test_infinite_data_loader():
    """Test the infinite data loader"""
    logger.info("🧪 Testing Infinite Data Loader")
    logger.info("=" * 50)
    
    # Test loading
    loader = InfiniteDataLoader("continuous_data", device='cpu')
    
    if len(loader) == 0:
        logger.error("❌ No continuous batches loaded")
        return
    
    # Show stats
    stats = loader.get_stats()
    logger.info(f"📊 Loader Statistics:")
    logger.info(f"   Total batches: {stats['total_batches']}")
    logger.info(f"   Total duration: {stats['total_duration']:.1f}s ({stats['total_minutes']:.1f} min)")
    logger.info(f"   Total tokens: {stats['total_tokens']:,}")
    logger.info(f"   Average duration: {stats['avg_duration']:.1f}s ± {stats['std_duration']:.1f}s")
    logger.info(f"   Infinite ready: {stats['infinite_ready']}")
    
    # Test random batch
    logger.info(f"\n🎲 Testing random batch selection:")
    random_batch = loader.get_random_continuous_batch()
    
    if random_batch:
        logger.info(f"   Batch: {random_batch['batch_dir']}")
        logger.info(f"   Duration: {random_batch['duration']:.1f}s")
        logger.info(f"   Audio tokens: {random_batch['audio_tokens']:,}")
        logger.info(f"   Virtual checkpoints: {random_batch['num_checkpoints']}")
        logger.info(f"   Continuous: {random_batch['continuous_batch']}")
        logger.info(f"   No chunking: {random_batch['no_internal_chunking']}")
        
        # Test preparation for training
        training_batch = loader.prepare_batch_for_training(random_batch)
        logger.info(f"   Training shapes:")
        logger.info(f"     Text tokens: {training_batch['text_tokens'].shape}")
        logger.info(f"     Audio codes: {training_batch['audio_codes'].shape}")
        logger.info(f"     Checkpoints: {len(training_batch['virtual_checkpoints'])}")
    
    # Test duration filtering
    logger.info(f"\n⏱️  Testing duration filtering:")
    filtered_batch = loader.get_batch_by_duration(55.0, 65.0)
    
    if filtered_batch:
        logger.info(f"   Found batch with duration: {filtered_batch['duration']:.1f}s")
    else:
        logger.info(f"   No batches found in 55-65s range")
    
    # Test iteration
    logger.info(f"\n🔄 Testing batch iteration:")
    for i, batch in enumerate(loader.iterate_all_batches()):
        logger.info(f"   Batch {i}: {batch['batch_dir']}, {batch['duration']:.1f}s")
        if i >= 2:  # Show first 3
            logger.info(f"   ... and {len(loader) - 3} more batches")
            break
    
    logger.info(f"\n✅ Infinite Data Loader test complete!")


if __name__ == "__main__":
    test_infinite_data_loader()