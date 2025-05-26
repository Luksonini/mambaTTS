#!/usr/bin/env python3
"""
Check No-Overlap Data Structure
==============================
Simple script to verify no_overlap_data directory and contents
"""

import json
import torch
from pathlib import Path

def check_no_overlap_data():
    """Check if no_overlap_data exists and is properly structured"""
    
    print("🔍 Checking NO-OVERLAP data structure...")
    print("=" * 60)
    
    # Check main directory
    data_dir = Path("no_overlap_data")
    if not data_dir.exists():
        print("❌ no_overlap_data directory NOT FOUND!")
        print(f"   Looking for: {data_dir.absolute()}")
        
        # Check current directory contents
        current_dir = Path(".")
        print(f"\n📁 Current directory contents:")
        items = list(current_dir.iterdir())
        for item in sorted(items):
            if item.is_dir():
                print(f"   📁 {item.name}/")
            else:
                print(f"   📄 {item.name}")
        
        # Look for similar directories
        data_dirs = [d for d in items if d.is_dir() and "data" in d.name.lower()]
        if data_dirs:
            print(f"\n🔍 Found directories with 'data' in name:")
            for d in data_dirs:
                print(f"   📁 {d.name}/")
        
        return False
    
    print(f"✅ Found no_overlap_data directory: {data_dir.absolute()}")
    
    # Check subdirectories
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    batch_dirs = [d for d in subdirs if d.name.startswith('clean_batch_')]
    
    print(f"📊 Directory structure:")
    print(f"   Total subdirectories: {len(subdirs)}")
    print(f"   Clean batch directories: {len(batch_dirs)}")
    
    if len(batch_dirs) == 0:
        print("❌ No clean_batch_* directories found!")
        print("   Expected directories like: clean_batch_00, clean_batch_01, etc.")
        
        if subdirs:
            print("   Found these subdirectories instead:")
            for d in subdirs[:10]:
                print(f"     📁 {d.name}/")
        
        return False
    
    # Check first few batches
    print(f"\n🔍 Checking batch contents...")
    total_chunks = 0
    
    for i, batch_dir in enumerate(sorted(batch_dirs)[:5]):  # Check first 5 batches
        print(f"   📁 {batch_dir.name}/")
        
        # Check metadata
        meta_path = batch_dir / "batch_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                no_overlaps = meta.get('no_overlaps', False)
                num_chunks = meta.get('num_chunks', 0)
                chunk_files = meta.get('chunk_files', [])
                
                print(f"     ✅ Metadata: {num_chunks} chunks, no_overlaps={no_overlaps}")
                
                # Check chunk files
                existing_chunks = 0
                for chunk_file in chunk_files[:3]:  # Check first 3 chunks
                    chunk_path = batch_dir / chunk_file
                    if chunk_path.exists():
                        try:
                            chunk_data = torch.load(chunk_path, map_location='cpu')
                            clean_chunk = chunk_data.get('clean_chunk', False)
                            has_overlap = chunk_data.get('has_overlap', True)
                            text = chunk_data.get('text', '')[:50]
                            
                            print(f"       ✅ {chunk_file}: clean={clean_chunk}, overlap={has_overlap}")
                            print(f"          Text: '{text}...'")
                            existing_chunks += 1
                            
                        except Exception as e:
                            print(f"       ❌ {chunk_file}: Failed to load - {e}")
                    else:
                        print(f"       ❌ {chunk_file}: File not found")
                
                total_chunks += existing_chunks
                print(f"     📊 Valid chunks: {existing_chunks}/{len(chunk_files)}")
                
            except Exception as e:
                print(f"     ❌ Failed to read metadata: {e}")
        else:
            print(f"     ❌ No batch_meta.json found")
    
    if len(batch_dirs) > 5:
        print(f"   ... and {len(batch_dirs) - 5} more batch directories")
    
    print(f"\n📊 Summary:")
    print(f"   Total batch directories: {len(batch_dirs)}")
    print(f"   Sample chunks checked: {total_chunks}")
    
    if total_chunks > 0:
        print(f"✅ NO-OVERLAP data structure looks good!")
        print(f"   Ready for training with no_overlap_training.py")
        return True
    else:
        print(f"❌ No valid chunks found!")
        print(f"   Check if audio_preprocessor.py completed successfully")
        return False


if __name__ == "__main__":
    success = check_no_overlap_data()
    
    if success:
        print(f"\n🚀 Next steps:")
        print(f"   1. Run: python no_overlap_training.py")
        print(f"   2. Or import NoOverlapDataLoader in your training script")
    else:
        print(f"\n🔧 To fix:")
        print(f"   1. Run: python audio_preprocessor.py")
        print(f"   2. Make sure it completes without errors")
        print(f"   3. Then run this check script again")