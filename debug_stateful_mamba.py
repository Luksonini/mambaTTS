#!/usr/bin/env python3
"""
Debug Script for Stateful Mamba Issues
======================================
Comprehensive debugging of the stateful system
"""

import torch
import logging
import traceback
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_data_loading():
    """Debug NO-OVERLAP data loading"""
    logger.info("🔍 DEBUG: NO-OVERLAP Data Loading")
    logger.info("=" * 50)
    
    data_dir = Path("no_overlap_data")
    
    if not data_dir.exists():
        logger.error("❌ no_overlap_data directory not found!")
        return False
    
    # Check directory structure
    batch_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('clean_batch_')]
    logger.info(f"📁 Found {len(batch_dirs)} batch directories")
    
    for i, batch_dir in enumerate(batch_dirs[:3]):  # Check first 3
        logger.info(f"\n📂 {batch_dir.name}:")
        
        # Check metadata
        meta_path = batch_dir / "batch_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                logger.info(f"   ✅ Metadata: {meta.get('num_chunks', 0)} chunks")
                logger.info(f"   ✅ NO-OVERLAP: {meta.get('no_overlaps', False)}")
                logger.info(f"   ✅ Sequential: {meta.get('sequential', False)}")
            except Exception as e:
                logger.error(f"   ❌ Metadata error: {e}")
                continue
        else:
            logger.error(f"   ❌ No metadata file")
            continue
        
        # Check chunk files
        chunk_files = list(batch_dir.glob("*.pt"))
        logger.info(f"   📄 Found {len(chunk_files)} .pt files")
        
        # Test loading first chunk
        if chunk_files:
            try:
                first_chunk = torch.load(chunk_files[0], map_location='cpu', weights_only=False)
                
                logger.info(f"   🧪 First chunk test:")
                logger.info(f"      Text: '{first_chunk.get('text', 'N/A')[:50]}...'")
                logger.info(f"      Duration: {first_chunk.get('duration', 0):.2f}s")
                logger.info(f"      Clean chunk: {first_chunk.get('clean_chunk', False)}")
                logger.info(f"      Has overlap: {first_chunk.get('has_overlap', True)}")
                logger.info(f"      Word count: {first_chunk.get('word_count', 0)}")
                
                # Check tensors
                text_tokens = first_chunk.get('text_tokens')
                audio_codes = first_chunk.get('audio_codes')
                
                if text_tokens is not None:
                    logger.info(f"      Text tokens: {text_tokens.shape} (dtype: {text_tokens.dtype})")
                else:
                    logger.error(f"      ❌ No text_tokens")
                
                if audio_codes is not None:
                    logger.info(f"      Audio codes: {audio_codes.shape} (dtype: {audio_codes.dtype})")
                else:
                    logger.error(f"      ❌ No audio_codes")
                
            except Exception as e:
                logger.error(f"   ❌ Chunk loading error: {e}")
                traceback.print_exc()
    
    return len(batch_dirs) > 0

def debug_stateful_modules():
    """Debug stateful module initialization"""
    logger.info("\n🧠 DEBUG: Stateful Modules")
    logger.info("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    try:
        # Test imports
        logger.info("📦 Testing imports...")
        from stateful_modules import (
            StatefulMambaTextEncoder,
            StatefulMambaAudioProcessor, 
            StatefulDurationRegulator,
            AudioStyleExtractor,
            BatchStateManager,
            StreamingInferenceManager
        )
        logger.info("   ✅ All imports successful")
        
        # Test StatefulSSM core
        logger.info("🔧 Testing StatefulSSM...")
        from stateful_modules import StatefulSSM
        ssm = StatefulSSM(dim=128, state_size=64).to(device)
        
        test_input = torch.randn(2, 10, 128, device=device)
        output1 = ssm(test_input)
        logger.info(f"   ✅ SSM forward: {test_input.shape} -> {output1.shape}")
        
        # Test with state
        output2, state = ssm(test_input, return_state=True)
        logger.info(f"   ✅ SSM with state: output {output2.shape}, state {state.shape}")
        
        # Test state continuation
        output3, new_state = ssm(test_input, initial_state=state, return_state=True)
        logger.info(f"   ✅ SSM state continuation: output {output3.shape}, new_state {new_state.shape}")
        
        # Test Text Encoder
        logger.info("📝 Testing StatefulMambaTextEncoder...")
        text_encoder = StatefulMambaTextEncoder(vocab_size=131, embed_dim=128, state_size=64).to(device)
        
        tokens = torch.randint(0, 131, (2, 15), device=device)
        text_features1 = text_encoder(tokens, return_sequence=True)
        logger.info(f"   ✅ Text features: {tokens.shape} -> {text_features1.shape}")
        
        # Test with states
        text_features2, states = text_encoder(tokens, return_states=True, return_sequence=True)
        logger.info(f"   ✅ Text with states: features {text_features2.shape}, {len(states)} states")
        
        # Test state shapes
        for i, state in enumerate(states):
            if state is not None:
                logger.info(f"      State {i}: {state.shape}")
            else:
                logger.info(f"      State {i}: None")
        
        # Test Audio Processor
        logger.info("🎵 Testing StatefulMambaAudioProcessor...")
        audio_processor = StatefulMambaAudioProcessor(
            hidden_dim=256, num_codebooks=4, codebook_size=1024, state_size=64
        ).to(device)
        
        audio_tokens = torch.randint(0, 1024, (2, 4, 20), device=device)
        text_context = torch.randn(2, 256, device=device)
        
        audio_logits1 = audio_processor(audio_tokens, text_context)
        logger.info(f"   ✅ Audio logits: {audio_tokens.shape} -> {audio_logits1.shape}")
        
        # Test with states
        audio_logits2, audio_states = audio_processor(
            audio_tokens, text_context, return_states=True
        )
        logger.info(f"   ✅ Audio with states: logits {audio_logits2.shape}, {len(audio_states)} states")
        
        # Test Duration Regulator
        logger.info("⏱️  Testing StatefulDurationRegulator...")
        duration_reg = StatefulDurationRegulator(
            text_dim=128, style_dim=64, hidden_dim=128, state_size=64
        ).to(device)
        
        text_features = torch.randn(2, 15, 128, device=device)
        style_emb = torch.randn(2, 64, device=device)
        
        regulated, durations, duration_tokens, confidence = duration_reg(text_features, style_emb)
        logger.info(f"   ✅ Duration regulation:")
        logger.info(f"      Regulated: {regulated.shape}")
        logger.info(f"      Durations: {durations.shape}")
        logger.info(f"      Duration tokens: {duration_tokens.shape}")
        logger.info(f"      Confidence: {confidence.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Stateful modules error: {e}")
        traceback.print_exc()
        return False

def debug_training_data_loader():
    """Debug NoOverlapDataLoader"""
    logger.info("\n📦 DEBUG: NoOverlapDataLoader")
    logger.info("=" * 50)
    
    try:
        from sequential_trainer import NoOverlapDataLoader
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_loader = NoOverlapDataLoader(data_dir="no_overlap_data", device=device)
        
        stats = data_loader.get_stats()
        logger.info(f"📊 Data loader stats:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
        
        if stats['total_chunks'] > 0:
            # Test random chunk
            logger.info("🎲 Testing random chunk...")
            chunk = data_loader.get_random_chunk()
            
            if chunk:
                logger.info(f"   ✅ Random chunk loaded:")
                logger.info(f"      Text: '{chunk.get('text', '')[:50]}...'")
                logger.info(f"      Duration: {chunk.get('duration', 0):.2f}s")
                logger.info(f"      Batch dir: {chunk.get('batch_dir', 'N/A')}")
                logger.info(f"      Clean chunk: {chunk.get('clean_chunk', False)}")
                
                # Test tensor shapes
                text_tokens = chunk.get('text_tokens')
                audio_codes = chunk.get('audio_codes')
                
                if text_tokens is not None:
                    logger.info(f"      Text tokens: {text_tokens.shape}")
                if audio_codes is not None:
                    logger.info(f"      Audio codes: {audio_codes.shape}")
            else:
                logger.error("   ❌ No chunk returned")
                return False
            
            # Test sequential batch
            logger.info("🔄 Testing sequential batch...")
            batch = data_loader.get_sequential_batch()
            
            if batch:
                logger.info(f"   ✅ Sequential batch loaded:")
                logger.info(f"      Batch idx: {batch.get('batch_idx', 'N/A')}")
                logger.info(f"      Chunks: {len(batch.get('chunks', []))}")
                logger.info(f"      No overlaps: {batch.get('no_overlaps', False)}")
                logger.info(f"      Clean sequential: {batch.get('clean_sequential', False)}")
            else:
                logger.error("   ❌ No batch returned")
                return False
        else:
            logger.error("❌ No chunks available")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader error: {e}")
        traceback.print_exc()
        return False

def debug_model_creation():
    """Debug StatefulTTSModel creation"""
    logger.info("\n🎯 DEBUG: StatefulTTSModel")
    logger.info("=" * 50)
    
    try:
        from nucleotide_tokenizer import NucleotideTokenizer
        from sequential_trainer import StatefulTTSModel
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test tokenizer
        logger.info("🔤 Testing tokenizer...")
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"   ✅ Tokenizer: vocab_size={vocab_size}")
        
        # Test model creation
        logger.info("🏗️  Creating StatefulTTSModel...")
        model = StatefulTTSModel(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_codebooks=4,
            codebook_size=1024,
            state_size=64
        ).to(device)
        
        logger.info(f"   ✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        logger.info("🔄 Testing model forward pass...")
        text_tokens = torch.randint(0, vocab_size, (2, 15), device=device)
        audio_tokens = torch.randint(0, 1024, (2, 4, 20), device=device)
        
        output = model(text_tokens, audio_tokens)
        
        logger.info(f"   ✅ Forward pass successful:")
        logger.info(f"      Input: text {text_tokens.shape}, audio {audio_tokens.shape}")
        
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"      {key}: {value.shape}")
            else:
                logger.info(f"      {key}: {type(value)}")
        
        # Test with states
        logger.info("🔄 Testing with state management...")
        output_with_states = model(
            text_tokens, audio_tokens, 
            return_states=True
        )
        
        states = output_with_states.get('states', {})
        logger.info(f"   ✅ States returned: {len(states)} types")
        for state_name, state_list in states.items():
            if state_list:
                logger.info(f"      {state_name}: {len(state_list)} states")
                for i, state in enumerate(state_list):
                    if state is not None:
                        logger.info(f"         State {i}: {state.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation error: {e}")
        traceback.print_exc()
        return False

def debug_loss_computation():
    """Debug loss computation"""
    logger.info("\n📊 DEBUG: Loss Computation")
    logger.info("=" * 50)
    
    try:
        from losses import compute_combined_loss
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy model output
        model_output = {
            'logits': torch.randn(1, 4, 20, 1024, device=device),
            'predicted_durations': torch.rand(1, 15, device=device) * 0.3 + 0.05,
            'duration_confidence': torch.rand(1, 15, device=device)
        }
        
        # Create dummy chunk data
        chunk_data = {
            'audio_codes': torch.randint(0, 1024, (4, 20), device=device),
            'duration': 4.0,
            'text': "Test text for duration calculation",
            'word_count': 6
        }
        
        # Create dummy text tokens
        text_tokens = torch.randint(0, 131, (1, 15), device=device)
        
        logger.info("🧮 Testing loss computation...")
        loss_dict = compute_combined_loss(model_output, chunk_data, text_tokens, device)
        
        logger.info("   ✅ Loss computation successful:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"      {key}: {value.item():.6f}")
            else:
                logger.info(f"      {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Loss computation error: {e}")
        traceback.print_exc()
        return False

def debug_training_step():
    """Debug single training step"""
    logger.info("\n🏃 DEBUG: Training Step")
    logger.info("=" * 50)
    
    try:
        from sequential_trainer import NoOverlapDataLoader, StatefulTTSModel, NoOverlapTrainer
        from nucleotide_tokenizer import NucleotideTokenizer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        data_loader = NoOverlapDataLoader(data_dir="no_overlap_data", device=device)
        
        model = StatefulTTSModel(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_codebooks=4,
            codebook_size=1024,
            state_size=64
        ).to(device)
        
        trainer = NoOverlapTrainer(model, tokenizer, data_loader)
        
        logger.info("🏋️ Testing single chunk training...")
        
        # Get a chunk
        chunk_data = data_loader.get_random_chunk()
        if not chunk_data:
            logger.error("❌ No chunk available")
            return False
        
        logger.info(f"   📦 Chunk: '{chunk_data.get('text', '')[:50]}...'")
        
        # Test training step
        loss_dict, states = trainer.train_single_chunk(chunk_data)
        
        if loss_dict:
            logger.info("   ✅ Training step successful:")
            logger.info(f"      Total loss: {loss_dict['total_loss'].item():.6f}")
            logger.info(f"      Token loss: {loss_dict['token_loss'].item():.6f}")
            logger.info(f"      Duration loss: {loss_dict['duration_loss'].item():.6f}")
            logger.info(f"      Accuracy: {loss_dict['accuracy']:.4f}")
            logger.info(f"      Duration accuracy: {loss_dict['duration_accuracy']:.4f}")
            
            if states:
                logger.info(f"      States: {len(states)} types")
            
            return True
        else:
            logger.error("❌ Training step failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Training step error: {e}")
        traceback.print_exc()
        return False

def debug_sequential_batch_training():
    """Debug sequential batch training"""
    logger.info("\n🔄 DEBUG: Sequential Batch Training")
    logger.info("=" * 50)
    
    try:
        from sequential_trainer import NoOverlapDataLoader, StatefulTTSModel, NoOverlapTrainer
        from nucleotide_tokenizer import NucleotideTokenizer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup components
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        data_loader = NoOverlapDataLoader(data_dir="no_overlap_data", device=device)
        
        model = StatefulTTSModel(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_codebooks=4,
            codebook_size=1024,
            state_size=64
        ).to(device)
        
        trainer = NoOverlapTrainer(model, tokenizer, data_loader)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        logger.info("🔄 Testing sequential batch training...")
        
        # Get a sequential batch
        batch_info = data_loader.get_sequential_batch()
        if not batch_info:
            logger.error("❌ No batch available")
            return False
        
        logger.info(f"   📦 Batch {batch_info['batch_idx']}: {len(batch_info['chunks'])} chunks")
        
        # Test batch training
        batch_result = trainer.train_sequential_batch(batch_info, optimizer)
        
        if batch_result:
            logger.info("   ✅ Sequential batch training successful:")
            logger.info(f"      Batch loss: {batch_result['batch_loss']:.6f}")
            logger.info(f"      Batch accuracy: {batch_result['batch_accuracy']:.4f}")
            logger.info(f"      Batch duration accuracy: {batch_result['batch_duration_accuracy']:.4f}")
            logger.info(f"      Successful chunks: {batch_result['successful_chunks']}/{batch_result['total_chunks']}")
            
            return True
        else:
            logger.error("❌ Sequential batch training failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Sequential batch training error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive debugging"""
    logger.info("🔍 COMPREHENSIVE STATEFUL MAMBA DEBUG")
    logger.info("=" * 60)
    
    debug_results = {}
    
    # Run all debug functions
    debug_functions = [
        ("Data Loading", debug_data_loading),
        ("Stateful Modules", debug_stateful_modules),
        ("Training Data Loader", debug_training_data_loader),
        ("Model Creation", debug_model_creation),
        ("Loss Computation", debug_loss_computation),
        ("Training Step", debug_training_step),
        ("Sequential Batch", debug_sequential_batch_training)
    ]
    
    for name, debug_func in debug_functions:
        try:
            logger.info(f"\n{'='*20} {name} {'='*20}")
            result = debug_func()
            debug_results[name] = result
            
            if result:
                logger.info(f"✅ {name}: PASSED")
            else:
                logger.error(f"❌ {name}: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {name}: CRASHED - {e}")
            debug_results[name] = False
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🏁 DEBUG SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in debug_results.values() if result)
    total = len(debug_results)
    
    for name, result in debug_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {name:<20}: {status}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All debug tests passed! System should work.")
    elif passed >= total * 0.8:
        logger.info("⚠️  Most tests passed, minor issues detected.")
    else:
        logger.error("❌ Multiple failures detected. System needs attention.")
    
    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    
    if not debug_results.get("Data Loading", False):
        logger.info("   📁 Check if no_overlap_data directory exists and contains clean_batch_* folders")
        logger.info("   🔧 Run audio_processor_sequential.py to generate NO-OVERLAP data")
    
    if not debug_results.get("Stateful Modules", False):  
        logger.info("   🧠 Check stateful_modules.py imports and tensor operations")
        logger.info("   🔧 Verify PyTorch version compatibility")
    
    if not debug_results.get("Training Data Loader", False):
        logger.info("   📦 Check NoOverlapDataLoader class in sequential_trainer.py")
        logger.info("   🔧 Verify chunk metadata and tensor formats")
    
    if not debug_results.get("Model Creation", False):
        logger.info("   🏗️  Check StatefulTTSModel class and component initialization")
        logger.info("   🔧 Verify model parameter sizes and device placement")
    
    if not debug_results.get("Loss Computation", False):
        logger.info("   📊 Check losses.py compute_combined_loss function")
        logger.info("   🔧 Verify tensor shapes and loss calculations")
    
    logger.info(f"\n🔧 Next steps:")
    logger.info(f"   1. Fix any failed components above")
    logger.info(f"   2. Run: python sequential_trainer.py")
    logger.info(f"   3. Monitor training progress and metrics")

if __name__ == "__main__":
    main()