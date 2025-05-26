#!/usr/bin/env python3
"""
PROSTY test training - znajdź dokładny problem
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def simple_training_test():
    """Prosty test jednego kroku trainingu"""
    print("🧪 SIMPLE TRAINING TEST - Find the exact issue")
    print("=" * 60)
    
    try:
        # Import wszystkiego
        from infinite_modules import (
            InfiniteMambaTextEncoder,
            InfiniteMambaAudioProcessor, 
            InfiniteDurationRegulator,
            InfiniteAudioStyleExtractor
        )
        from infinite_data_loader import InfiniteDataLoader
        from nucleotide_tokenizer import NucleotideTokenizer
        from losses import compute_combined_loss
        import torch.nn as nn
        
        print("✅ All imports successful")
        
        # Setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = NucleotideTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        data_loader = InfiniteDataLoader("continuous_data", device)
        
        print(f"✅ Setup complete: {len(data_loader)} batches, device={device}")
        
        # Get batch
        batch_data = data_loader.get_random_continuous_batch()
        if batch_data is None:
            print("❌ No batch data!")
            return
        
        training_batch = data_loader.prepare_batch_for_training(batch_data)
        text_tokens = training_batch['text_tokens']
        audio_codes = training_batch['audio_codes']
        
        print(f"✅ Batch ready: text={text_tokens.shape}, audio={audio_codes.shape}")
        
        # Stwórz uproszczony model
        class SimpleTestModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.text_encoder = InfiniteMambaTextEncoder(vocab_size, 128, 4, 64)
                self.duration_regulator = InfiniteDurationRegulator(128, 64, 128, 75.0, 64)
                self.audio_processor = InfiniteMambaAudioProcessor(256, 4, 1024, 3, 64)
                self.style_extractor = InfiniteAudioStyleExtractor(256, 64)
                self.text_proj = nn.Linear(128, 256)
                self.default_style = nn.Parameter(torch.randn(64) * 0.01)
            
            def reset_infinite_states(self, batch_size=1):
                self.text_encoder.reset_infinite_state(batch_size)
                self.duration_regulator.reset_infinite_state(batch_size)
                self.audio_processor.reset_infinite_state(batch_size)
            
            def forward(self, text_tokens, audio_tokens=None):
                print(f"   🔄 Model forward: text={text_tokens.shape}, audio={audio_tokens.shape if audio_tokens is not None else None}")
                
                B = text_tokens.shape[0]
                
                # Reset states
                self.reset_infinite_states(B)
                print(f"   ✅ States reset")
                
                # Text encoding
                print(f"   🧠 Text encoding...")
                text_features = self.text_encoder(text_tokens, reset_state=True)
                print(f"   ✅ Text features: {text_features.shape}")
                
                # Text context
                if text_features.dim() == 3:
                    text_context = torch.mean(text_features, dim=1)
                else:
                    text_context = text_features
                text_context = self.text_proj(text_context)
                print(f"   ✅ Text context: {text_context.shape}")
                
                # Style
                print(f"   🎨 Style extraction...")
                if audio_tokens is not None:
                    B, C, T = audio_tokens.shape
                    audio_mean = torch.mean(audio_tokens.float(), dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
                    pseudo_audio = audio_mean.expand(B, 256, min(T, 200))
                    style_embedding = self.style_extractor(pseudo_audio)
                else:
                    style_embedding = self.default_style.unsqueeze(0).expand(B, -1)
                print(f"   ✅ Style: {style_embedding.shape}")
                
                # Duration
                print(f"   ⏱️  Duration regulation...")
                regulated_features, predicted_durations, duration_tokens, duration_confidence = \
                    self.duration_regulator(text_features, style_embedding, reset_state=True)
                print(f"   ✅ Duration: regulated={regulated_features.shape}, durations={predicted_durations.shape}")
                
                # Audio processing
                print(f"   🎵 Audio processing...")
                if audio_tokens is not None:
                    if regulated_features.dim() == 3:
                        regulated_context = torch.mean(regulated_features, dim=1)
                    else:
                        regulated_context = regulated_features
                    regulated_context = self.text_proj(regulated_context)
                    
                    audio_logits = self.audio_processor(audio_tokens, regulated_context, reset_state=True)
                    print(f"   ✅ Audio logits: {audio_logits.shape}")
                else:
                    audio_logits = None
                
                return {
                    'logits': audio_logits,
                    'predicted_durations': predicted_durations,
                    'duration_tokens': duration_tokens,
                    'duration_confidence': duration_confidence,
                    'text_features': text_features,
                    'regulated_features': regulated_features,
                    'style_embedding': style_embedding
                }
        
        # Test model
        print("🏗️  Creating model...")
        model = SimpleTestModel(vocab_size).to(device)
        print("✅ Model created")
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        with torch.no_grad():
            output = model(text_tokens, audio_codes)
        
        if output is None:
            print("❌ Forward pass failed!")
            return
        
        print("✅ Forward pass successful!")
        
        # Test loss
        print("💰 Testing loss computation...")
        loss_dict = compute_combined_loss(output, training_batch, text_tokens, device)
        
        if loss_dict is None:
            print("❌ Loss computation failed!")
            return
        
        print(f"✅ Loss computed: {loss_dict['total_loss'].item():.4f}")
        
        # Test training step
        print("🚀 Testing full training step...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(text_tokens, audio_codes)
        if output is None:
            print("❌ Training forward failed!")
            return
        
        # Loss
        loss_dict = compute_combined_loss(output, training_batch, text_tokens, device)
        if loss_dict is None:
            print("❌ Training loss failed!")
            return
        
        total_loss = loss_dict['total_loss']
        print(f"   Loss: {total_loss.item():.4f}")
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print("🎉 FULL TRAINING STEP SUCCESSFUL!")
        print(f"   Final loss: {total_loss.item():.4f}")
        print(f"   Accuracy: {loss_dict['accuracy']:.4f}")
        print(f"   Duration accuracy: {loss_dict['duration_accuracy']:.4f}")
        
        print("\n✅ TRAINING SHOULD WORK!")
        print("   The issue must be in InfiniteTrainer class or virtual checkpoint processing")
        
    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_training_test()