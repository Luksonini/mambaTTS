#!/usr/bin/env python3
"""
Duration Debug Test - Isolated Testing
=====================================
Test duration predictor in isolation and compare models
"""

import torch
import torch.nn.functional as F
import logging
import sys
from pathlib import Path

# Setup
sys.path.append('.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from data_loader import SafeDataLoader
from nucleotide_tokenizer import NucleotideTokenizer

class SimpleDurationTester:
    """Isolated duration predictor testing"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = NucleotideTokenizer()
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # Load data
        self.data_loader = SafeDataLoader(
            data_dir="precomputed",
            vocab_size=self.vocab_size,
            codebook_size=1024,
            device=self.device
        )
        
        logger.info(f"🔧 SimpleDurationTester initialized on {self.device}")
        logger.info(f"   Vocab size: {self.vocab_size}")
        logger.info(f"   Data chunks: {self.data_loader.get_stats()['total_chunks']}")
    
    def create_isolated_duration_model(self, tokens_per_second=15.0, style_dim=64):
        """Create ONLY duration regulator for isolated testing"""
        from modules import DurationRegulator, MambaConvTextEncoder
        
        class IsolatedDurationModel(torch.nn.Module):
            def __init__(self, vocab_size, tokens_per_second, style_dim):
                super().__init__()
                self.embed_dim = 128
                
                # Text encoder (needed for features)
                self.text_encoder = MambaConvTextEncoder(vocab_size, self.embed_dim)
                
                # ONLY duration regulator
                self.duration_regulator = DurationRegulator(
                    text_dim=self.embed_dim,
                    style_dim=style_dim,
                    hidden_dim=64,
                    tokens_per_second=tokens_per_second
                )
                
                # Simple default style
                self.default_style = torch.nn.Parameter(torch.randn(style_dim) * 0.01)
                
                logger.info(f"🎯 IsolatedDurationModel:")
                logger.info(f"   tokens_per_second: {tokens_per_second}")
                logger.info(f"   style_dim: {style_dim}")
                logger.info(f"   parameters: {sum(p.numel() for p in self.parameters()):,}")
            
            def forward(self, text_tokens):
                batch_size = text_tokens.shape[0]
                
                # Text encoding
                text_features = self.text_encoder(text_tokens, return_sequence=True)
                
                # Style (simple default)
                style_embedding = self.default_style.unsqueeze(0).expand(batch_size, -1)
                
                # Duration prediction
                regulated_features, predicted_durations, duration_tokens, duration_confidence = self.duration_regulator(
                    text_features, style_embedding
                )
                
                return {
                    'predicted_durations': predicted_durations,
                    'duration_confidence': duration_confidence,
                    'text_features': text_features,
                    'regulated_features': regulated_features
                }
        
        return IsolatedDurationModel(self.vocab_size, tokens_per_second, style_dim)
    
    def create_target_durations(self, chunk_data, text_tokens):
        """Create realistic target durations"""
        device = text_tokens.device
        
        # Get basic info
        total_duration = chunk_data['duration']
        text_length = text_tokens.shape[1] if text_tokens.dim() > 1 else text_tokens.shape[0]
        
        # Simple uniform distribution
        avg_duration_per_token = total_duration / text_length
        
        # Clamp to realistic range
        avg_duration_per_token = max(0.05, min(0.3, avg_duration_per_token))
        
        # Create targets with slight variation
        targets = torch.full((1, text_length), avg_duration_per_token, device=device)
        
        # Add some realistic variation based on token type
        text_list = text_tokens[0].cpu().tolist() if text_tokens.dim() > 1 else text_tokens.cpu().tolist()
        
        for i, token_id in enumerate(text_list):
            token_char = self.tokenizer.id2token.get(token_id, '')
            
            if token_char in ['<s>', '</s>', '<pad>', '<unk>']:
                targets[0, i] = 0.03  # Short for special tokens
            elif token_char == ' ':
                targets[0, i] = 0.08  # Slightly longer for spaces
            # else: keep default avg_duration_per_token
        
        return targets
    
    def test_duration_learning(self, model, num_steps=1000, learning_rate=1e-3):
        """Test if duration predictor can learn on isolated task"""
        logger.info(f"🎯 Testing duration learning for {num_steps} steps...")
        
        model.to(self.device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        losses = []
        
        for step in range(num_steps):
            # Get random chunk
            chunk_data = self.data_loader.get_random_chunk()
            
            # Prepare text tokens
            text_tokens = chunk_data['text_tokens']
            if text_tokens.dim() == 1:
                text_tokens = text_tokens.unsqueeze(0)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(text_tokens)
            
            # Get targets
            target_durations = self.create_target_durations(chunk_data, text_tokens)
            predicted_durations = output['predicted_durations']
            
            # Compute loss
            min_len = min(predicted_durations.shape[1], target_durations.shape[1])
            pred_trunc = predicted_durations[:, :min_len]
            target_trunc = target_durations[:, :min_len]
            
            # Strong duration loss
            duration_loss = F.l1_loss(pred_trunc, target_trunc) * 10.0
            
            # Regularization for realistic range
            duration_reg = (
                torch.mean(torch.relu(pred_trunc - 0.3)) +  # Penalty for > 0.3s
                torch.mean(torch.relu(0.05 - pred_trunc))   # Penalty for < 0.05s
            )
            
            total_loss = duration_loss + duration_reg
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(total_loss.item())
            
            # Logging
            if step % 200 == 0:
                pred_total = pred_trunc.sum().item()
                target_total = target_trunc.sum().item()
                real_duration = chunk_data['duration']
                
                logger.info(f"Step {step:4d}: Loss={total_loss.item():.4f}")
                logger.info(f"         Pred={pred_total:.2f}s, Target={target_total:.2f}s, Real={real_duration:.2f}s")
                logger.info(f"         Ratio: {pred_total/real_duration:.2f}x")
        
        return losses
    
    def evaluate_model(self, model, num_samples=5):
        """Evaluate model performance"""
        logger.info(f"📊 Evaluating model on {num_samples} samples...")
        
        model.eval()
        results = []
        
        with torch.no_grad():
            for i in range(num_samples):
                chunk_data = self.data_loader.get_chunk(i * 10)
                
                # Prepare input
                text_tokens = chunk_data['text_tokens']
                if text_tokens.dim() == 1:
                    text_tokens = text_tokens.unsqueeze(0)
                
                # Forward pass
                output = model(text_tokens)
                
                predicted_durations = output['predicted_durations'][0]
                duration_confidence = output['duration_confidence'][0]
                
                # Analysis
                pred_total = predicted_durations.sum().item()
                real_duration = chunk_data['duration']
                avg_per_token = predicted_durations.mean().item()
                avg_confidence = duration_confidence.mean().item()
                ratio = pred_total / real_duration
                
                result = {
                    'text': chunk_data['text'][:60] + "...",
                    'real_duration': real_duration,
                    'predicted_total': pred_total,
                    'avg_per_token': avg_per_token,
                    'avg_confidence': avg_confidence,
                    'ratio': ratio,
                    'text_length': len(text_tokens[0])
                }
                
                results.append(result)
                
                logger.info(f"Sample {i+1}:")
                logger.info(f"   Text: '{result['text']}'")
                logger.info(f"   Real: {real_duration:.2f}s, Pred: {pred_total:.2f}s")
                logger.info(f"   Ratio: {ratio:.2f}x, Avg/token: {avg_per_token:.3f}s")
                logger.info(f"   Confidence: {avg_confidence:.3f}")
        
        return results
    
    def compare_configurations(self):
        """Compare different configuration settings"""
        logger.info("🔬 Comparing different configurations...")
        
        configs = [
            {'tokens_per_second': 75.0, 'style_dim': 128, 'name': 'Original (Broken)'},
            {'tokens_per_second': 15.0, 'style_dim': 128, 'name': 'Fixed TPS only'},
            {'tokens_per_second': 15.0, 'style_dim': 64, 'name': 'Fixed TPS + Small Style'},
        ]
        
        results = {}
        
        for config in configs:
            logger.info(f"\n🧪 Testing: {config['name']}")
            logger.info(f"   tokens_per_second: {config['tokens_per_second']}")
            logger.info(f"   style_dim: {config['style_dim']}")
            
            # Create model
            model = self.create_isolated_duration_model(
                tokens_per_second=config['tokens_per_second'],
                style_dim=config['style_dim']
            )
            
            # Train briefly
            losses = self.test_duration_learning(model, num_steps=500, learning_rate=2e-3)
            
            # Evaluate
            eval_results = self.evaluate_model(model, num_samples=3)
            
            # Compute metrics
            avg_ratio = sum(r['ratio'] for r in eval_results) / len(eval_results)
            avg_per_token = sum(r['avg_per_token'] for r in eval_results) / len(eval_results)
            final_loss = losses[-10:] if len(losses) >= 10 else losses
            avg_final_loss = sum(final_loss) / len(final_loss)
            
            results[config['name']] = {
                'avg_ratio': avg_ratio,
                'avg_per_token': avg_per_token,
                'final_loss': avg_final_loss,
                'config': config
            }
            
            logger.info(f"✅ {config['name']} Results:")
            logger.info(f"   Avg ratio: {avg_ratio:.2f}x")
            logger.info(f"   Avg per token: {avg_per_token:.3f}s")
            logger.info(f"   Final loss: {avg_final_loss:.4f}")
        
        # Summary
        logger.info("\n📋 COMPARISON SUMMARY:")
        logger.info("=" * 60)
        
        for name, result in results.items():
            status = "✅ GOOD" if 0.7 <= result['avg_ratio'] <= 1.5 else "❌ BAD"
            logger.info(f"{name:25s}: Ratio={result['avg_ratio']:.2f}x, Token={result['avg_per_token']:.3f}s {status}")
        
        return results

def main():
    """Main testing function"""
    logger.info("🧪 Duration Predictor Debug Test")
    logger.info("=" * 50)
    
    try:
        tester = SimpleDurationTester()
        
        # Test 1: Compare configurations
        logger.info("\n🔬 CONFIGURATION COMPARISON TEST")
        results = tester.compare_configurations()
        
        # Test 2: Find best configuration
        best_config = min(results.items(), key=lambda x: abs(x[1]['avg_ratio'] - 1.0))
        logger.info(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
        logger.info(f"   Ratio: {best_config[1]['avg_ratio']:.2f}x (closest to 1.0)")
        logger.info(f"   Per token: {best_config[1]['avg_per_token']:.3f}s")
        
        # Test 3: Extended training with best config
        logger.info(f"\n🚀 EXTENDED TRAINING WITH BEST CONFIG")
        best_model = tester.create_isolated_duration_model(
            tokens_per_second=best_config[1]['config']['tokens_per_second'],
            style_dim=best_config[1]['config']['style_dim']
        )
        
        # Train longer
        losses = tester.test_duration_learning(best_model, num_steps=2000, learning_rate=1e-3)
        
        # Final evaluation
        logger.info("\n📊 FINAL EVALUATION:")
        final_results = tester.evaluate_model(best_model, num_samples=5)
        
        final_avg_ratio = sum(r['ratio'] for r in final_results) / len(final_results)
        final_avg_per_token = sum(r['avg_per_token'] for r in final_results) / len(final_results)
        
        logger.info(f"\n🎯 FINAL RESULTS:")
        logger.info(f"   Best config: {best_config[0]}")
        logger.info(f"   Final avg ratio: {final_avg_ratio:.2f}x")
        logger.info(f"   Final avg per token: {final_avg_per_token:.3f}s")
        logger.info(f"   Success: {'✅ YES' if 0.8 <= final_avg_ratio <= 1.2 else '❌ NO'}")
        
        # Save best model
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': best_config[1]['config'],
            'final_ratio': final_avg_ratio,
            'final_per_token': final_avg_per_token,
            'vocab_size': tester.vocab_size
        }, 'best_duration_model.pt')
        
        logger.info("💾 Best duration model saved as 'best_duration_model.pt'")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()