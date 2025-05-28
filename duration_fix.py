#!/usr/bin/env python3
"""
Duration Predictor Fix and Monitor
=================================
Naprawia duration predictor i dodaje monitoring podczas treningu
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FixedDurationRegulator(nn.Module):
    """NAPRAWIONY Duration Regulator z monitoringiem"""
    def __init__(self, text_dim=384, style_dim=128, hidden_dim=256, tokens_per_second=75.0):
        super().__init__()
        self.tokens_per_second = tokens_per_second
        
        # LEPSZY duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(text_dim + style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Zapewnia dodatnie wartości
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(text_dim + style_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # MONITORING
        self.duration_history = []
        self.step_count = 0
        
    def forward(self, text_features, style_embedding, chunk_duration=None, monitor=True):
        # PIERWSZA LINIA - sprawdź czy dochodzi!
        print(f"🔥 FORWARD CALLED: chunk_duration={chunk_duration}")
        logger.info(f"🔥 FORWARD CALLED: chunk_duration={chunk_duration}")
        
        # text_features: [B, L, text_dim]
        # style_embedding: [B, style_dim]
        B, L, D = text_features.shape
        
        # Expand style to match sequence length
        style_expanded = style_embedding.unsqueeze(1).expand(B, L, -1)
        
        # Concatenate text and style features
        combined = torch.cat([text_features, style_expanded], dim=-1)
        
        # PREDICT durations - BEZ CLAMP!
        raw_durations = self.duration_predictor(combined).squeeze(-1)  # [B, L]
        
        # LEPSZE ograniczenia dla polskiej mowy
        # Zamiast twardego clamp, użyj sigmoid scaling
        predicted_durations = 0.03 + 0.25 * torch.sigmoid(raw_durations)  # 0.03s - 0.28s
        
        # Predict confidence
        duration_confidence = self.confidence_predictor(combined).squeeze(-1)
        
        # Duration tokens
        duration_tokens = (predicted_durations * self.tokens_per_second).round().long()
        duration_tokens = torch.clamp(duration_tokens, min=2, max=21)  # 2-21 tokenów
        
        # MONITORING
        if monitor and self.training:
            self._monitor_durations(raw_durations, predicted_durations, chunk_duration, L)
        
        return text_features, predicted_durations, duration_tokens, duration_confidence
    
    def _monitor_durations(self, raw_durations, predicted_durations, chunk_duration, text_length):
        """Monitor duration predictions podczas treningu"""
        try:
            self.step_count += 1
            
            # Podstawowe statystyki
            raw_mean = raw_durations.mean().item()
            raw_std = raw_durations.std().item()
            pred_mean = predicted_durations.mean().item()
            pred_std = predicted_durations.std().item()
            
            # Przewidywana całkowita długość
            total_pred_duration = predicted_durations.sum().item()
            
            # Statystyki
            stats = {
                'step': self.step_count,
                'raw_mean': raw_mean,
                'raw_std': raw_std,
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'total_predicted': total_pred_duration,
                'target_duration': chunk_duration if chunk_duration else 0.0,
                'text_length': text_length,
                'error': abs(total_pred_duration - (chunk_duration or 0.0))
            }
            
            self.duration_history.append(stats)
            
            # Log co 50 kroków
            if self.step_count % 50 == 0:
                logger.info(f"📊 Duration Monitor Step {self.step_count}:")
                logger.info(f"   Raw: μ={raw_mean:.4f}, σ={raw_std:.4f}")
                logger.info(f"   Pred: μ={pred_mean:.4f}, σ={pred_std:.4f} (range: {predicted_durations.min():.3f}-{predicted_durations.max():.3f})")
                logger.info(f"   Total: {total_pred_duration:.2f}s vs target {chunk_duration or 0:.2f}s (error: {stats['error']:.2f}s)")
                
                # Pokaż scaling info jeśli jest chunk_duration
                if chunk_duration and chunk_duration > 0:
                    scale_factor = chunk_duration / predicted_durations.sum()
                    logger.info(f"   Scaling: {scale_factor:.3f}x {'✅' if 0.7 < scale_factor < 1.3 else '⚠️'}")
                
                # OSTRZEŻENIA
                if pred_std < 0.01:
                    logger.warning("⚠️  Duration variance too low - model may be stuck!")
                if stats['error'] > 2.0:
                    logger.warning(f"⚠️  High duration error: {stats['error']:.2f}s")
                if pred_mean < 0.05 or pred_mean > 0.25:
                    logger.warning(f"⚠️  Duration mean out of range: {pred_mean:.3f}s")
            
            # Zapisz monitoring co 200 kroków
            if self.step_count % 200 == 0:
                self._save_monitoring_plots()
                
        except Exception as e:
            logger.debug(f"Monitoring error: {e}")
    
    def _save_monitoring_plots(self):
        """Zapisz wykresy monitoringu"""
        try:
            if len(self.duration_history) < 10:
                return
            
            # Przygotuj dane
            steps = [s['step'] for s in self.duration_history]
            pred_means = [s['pred_mean'] for s in self.duration_history]
            pred_stds = [s['pred_std'] for s in self.duration_history]
            errors = [s['error'] for s in self.duration_history]
            
            # Utwórz wykres
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Duration Predictor Monitoring', fontsize=14)
            
            # 1. Średnia długość przewidywana
            axes[0, 0].plot(steps, pred_means)
            axes[0, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Target ~0.1s')
            axes[0, 0].set_title('Predicted Duration Mean')
            axes[0, 0].set_ylabel('Duration (s)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Odchylenie standardowe
            axes[0, 1].plot(steps, pred_stds)
            axes[0, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Good variance')
            axes[0, 1].set_title('Duration Prediction Variance')
            axes[0, 1].set_ylabel('Std Dev (s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Błąd całkowitej długości
            axes[1, 0].plot(steps, errors)
            axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1s error')
            axes[1, 0].set_title('Total Duration Error')
            axes[1, 0].set_ylabel('Error (s)')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Histogram ostatnich przewidywań
            recent_means = pred_means[-50:] if len(pred_means) > 50 else pred_means
            axes[1, 1].hist(recent_means, bins=20, alpha=0.7)
            axes[1, 1].axvline(x=0.1, color='r', linestyle='--', label='Target')
            axes[1, 1].set_title('Recent Duration Distribution')
            axes[1, 1].set_xlabel('Duration (s)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(f'duration_monitoring_step_{self.step_count}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Duration monitoring plot saved: duration_monitoring_step_{self.step_count}.png")
            
        except Exception as e:
            logger.debug(f"Plot saving error: {e}")
    
    def get_monitoring_summary(self):
        """Zwróć podsumowanie monitoringu"""
        if not self.duration_history:
            return "No monitoring data available"
        
        recent_history = self.duration_history[-100:] if len(self.duration_history) > 100 else self.duration_history
        
        pred_means = [s['pred_mean'] for s in recent_history]
        pred_stds = [s['pred_std'] for s in recent_history]
        errors = [s['error'] for s in recent_history]
        
        summary = {
            'total_steps': len(self.duration_history),
            'recent_mean_duration': np.mean(pred_means),
            'recent_std_duration': np.mean(pred_stds),
            'recent_avg_error': np.mean(errors),
            'duration_variance': np.var(pred_means),
            'is_stuck': np.mean(pred_stds) < 0.01,
            'high_error': np.mean(errors) > 2.0
        }
        
        return summary


def test_duration_predictor():
    """Test naprawionego duration predictor"""
    logger.info("🧪 Testing Fixed Duration Predictor")
    logger.info("=" * 40)
    
    # Utwórz model
    regulator = FixedDurationRegulator()
    regulator.train()
    
    # Test data
    batch_size, seq_len, text_dim = 1, 50, 384
    style_dim = 128
    
    text_features = torch.randn(batch_size, seq_len, text_dim)
    style_embedding = torch.randn(batch_size, style_dim)
    chunk_duration = 5.0  # 5 sekund
    
    # Forward pass
    logger.info("🔄 Testing forward pass...")
    
    for step in range(10):
        # Symulacja różnych wejść
        text_features = torch.randn(batch_size, seq_len, text_dim)
        style_embedding = torch.randn(batch_size, style_dim) * 0.1  # Realistic style variation
        
        # WAŻNE: Przekaż chunk_duration do forward! + DEBUG
        logger.info(f"TEST DEBUG: Calling forward with chunk_duration={chunk_duration}")
        _, predicted_durations, duration_tokens, confidence = regulator(
            text_features, style_embedding, chunk_duration=chunk_duration, monitor=True
        )
        
        logger.info(f"Step {step+1}:")
        logger.info(f"   Duration range: {predicted_durations.min():.3f}s - {predicted_durations.max():.3f}s")
        logger.info(f"   Total predicted: {predicted_durations.sum():.2f}s (target: {chunk_duration}s)")
        logger.info(f"   Token range: {duration_tokens.min()}-{duration_tokens.max()} tokens")
        
        # DEBUG: Sprawdź czy scaling działa
        scale_factor = chunk_duration / predicted_durations.sum() if predicted_durations.sum() > 0 else 1.0
        logger.info(f"   DEBUG: Would scale by {scale_factor:.3f}x")
        
        # SPRAWDŹ czy wartości faktycznie się zmieniły po scaling
        if abs(predicted_durations.sum().item() - chunk_duration) < 0.1:
            logger.info(f"   ✅ SCALING WORKED!")
        else:
            logger.info(f"   ❌ SCALING FAILED!")
    
    # Podsumowanie
    summary = regulator.get_monitoring_summary()
    logger.info(f"\n📊 Monitoring Summary:")
    for key, value in summary.items():
        logger.info(f"   {key}: {value}")
    
    logger.info("✅ Fixed Duration Predictor test completed!")


if __name__ == "__main__":
    test_duration_predictor()