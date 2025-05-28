#!/usr/bin/env python3
"""
Nowe Loss Functions od ZERA - Rozwiązanie problemu duration prediction
=====================================================================
Skupia się na diversity i realistycznych predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_realistic_duration_targets(chunk_data, text_tokens, device):
    """
    Tworzy REALISTYCZNE duration targets z prawdziwą różnorodnością
    """
    try:
        # Basic info
        total_duration = chunk_data.get('duration', 4.0)
        
        if torch.is_tensor(text_tokens):
            if text_tokens.dim() > 1:
                text_length = text_tokens.shape[1]
                token_list = text_tokens[0].cpu().tolist()
            else:
                text_length = text_tokens.shape[0]
                token_list = text_tokens.cpu().tolist()
        else:
            text_length = len(text_tokens)
            token_list = text_tokens
        
        if text_length <= 0:
            return torch.full((1, 10), 0.12, device=device)
        
        # Podstawowa średnia duration na token
        avg_duration = total_duration / text_length
        avg_duration = max(0.04, min(0.35, avg_duration))
        
        # Inicjalizuj z baseline
        durations = torch.full((1, text_length), avg_duration, device=device)
        
        # REALISTYCZNE RÓŻNICE based na token characteristics
        for i, token_id in enumerate(token_list[:text_length]):
            base_dur = avg_duration
            
            # 1. SPECIAL TOKENS - bardzo krótkie
            if token_id <= 3:  # <pad>, <unk>, <sos>, <eos>
                durations[0, i] = 0.01
            
            # 2. PUNCTUATION/SPACE (4-20) - krótkie pauzy
            elif 4 <= token_id <= 20:
                durations[0, i] = 0.03 + 0.02 * (token_id % 5) / 5  # 0.03-0.05s
            
            # 3. CONSONANTS (21-60) - krótkie do średnich
            elif 21 <= token_id <= 60:
                consonant_factor = 0.5 + 0.4 * (token_id % 15) / 15  # 0.5-0.9
                durations[0, i] = base_dur * consonant_factor
            
            # 4. VOWELS (61-100) - średnie do długie
            elif 61 <= token_id <= 100:
                vowel_factor = 0.9 + 0.6 * (token_id % 12) / 12  # 0.9-1.5
                durations[0, i] = base_dur * vowel_factor
            
            # 5. COMPLEX SOUNDS (101-200) - długie
            elif 101 <= token_id <= 200:
                complex_factor = 1.2 + 0.8 * (token_id % 8) / 8  # 1.2-2.0
                durations[0, i] = base_dur * complex_factor
            
            # 6. RARE TOKENS (201+) - bardzo zróżnicowane
            else:
                rare_factor = 0.6 + 1.0 * (token_id % 20) / 20  # 0.6-1.6
                durations[0, i] = base_dur * rare_factor
        
        # POZYCYJNE MODIFIKACJE
        for i in range(text_length):
            # Początki i końce - dłuższe (sentence boundaries)
            if i == 0 or i == text_length - 1:
                durations[0, i] *= 1.3
            
            # Co 8-12 token - word boundaries (krótsze)
            elif i % 10 == 0:
                durations[0, i] *= 0.7
            
            # Środki długich fraz - stabilne
            elif text_length > 50 and 20 <= i <= text_length - 20:
                durations[0, i] *= 1.0  # Bez zmian
        
        # WARIACJA BASED NA DŁUGOŚCI TEKSTU
        if text_length < 30:  # Krótki tekst - szybsza mowa
            durations *= 0.8
        elif text_length > 100:  # Długi tekst - wolniejsza mowa
            durations *= 1.2
        
        # FINAL CLAMP do sensownych wartości
        durations = torch.clamp(durations, min=0.01, max=0.4)
        
        # QUALITY CHECK - czy suma się zgadza
        predicted_total = durations.sum().item()
        if abs(predicted_total - total_duration) > total_duration * 0.4:  # >40% error
            # Przeskaluj aby lepiej pasować
            scale_factor = total_duration / predicted_total
            durations = durations * scale_factor
            durations = torch.clamp(durations, min=0.01, max=0.4)
        
        return durations
        
    except Exception as e:
        logger.warning(f"Duration target creation failed: {e}")
        # Safe fallback z małą wariancją
        fallback = torch.full((1, max(10, text_length)), 0.08, device=device)
        # Dodaj małą wariację
        noise = torch.randn_like(fallback) * 0.02
        fallback = fallback + noise
        return torch.clamp(fallback, min=0.02, max=0.25)


def diversity_loss(predicted_durations, target_weight=0.08):
    """
    Loss promujący różnorodność w duration predictions
    Główny problem: model przewiduje zawsze podobne wartości
    """
    if predicted_durations is None:
        return torch.tensor(0.0, requires_grad=True)
    
    if predicted_durations.dim() == 1:
        predicted_durations = predicted_durations.unsqueeze(0)
    
    B, L = predicted_durations.shape
    if L <= 1:
        return torch.tensor(0.0, requires_grad=True)
    
    # 1. STANDARD DEVIATION PENALTY
    # Karzemy za zbyt małe odchylenie standardowe
    pred_std = torch.std(predicted_durations, dim=1, unbiased=False)  # [B]
    target_std = target_weight * 0.5  # Target: 50% of average duration as std
    
    std_penalty = F.relu(target_std - pred_std.mean()) * 8.0
    
    # 2. RANGE PENALTY  
    # Karzemy za zbyt mały range min-max
    pred_min = torch.min(predicted_durations, dim=1)[0]  # [B]
    pred_max = torch.max(predicted_durations, dim=1)[0]  # [B]
    pred_range = pred_max - pred_min
    
    target_range = target_weight * 3.0  # Target: 3x average duration range
    range_penalty = F.relu(target_range - pred_range.mean()) * 5.0
    
    # 3. VARIANCE PENALTY
    # Bezpośrednio promujemy wariancję
    pred_var = torch.var(predicted_durations, dim=1, unbiased=False)
    target_var = (target_weight * 0.4) ** 2  # Target variance
    
    var_penalty = F.relu(target_var - pred_var.mean()) * 12.0
    
    # 4. MONOTONICITY PENALTY
    # Lekko karzemy za zbyt monotoniczne sekwencje
    if L > 2:
        # Oblicz absolute differences między sąsiednimi elementami
        diffs = torch.abs(predicted_durations[:, 1:] - predicted_durations[:, :-1])
        avg_change = diffs.mean()
        
        # Karzemy jeśli średnia zmiana jest za mała
        mono_penalty = F.relu(0.02 - avg_change) * 3.0
    else:
        mono_penalty = torch.tensor(0.0, device=predicted_durations.device)
    
    total_diversity_loss = std_penalty + range_penalty + var_penalty + mono_penalty
    
    return total_diversity_loss


def duration_prediction_loss(predicted_durations, target_durations):
    """
    GŁÓWNY duration loss z naciskiem na accuracy I diversity
    """
    if predicted_durations is None or target_durations is None:
        return torch.tensor(1.0, requires_grad=True)
    
    # Handle shapes
    if predicted_durations.dim() == 1:
        predicted_durations = predicted_durations.unsqueeze(0)
    if target_durations.dim() == 1:
        target_durations = target_durations.unsqueeze(0)
    
    # Match lengths
    min_len = min(predicted_durations.shape[1], target_durations.shape[1])
    if min_len <= 0:
        return torch.tensor(1.0, requires_grad=True)
    
    pred_trunc = predicted_durations[:, :min_len]
    target_trunc = target_durations[:, :min_len]
    
    # Clamp targets do rozsądnych wartości
    target_trunc = torch.clamp(target_trunc, min=0.01, max=0.5)
    
    # 1. PODSTAWOWY ACCURACY LOSS
    # L1 loss jest lepszy dla duration niż MSE
    l1_loss = F.l1_loss(pred_trunc, target_trunc)
    
    # 2. RELATIVE ERROR LOSS
    # Promuje względną accuracy (ważniejsze dla krótkich dźwięków)
    relative_error = torch.abs(pred_trunc - target_trunc) / (target_trunc + 1e-6)
    relative_loss = torch.mean(relative_error)
    
    # 3. SMOOTHNESS REGULARIZATION
    # Nie pozwalamy na zbyt gwałtowne skoki
    if min_len > 1:
        pred_diffs = torch.abs(pred_trunc[:, 1:] - pred_trunc[:, :-1])
        target_diffs = torch.abs(target_trunc[:, 1:] - target_trunc[:, :-1])
        
        # Karzemy za skoki większe niż w target
        smooth_loss = F.relu(pred_diffs - target_diffs * 2.0).mean()
    else:
        smooth_loss = torch.tensor(0.0, device=pred_trunc.device)
    
    # 4. RANGE CONSISTENCY
    # Pred i target powinny mieć podobne ranges
    pred_range = pred_trunc.max() - pred_trunc.min()
    target_range = target_trunc.max() - target_trunc.min()
    range_consistency = torch.abs(pred_range - target_range)
    
    # COMBINE wszystkie składniki
    total_duration_loss = (
        5.0 * l1_loss +              # Główna accuracy
        3.0 * relative_loss +        # Względna accuracy
        1.0 * smooth_loss +          # Smoothness
        2.0 * range_consistency      # Range consistency
    )
    
    return total_duration_loss


def confidence_loss(confidence_scores, predicted_durations, target_durations):
    """
    Confidence loss - model powinien być pewny dobrych predictions
    """
    if any(x is None for x in [confidence_scores, predicted_durations, target_durations]):
        return torch.tensor(0.1, requires_grad=True)
    
    # Handle shapes
    if confidence_scores.dim() == 1:
        confidence_scores = confidence_scores.unsqueeze(0)
    if predicted_durations.dim() == 1:
        predicted_durations = predicted_durations.unsqueeze(0)
    if target_durations.dim() == 1:
        target_durations = target_durations.unsqueeze(0)
    
    # Match lengths
    min_len = min(confidence_scores.shape[1], 
                  predicted_durations.shape[1], 
                  target_durations.shape[1])
    if min_len <= 0:
        return torch.tensor(0.1, requires_grad=True)
    
    conf_trunc = confidence_scores[:, :min_len]
    pred_trunc = predicted_durations[:, :min_len]
    target_trunc = target_durations[:, :min_len]
    
    # Oblicz accuracy dla każdego tokena
    absolute_error = torch.abs(pred_trunc - target_trunc)
    relative_error = absolute_error / (target_trunc + 1e-6)
    
    # Target confidence: wysoka gdy error jest niski
    # Użyj kombinacji absolute i relative error
    combined_error = (absolute_error / 0.1) + relative_error  # Normalizacja
    target_confidence = torch.exp(-combined_error)  # Im mniejszy error, tym wyższa confidence
    
    # MSE loss między predicted i target confidence
    conf_loss = F.mse_loss(conf_trunc, target_confidence)
    
    return conf_loss


def audio_token_loss(logits, audio_codes):
    """
    Audio token prediction loss dla multi-codebook system
    """
    if logits is None or audio_codes is None:
        return torch.tensor(2.0, requires_grad=True)
    
    B, C, T_logits, V = logits.shape
    _, C_audio, T_audio = audio_codes.shape
    
    if T_audio <= 1:
        return torch.tensor(2.0, requires_grad=True)
    
    total_loss = 0.0
    valid_codebooks = 0
    
    # Process każdy codebook osobno
    for cb_idx in range(min(C, C_audio)):
        try:
            # Teacher forcing: predict next token
            target_tokens = audio_codes[:, cb_idx, 1:]  # [B, T-1] - next tokens
            
            # Handle length mismatch
            min_T = min(T_logits, target_tokens.shape[1])
            if min_T <= 0:
                continue
            
            pred_logits = logits[:, cb_idx, :min_T, :]  # [B, min_T, V]
            target_tokens_trunc = target_tokens[:, :min_T]  # [B, min_T]
            
            # Clamp targets do valid range
            target_tokens_trunc = torch.clamp(target_tokens_trunc, 0, V - 1)
            
            # Cross entropy z label smoothing
            cb_loss = F.cross_entropy(
                pred_logits.reshape(-1, V),
                target_tokens_trunc.reshape(-1),
                label_smoothing=0.05,  # Lekkie label smoothing
                reduction='mean'
            )
            
            if torch.isfinite(cb_loss):
                total_loss += cb_loss
                valid_codebooks += 1
                
        except Exception as e:
            logger.debug(f"Codebook {cb_idx} processing failed: {e}")
            continue
    
    if valid_codebooks == 0:
        return torch.tensor(2.0, requires_grad=True)
    
    avg_loss = total_loss / valid_codebooks
    
    # Sanity check
    if not torch.isfinite(avg_loss):
        return torch.tensor(2.0, requires_grad=True)
    
    return avg_loss


def compute_accuracy_metrics(predicted_durations, target_durations, logits, audio_codes):
    """
    Oblicza metryki accuracy dla duration i audio tokens
    """
    metrics = {}
    
    # DURATION ACCURACY
    try:
        if predicted_durations is not None and target_durations is not None:
            # Handle shapes
            if predicted_durations.dim() == 1:
                predicted_durations = predicted_durations.unsqueeze(0)
            if target_durations.dim() == 1:
                target_durations = target_durations.unsqueeze(0)
            
            # Match lengths
            min_len = min(predicted_durations.shape[1], target_durations.shape[1])
            if min_len > 0:
                pred_trunc = predicted_durations[:, :min_len]
                target_trunc = target_durations[:, :min_len]
                
                # Multiple tolerance levels
                abs_error = torch.abs(pred_trunc - target_trunc)
                
                # Strict tolerance (±30ms)
                strict_acc = (abs_error < 0.03).float().mean().item()
                
                # Medium tolerance (±50ms OR 25% relative error)
                medium_abs = abs_error < 0.05
                medium_rel = (abs_error / (target_trunc + 1e-6)) < 0.25
                medium_acc = (medium_abs | medium_rel).float().mean().item()
                
                # Loose tolerance (±80ms OR 40% relative error)
                loose_abs = abs_error < 0.08
                loose_rel = (abs_error / (target_trunc + 1e-6)) < 0.40
                loose_acc = (loose_abs | loose_rel).float().mean().item()
                
                metrics.update({
                    'duration_accuracy_strict': strict_acc,
                    'duration_accuracy_medium': medium_acc,
                    'duration_accuracy_loose': loose_acc,
                    'duration_mae': abs_error.mean().item(),
                    'duration_std': pred_trunc.std().item()
                })
            else:
                metrics.update({
                    'duration_accuracy_strict': 0.0,
                    'duration_accuracy_medium': 0.0,
                    'duration_accuracy_loose': 0.0,
                    'duration_mae': 1.0,
                    'duration_std': 0.0
                })
        else:
            metrics.update({
                'duration_accuracy_strict': 0.0,
                'duration_accuracy_medium': 0.0,
                'duration_accuracy_loose': 0.0,
                'duration_mae': 1.0,
                'duration_std': 0.0
            })
    except Exception as e:
        logger.debug(f"Duration accuracy computation failed: {e}")
        metrics.update({
            'duration_accuracy_strict': 0.0,
            'duration_accuracy_medium': 0.0,
            'duration_accuracy_loose': 0.0,
            'duration_mae': 1.0,
            'duration_std': 0.0
        })
    
    # AUDIO TOKEN ACCURACY
    try:
        if logits is not None and audio_codes is not None:
            B, C, T_logits, V = logits.shape
            _, C_audio, T_audio = audio_codes.shape
            
            if T_audio > 1:
                predicted_tokens = torch.argmax(logits, dim=-1)  # [B, C, T_logits]
                target_tokens = audio_codes[:, :, 1:]  # [B, C, T_audio-1]
                
                # Match dimensions
                min_C = min(C, C_audio)
                min_T = min(T_logits, target_tokens.shape[-1])
                
                if min_T > 0:
                    pred_trunc = predicted_tokens[:, :min_C, :min_T]
                    target_trunc = target_tokens[:, :min_C, :min_T]
                    
                    # Exact matches
                    exact_matches = (pred_trunc == target_trunc).float()
                    audio_accuracy = exact_matches.mean().item()
                    
                    # Per-codebook accuracy
                    cb_accuracies = []
                    for cb in range(min_C):
                        cb_acc = exact_matches[:, cb, :].mean().item()
                        cb_accuracies.append(cb_acc)
                    
                    metrics.update({
                        'audio_accuracy': audio_accuracy,
                        'audio_accuracy_per_codebook': cb_accuracies
                    })
                else:
                    metrics.update({
                        'audio_accuracy': 0.0,
                        'audio_accuracy_per_codebook': [0.0] * min(C, 8)
                    })
            else:
                metrics.update({
                    'audio_accuracy': 0.0,
                    'audio_accuracy_per_codebook': [0.0] * min(C, 8)
                })
        else:
            metrics.update({
                'audio_accuracy': 0.0,
                'audio_accuracy_per_codebook': [0.0] * 8
            })
    except Exception as e:
        logger.debug(f"Audio accuracy computation failed: {e}")
        metrics.update({
            'audio_accuracy': 0.0,
            'audio_accuracy_per_codebook': [0.0] * 8
        })
    
    return metrics


def compute_combined_loss(model_output, chunk_data, text_tokens, device):
    """
    GŁÓWNA FUNKCJA LOSS - kombinuje wszystkie składniki
    """
    # Extract outputs z modelu
    logits = model_output.get('logits')
    predicted_durations = model_output.get('predicted_durations')
    duration_confidence = model_output.get('duration_confidence')
    
    # Extract audio codes z chunk data
    audio_codes = chunk_data.get('audio_codes')
    if audio_codes is not None and audio_codes.dim() == 2:
        audio_codes = audio_codes.unsqueeze(0)
    
    # Generate REALISTIC duration targets
    target_durations = create_realistic_duration_targets(chunk_data, text_tokens, device)
    
    # COMPUTE INDIVIDUAL LOSSES
    
    # 1. DURATION LOSS (najważniejszy dla rozwiązania problemu)
    dur_loss = duration_prediction_loss(predicted_durations, target_durations)
    
    # 2. DIVERSITY LOSS (rozwiązuje problem stałych predictions)
    div_loss = diversity_loss(predicted_durations, target_weight=0.08)
    
    # 3. CONFIDENCE LOSS
    conf_loss = confidence_loss(duration_confidence, predicted_durations, target_durations)
    
    # 4. AUDIO TOKEN LOSS
    token_loss = audio_token_loss(logits, audio_codes)
    
    # COMBINE Z ODPOWIEDNIMI WAGAMI
    total_loss = (
        8.0 * dur_loss +      # GŁÓWNY - duration accuracy
        4.0 * div_loss +      # KLUCZOWY - diversity promotion  
        1.5 * conf_loss +     # Confidence
        1.0 * token_loss      # Audio tokens
    )
    
    # COMPUTE METRICS
    metrics = compute_accuracy_metrics(predicted_durations, target_durations, logits, audio_codes)
    
    # RETURN COMPLETE RESULTS
    result = {
        'total_loss': total_loss,
        'duration_loss': dur_loss,
        'diversity_loss': div_loss,
        'confidence_loss': conf_loss,
        'token_loss': token_loss,
        
        # Legacy compatibility
        'accuracy': metrics['audio_accuracy'],
        'duration_accuracy': metrics['duration_accuracy_medium'],  # Medium tolerance
        
        # Detailed metrics
        'duration_accuracy_strict': metrics['duration_accuracy_strict'],
        'duration_accuracy_loose': metrics['duration_accuracy_loose'],
        'duration_mae': metrics['duration_mae'],
        'duration_std': metrics['duration_std'],
        'audio_accuracy_per_codebook': metrics['audio_accuracy_per_codebook']
    }
    
    return result


# POMOCNICZE FUNKCJE dla backward compatibility

def safe_duration_loss(predicted_durations, target_durations, device):
    """Backward compatibility wrapper"""
    return duration_prediction_loss(predicted_durations, target_durations)

def safe_confidence_loss(confidence, predicted_durations, target_durations, device):
    """Backward compatibility wrapper"""
    return confidence_loss(confidence, predicted_durations, target_durations)

def safe_token_loss(logits, audio_codes, device):
    """Backward compatibility wrapper"""
    return audio_token_loss(logits, audio_codes)

def safe_accuracy(logits, audio_codes):
    """Backward compatibility wrapper"""
    metrics = compute_accuracy_metrics(None, None, logits, audio_codes)
    return metrics['audio_accuracy']

def safe_duration_accuracy(predicted_durations, target_durations, tolerance=0.3):
    """Backward compatibility wrapper"""
    metrics = compute_accuracy_metrics(predicted_durations, target_durations, None, None)
    return metrics['duration_accuracy_medium']

def get_safe_duration_targets(chunk_data, text_tokens, device):
    """Backward compatibility wrapper"""
    return create_realistic_duration_targets(chunk_data, text_tokens, device)


if __name__ == "__main__":
    print("🎯 Nowe losses.py - rozwiązuje problem diversity w duration prediction")
    print("Główne zmiany:")
    print("- Realistyczne duration targets z prawdziwą różnorodnością")
    print("- Diversity loss promujący wariację w predictions")
    print("- Ulepszone accuracy metrics z multiple tolerance levels")
    print("- Backward compatibility z istniejącym kodem")