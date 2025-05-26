#!/usr/bin/env python3
"""
Main TTS Model Architecture
===========================
Complete TTS system combining:
- MambaConvTextEncoder: Forward Mamba + Backward Convolutions
- ProsodyAwareMambaConvModel: Main audio generation model
- ImprovedProsodyAwareTTS: Complete system integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from modules import (
    AudioStyleExtractor, FilmConditionedDurationPredictor, NormalizedVarianceAdaptor,
    EfficientLengthRegulator, AdaptiveGenerationController, OptimizedSSMBlock
)

logger = logging.getLogger(__name__)

# ============================================================================
# TEXT ENCODER
# ============================================================================

class MambaConvTextEncoder(nn.Module):
    """
    Text Encoder: Forward Mamba + Backward Multi-scale Convolutions + Prosody
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, style_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # FORWARD: Sequential Mamba
        self.forward_mamba = nn.ModuleList([
            OptimizedSSMBlock(embed_dim) for _ in range(3)
        ])
        
        # BACKWARD: Multi-scale Convolutions
        self.backward_pyramid = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=1, padding=1, groups=4),
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=2, padding=2, groups=4),
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=4, padding=4, groups=4),
            nn.Conv1d(embed_dim, embed_dim//4, kernel_size=3, dilation=8, padding=8, groups=4),
        ])
        
        self.backward_pointwise = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.backward_norm = nn.LayerNorm(embed_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Style conditioning with FiLM
        self.style_to_film = nn.Linear(style_dim, embed_dim * 2)
        
        # Global attention
        self.global_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        logger.info(f"🧠 MambaConvTextEncoder: {embed_dim}D embeddings, FiLM style conditioning")
        
    def forward(self, tokens, style_embedding=None, return_sequence=True):
        """
        Args:
            tokens: [B, T] token indices
            style_embedding: [B, style_dim] - optional prosody style
            return_sequence: If True, return per-token embeddings
        """
        B, T = tokens.shape
        device = tokens.device
        
        # Token embedding
        x = self.token_embedding(tokens)  # [B, T, embed_dim]
        
        # Style conditioning with FiLM (if provided)
        if style_embedding is not None:
            film_params = self.style_to_film(style_embedding)  # [B, embed_dim * 2]
            gamma, beta = torch.chunk(film_params, 2, dim=-1)  # [B, embed_dim] each
            gamma = gamma.unsqueeze(1).expand(-1, T, -1)
            beta = beta.unsqueeze(1).expand(-1, T, -1)
            x = gamma * x + beta
        
        # FORWARD BRANCH: Sequential Mamba
        forward_features = x
        for mamba_layer in self.forward_mamba:
            u = forward_features.mean(-1, keepdim=True)  # [B, T, 1]
            y = mamba_layer(u)  # [B, T, 1] 
            forward_features = forward_features + y.expand(-1, -1, self.embed_dim)
        
        # BACKWARD BRANCH: Multi-scale Convolutions
        x_conv = x.transpose(1, 2)  # [B, embed_dim, T]
        
        pyramid_features = []
        for conv_layer in self.backward_pyramid:
            scale_features = conv_layer(x_conv)  # [B, embed_dim//4, T]
            scale_features = scale_features.to(device)
            pyramid_features.append(scale_features)
        
        multi_scale_features = torch.cat(pyramid_features, dim=1)  # [B, embed_dim, T]
        backward_features = self.backward_pointwise(multi_scale_features)
        backward_features = backward_features + x_conv  # Residual
        backward_features = self.backward_norm(backward_features.transpose(1, 2)).transpose(1, 2)
        backward_features = backward_features.transpose(1, 2)  # [B, T, embed_dim]
        
        # FUSION
        combined = torch.cat([forward_features, backward_features], dim=-1)
        token_embeddings = self.fusion(combined)  # [B, T, embed_dim]
        
        if return_sequence:
            return token_embeddings
        else:
            attention_scores = self.global_attention(token_embeddings).squeeze(-1)  # [B, T]
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(1)  # [B, 1, T]
            pooled = torch.bmm(attention_weights, token_embeddings).squeeze(1)  # [B, embed_dim]
            return pooled


# ============================================================================
# MAIN AUDIO MODEL
# ============================================================================

class ProsodyAwareMambaConvModel(nn.Module):
    """
    Main model: Hybrid Mamba-Conv + Improved Prosody features
    """
    def __init__(self, text_dim, num_codebooks, codebook_size, hidden_dim, style_dim=128):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        
        # Token embeddings
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim) for _ in range(num_codebooks)
        ])
        
        # Position and confidence embeddings
        self.pos_embedding = nn.Embedding(1000, hidden_dim)
        self.confidence_embedding = nn.Embedding(20, hidden_dim)
        
        # Text conditioning
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Style conditioning with FiLM
        self.style_to_film = nn.Linear(style_dim, hidden_dim * 2)
        
        # FORWARD BRANCH: Sequential Mamba layers
        self.forward_layers = nn.ModuleList([
            OptimizedSSMBlock(hidden_dim) for _ in range(3)
        ])
        
        # BACKWARD BRANCH: Multi-scale dilated convolutions
        self.backward_pyramid = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=1, padding=1, groups=4),
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=3, padding=3, groups=4),
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=9, padding=9, groups=4),
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=3, dilation=27, padding=27, groups=4),
        ])
        
        self.backward_pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.backward_norm = nn.LayerNorm(hidden_dim)
        
        # Branch fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, codebook_size)
            ) for _ in range(num_codebooks)
        ])
        
        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"🔧 ProsodyAwareMambaConvModel: {hidden_dim}D, FiLM conditioning")
    
    def forward(self, text_emb, current_tokens, style_embedding=None, confidence_level=5):
        B, C, T = current_tokens.shape
        
        # Token embedding
        token_embeds = []
        for cb_idx in range(C):
            cb_tokens = current_tokens[:, cb_idx, :]
            cb_embed = self.token_embeddings[cb_idx](cb_tokens)
            token_embeds.append(cb_embed)
        
        x = torch.stack(token_embeds, dim=2).mean(dim=2)  # [B, T, H]
        
        # Conditioning
        positions = torch.arange(T, device=current_tokens.device)
        pos_embed = self.pos_embedding(positions).unsqueeze(0).expand(B, -1, -1)
        
        conf_level = torch.full((T,), min(confidence_level, 19), device=current_tokens.device)
        conf_embed = self.confidence_embedding(conf_level).unsqueeze(0).expand(B, -1, -1)
        
        text_cond = self.text_proj(text_emb).unsqueeze(1).expand(-1, T, -1)
        
        # Base features
        x = x + pos_embed + conf_embed + text_cond
        
        # Style conditioning with FiLM (if provided)
        if style_embedding is not None:
            film_params = self.style_to_film(style_embedding)  # [B, hidden_dim * 2]
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            gamma = gamma.unsqueeze(1).expand(-1, T, -1)
            beta = beta.unsqueeze(1).expand(-1, T, -1)
            x = gamma * x + beta
        
        # FORWARD BRANCH: Sequential Mamba
        forward_features = x
        for layer in self.forward_layers:
            u = forward_features.mean(-1, keepdim=True)
            y = layer(u)
            forward_features = forward_features + y.expand(-1, -1, self.hidden_dim)
        
        # BACKWARD BRANCH: Multi-scale convolutions
        x_conv = x.transpose(1, 2)  # [B, H, T]
        
        pyramid_features = []
        for conv_layer in self.backward_pyramid:
            scale_features = conv_layer(x_conv)
            pyramid_features.append(scale_features)
        
        multi_scale_features = torch.cat(pyramid_features, dim=1)
        backward_features = self.backward_pointwise(multi_scale_features)
        backward_features = backward_features + x_conv
        backward_features = self.backward_norm(backward_features.transpose(1, 2)).transpose(1, 2)
        backward_features = backward_features.transpose(1, 2)  # [B, T, H]
        
        # FUSION
        fused_features = torch.cat([forward_features, backward_features], dim=-1)
        final_features = self.fusion_layer(fused_features)
        
        # OUTPUT GENERATION
        outputs = []
        for cb_idx in range(self.num_codebooks):
            cb_output = self.output_heads[cb_idx](final_features)
            outputs.append(cb_output)
        
        logits = torch.stack(outputs, dim=1)  # [B, C, T, codebook_size]
        confidence = self.confidence_head(final_features).squeeze(-1)  # [B, T]
        
        return {
            'logits': logits,
            'confidence': confidence,
            'forward_features': forward_features,
            'backward_features': backward_features,
            'final_features': final_features
        }


# ============================================================================
# COMPLETE TTS SYSTEM
# ============================================================================

class ImprovedProsodyAwareTTS(nn.Module):
    """
    Complete improved TTS system with better architecture and naming
    """
    def __init__(self, vocab_size, text_dim=256, style_dim=128, hidden_dim=512, 
                 num_codebooks=4, codebook_size=1024):
        super().__init__()
        self.text_dim = text_dim
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        
        # Core components with improved architecture
        self.text_encoder = MambaConvTextEncoder(vocab_size, text_dim, hidden_dim, style_dim)
        self.audio_style_extractor = AudioStyleExtractor(hidden_dim, style_dim)
        self.duration_predictor = FilmConditionedDurationPredictor(text_dim, style_dim)
        self.variance_adaptor = NormalizedVarianceAdaptor(text_dim, style_dim)
        self.length_regulator = EfficientLengthRegulator()
        self.generation_controller = AdaptiveGenerationController(text_dim, hidden_dim)
        self.main_model = ProsodyAwareMambaConvModel(text_dim, num_codebooks, codebook_size, hidden_dim, style_dim)
        
        logger.info(f"🚀 ImprovedProsodyAwareTTS: Complete system with enhanced architecture")
    
    def forward(self, text_tokens, reference_audio=None, target_durations=None, 
                target_pitch=None, target_energy=None):
        """
        Forward pass through improved TTS system
        
        Args:
            text_tokens: [B, T_text] - input text tokens
            reference_audio: [B, hidden_dim, T_audio] - reference audio for style
            target_durations: [B, T_text] - target durations (training only)
            target_pitch/energy: [B, T_text] - targets for variance adaptor
        """
        batch_size, text_len = text_tokens.shape
        device = text_tokens.device
        
        # 1. Extract prosody style from reference audio
        if reference_audio is not None:
            style_embedding = self.audio_style_extractor(reference_audio)  # [B, style_dim]
        else:
            style_embedding = torch.zeros(batch_size, self.style_dim, device=device)
        
        # 2. Encode text with style conditioning
        text_features = self.text_encoder(text_tokens, style_embedding, return_sequence=True)  # [B, T_text, text_dim]
        
        # 3. Predict durations with FiLM conditioning
        predicted_durations, duration_confidence = self.duration_predictor(text_features, style_embedding)
        durations = target_durations if target_durations is not None else predicted_durations
        
        # 4. Apply variance adaptor (pitch/energy) with normalization
        enhanced_features, pred_pitch, pred_energy = self.variance_adaptor(
            text_features, style_embedding, target_pitch, target_energy
        )
        
        # 5. Length regulation with efficient tensor operations
        regulated_features = self.length_regulator(enhanced_features, durations)  # [B, T_expanded, text_dim]
        
        # 6. Generate audio tokens through main model
        # Convert regulated features to proper format for main model
        text_context = torch.mean(regulated_features, dim=1)  # [B, text_dim] - global context
        target_length = regulated_features.shape[1]
        
        # Initialize random tokens for iterative refinement
        current_tokens = torch.randint(
            0, self.codebook_size, 
            (batch_size, self.num_codebooks, target_length), 
            device=device
        )
        
        # Forward through main model
        model_output = self.main_model(text_context, current_tokens, style_embedding)
        
        # 7. Adaptive stopping decision
        avg_audio_state = torch.mean(model_output['final_features'], dim=1)  # [B, hidden_dim]
        continue_prob, decision_confidence = self.generation_controller(avg_audio_state, text_context)
        
        return {
            'logits': model_output['logits'],
            'predicted_durations': predicted_durations,
            'duration_confidence': duration_confidence,
            'predicted_pitch': pred_pitch,
            'predicted_energy': pred_energy,
            'continue_prob': continue_prob,
            'decision_confidence': decision_confidence,
            'style_embedding': style_embedding,
            'regulated_length': target_length,
            'model_confidence': model_output['confidence']
        }
    
    def generate_iteratively(self, text_tokens, reference_audio=None, iterations=12, min_iterations=6):
        """
        Generate with iterative refinement and adaptive stopping
        
        Args:
            text_tokens: [B, T_text] - input text tokens
            reference_audio: [B, hidden_dim, T_audio] - optional reference audio
            iterations: Maximum number of refinement iterations
            min_iterations: Minimum iterations before allowing early stop
        
        Returns:
            generated_tokens: [B, C, T] - final generated audio tokens
            generation_info: Dict with generation statistics
        """
        self.eval()
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        with torch.no_grad():
            # Initial forward pass to get length and initial tokens
            initial_output = self.forward(text_tokens, reference_audio)
            
            current_tokens = torch.argmax(initial_output['logits'], dim=-1)  # [B, C, T]
            text_context = self.text_encoder(text_tokens, initial_output['style_embedding'], return_sequence=False)
            
            generation_history = []
            
            for iteration in range(iterations):
                # Forward through main model
                model_output = self.main_model(
                    text_context, current_tokens, 
                    style_embedding=initial_output['style_embedding']
                )
                
                logits = model_output['logits']
                confidence = model_output['confidence']
                
                # Adaptive refinement
                avg_confidence = confidence.mean().item()
                
                # Stopping decision
                avg_audio_state = torch.mean(model_output['final_features'], dim=1)
                continue_prob, decision_conf = self.generation_controller(avg_audio_state, text_context)
                
                # Dynamic threshold
                base_threshold = 0.15
                iteration_factor = 0.03 * (iteration / iterations)
                confidence_factor = 0.1 * (1 - avg_confidence)
                dynamic_threshold = base_threshold + iteration_factor + confidence_factor
                
                should_continue = continue_prob.mean().item() > dynamic_threshold
                
                # Update tokens based on confidence
                confidence_threshold = 0.2 + 0.7 * (iteration / iterations)
                updates_made = 0
                
                C, T = current_tokens.shape[1], current_tokens.shape[2]
                for cb_idx in range(C):
                    low_conf_mask = confidence < confidence_threshold
                    if low_conf_mask.sum() > 0:
                        new_tokens = torch.argmax(logits[:, cb_idx], dim=-1)
                        current_tokens[:, cb_idx][low_conf_mask] = new_tokens[low_conf_mask]
                        updates_made += low_conf_mask.sum().item()
                
                # Track generation progress
                generation_info = {
                    'iteration': iteration,
                    'confidence': avg_confidence,
                    'continue_prob': continue_prob.mean().item(),
                    'decision_confidence': decision_conf.mean().item(),
                    'threshold': dynamic_threshold,
                    'updates': updates_made,
                    'should_continue': should_continue
                }
                generation_history.append(generation_info)
                
                # Early stopping logic
                if not should_continue and iteration >= min_iterations:
                    break
                
                # Quality-based stopping
                if avg_confidence > 0.95 and decision_conf.mean().item() > 0.9 and iteration >= min_iterations:
                    break
            
            # Compile final results
            final_info = {
                'total_iterations': len(generation_history),
                'final_confidence': generation_history[-1]['confidence'],
                'final_continue_prob': generation_history[-1]['continue_prob'],
                'total_updates': sum(h['updates'] for h in generation_history),
                'generation_history': generation_history,
                'predicted_length': current_tokens.shape[-1],
                'style_used': reference_audio is not None
            }
            
            return current_tokens, final_info
    
    def get_parameter_count(self):
        """Get detailed parameter count for each component"""
        component_params = {
            'text_encoder': sum(p.numel() for p in self.text_encoder.parameters()),
            'audio_style_extractor': sum(p.numel() for p in self.audio_style_extractor.parameters()),
            'duration_predictor': sum(p.numel() for p in self.duration_predictor.parameters()),
            'variance_adaptor': sum(p.numel() for p in self.variance_adaptor.parameters()),
            'length_regulator': sum(p.numel() for p in self.length_regulator.parameters()),
            'generation_controller': sum(p.numel() for p in self.generation_controller.parameters()),
            'main_model': sum(p.numel() for p in self.main_model.parameters())
        }
        
        total_params = sum(component_params.values())
        component_params['total'] = total_params
        
        return component_params


# ============================================================================
# MODEL FACTORY FUNCTIONS
# ============================================================================

def create_improved_tts_model(vocab_size, text_dim=256, style_dim=128, hidden_dim=512, 
                             num_codebooks=4, codebook_size=1024, device='cuda'):
    """
    Factory function to create and initialize improved TTS model
    
    Args:
        vocab_size: Size of text vocabulary
        text_dim: Text embedding dimension
        style_dim: Style embedding dimension
        hidden_dim: Hidden layer dimension
        num_codebooks: Number of audio codebooks
        codebook_size: Size of each codebook
        device: Device to place model on
    
    Returns:
        model: Initialized ImprovedProsodyAwareTTS model
    """
    model = ImprovedProsodyAwareTTS(
        vocab_size=vocab_size,
        text_dim=text_dim,
        style_dim=style_dim,
        hidden_dim=hidden_dim,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size
    ).to(device)
    
    # Log model info
    param_counts = model.get_parameter_count()
    logger.info(f"🚀 Created ImprovedProsodyAwareTTS model:")
    for component, count in param_counts.items():
        logger.info(f"  📝 {component}: {count:,} parameters")
    
    return model


def load_improved_tts_model(checkpoint_path, vocab_size, device='cuda', **model_kwargs):
    """
    Load improved TTS model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        vocab_size: Size of text vocabulary
        device: Device to load model on
        **model_kwargs: Additional model configuration arguments
    
    Returns:
        model: Loaded model
        checkpoint_info: Dictionary with checkpoint metadata
    """
    # Create model
    model = create_improved_tts_model(vocab_size, device=device, **model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['tts_system'])
    
    checkpoint_info = {
        'step': checkpoint.get('step', 0),
        'accuracy': checkpoint.get('accuracy', 0.0),
        'training_metrics': checkpoint.get('training_metrics', {})
    }
    
    logger.info(f"✅ Loaded model from {checkpoint_path}")
    logger.info(f"   Step: {checkpoint_info['step']}, Accuracy: {checkpoint_info['accuracy']:.4f}")
    
    return model, checkpoint_info


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example of creating and using the model
    import torch
    
    # Setup
    vocab_size = 1000
    batch_size = 2
    text_length = 20
    audio_length = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = create_improved_tts_model(vocab_size, device=device)
    
    # Example input
    text_tokens = torch.randint(0, vocab_size, (batch_size, text_length), device=device)
    reference_audio = torch.randn(batch_size, 512, audio_length, device=device)
    
    # Forward pass
    output = model(text_tokens, reference_audio)
    
    print("✅ Model forward pass successful!")
    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Duration confidence: {output['duration_confidence'].mean().item():.3f}")
    print(f"   Model confidence: {output['model_confidence'].mean().item():.3f}")
    
    # Iterative generation
    generated_tokens, gen_info = model.generate_iteratively(text_tokens, reference_audio)
    
    print(f"✅ Iterative generation successful!")
    print(f"   Generated shape: {generated_tokens.shape}")
    print(f"   Total iterations: {gen_info['total_iterations']}")
    print(f"   Final confidence: {gen_info['final_confidence']:.3f}")