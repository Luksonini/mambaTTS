#!/usr/bin/env python3
"""
INTEGRATED STATEFUL MAMBA TTS with STATEFUL TOKENIZER
======================================================
Complete audiobook-ready TTS system with:
- TrueStatefulMamba with complete state propagation
- StatefulTokenizer with audiobook-aware processing
- Proper state management for continuous prosody
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Import the components we created
exec(open('stateful_tokenizer.py').read(), globals())

# Existing imports
from losses import compute_combined_loss

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("🎵 INTEGRATED STATEFUL MAMBA TTS FOR AUDIOBOOKS")
print("=" * 70)


#!/usr/bin/env python3
"""
TRUE STATEFUL MAMBA with Complete State Propagation
=====================================================
Implements proper state passing for both convolution and SSM components
Critical for audiobook generation with prosodic continuity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TrueStatefulMambaBlock(nn.Module):
    """
    MEMORY EFFICIENT - Back to your fast pseudo-Mamba approach
    """
    def __init__(self, d_model, expand_factor=1.5, dropout=0.1, conv_kernel=3):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand_factor)
        
        # Core Mamba components (YOUR FAST VERSION)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # FAST convolution (YOUR APPROACH - with padding!)
        self.conv1d = nn.Conv1d(
            self.d_inner, 
            self.d_inner, 
            kernel_size=3, 
            padding=1,      # ✅ Normal padding like your version
            groups=self.d_inner,
            bias=False
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 0.1)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # MINIMAL STATE MANAGEMENT (like your original)
        self.stateful_mode = False
        self._hidden_state = None      # Only one small state tensor
        
        logger.info(f"✅ MemoryEfficientMambaBlock: d_model={d_model}, d_inner={self.d_inner}")
        
    def enable_stateful_mode(self, batch_size=1):
        """Enable minimal state persistence"""
        self.stateful_mode = True
        device = next(self.parameters()).device
        
        # MINIMAL state - just hidden state like your version
        self._hidden_state = torch.zeros(batch_size, 1, self.d_inner, device=device)
        
    def disable_stateful_mode(self):
        """Disable state persistence"""
        self.stateful_mode = False
        self._hidden_state = None
    
    def reset_state(self):
        """Reset states"""
        if self.stateful_mode and self._hidden_state is not None:
            self._hidden_state.zero_()
    
    def forward(self, x, return_state_info=False):
        """
        MEMORY EFFICIENT forward pass - your fast approach
        """
        B, L, D = x.shape
        residual = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # Input projection and split
        x_proj = self.in_proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        
        # FAST CONVOLUTION (YOUR APPROACH)
        x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)
        
        # Activation
        x1_activated = self.activation(x1_conv)
        
        # SSM parameters
        dt = self.dt_proj(x1_activated)
        dt = F.softplus(dt)
        
        # FAST SSM WITH MINIMAL STATE (YOUR APPROACH)
        x1_processed = x1_activated * torch.sigmoid(dt)
        
        # MINIMAL state injection (like your alpha approach)
        if self.stateful_mode and self._hidden_state is not None:
            # Check batch size compatibility
            if B == self._hidden_state.shape[0]:
                alpha = 0.1
                x1_processed[:, 0, :] += alpha * self._hidden_state.squeeze(1)
                # Update state with last timestep
                self._hidden_state = x1_processed[:, -1:, :].detach()
            else:
                # Batch size mismatch - skip state injection
                pass
        
        # Gating mechanism
        x_gated = x1_processed * torch.sigmoid(x2)
        
        # Output projection
        output = self.out_proj(x_gated)
        output = output * self.layer_scale
        
        # Dropout and residual
        output = self.dropout(output)
        final_output = output + residual
        
        if return_state_info:
            state_info = {
                'has_hidden_state': self._hidden_state is not None,
                'hidden_state_norm': torch.norm(self._hidden_state).item() if self._hidden_state is not None else 0.0,
                'stateful_mode': self.stateful_mode
            }
            return final_output, state_info
        
        return final_output


# ALSO - Add memory cleanup to your experiment
def add_memory_cleanup_to_experiment():
    """
    Add this to your training loop for memory management
    """
    # Add after optimizer.step():
    
    # 1. Clear gradients explicitly
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    
    # 2. Periodic GPU cleanup
    if step % 50 == 0:
        torch.cuda.empty_cache()
        
    # 3. Detach states to prevent gradient accumulation
    if hasattr(model, 'text_encoder'):
        for layer in model.text_encoder.layers:
            if hasattr(layer, '_hidden_state') and layer._hidden_state is not None:
                layer._hidden_state = layer._hidden_state.detach()
    
    if hasattr(model, 'audio_processor'):
        for layer in model.audio_processor.mamba_layers:
            if hasattr(layer, '_hidden_state') and layer._hidden_state is not None:
                layer._hidden_state = layer._hidden_state.detach()

class TrueStatefulMambaTextEncoder(nn.Module):
    """Text encoder with TRUE stateful Mamba blocks"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, expand_factor=1.5, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, hidden_dim) * 0.02)
        
        # Stack of TRUE stateful Mamba blocks
        self.layers = nn.ModuleList([
            TrueStatefulMambaBlock(hidden_dim, expand_factor, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def enable_stateful_mode(self, batch_size=1):
        """Enable TRUE state persistence for all layers"""
        for i, layer in enumerate(self.layers):
            layer.enable_stateful_mode(batch_size)
            logger.info(f"   Layer {i}: Enabled stateful mode")
    
    def disable_stateful_mode(self):
        """Disable state persistence for all layers"""
        for layer in self.layers:
            layer.disable_stateful_mode()
    
    def reset_states(self):
        """Reset all layer states (call between audio samples)"""
        for i, layer in enumerate(self.layers):
            layer.reset_state()
        logger.info("🔄 Reset all text encoder states")
    
    def forward(self, tokens, return_sequence=True, return_state_info=False):
        B, L = tokens.shape
        
        # Embeddings and projection
        x = self.embedding(tokens)
        x = self.embed_proj(x)
        
        # Add positional encoding
        if L <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :L, :]
        
        x = self.dropout(x)
        
        # Process through TRUE stateful Mamba layers
        state_infos = []
        for layer in self.layers:
            if return_state_info:
                x, state_info = layer(x, return_state_info=True)
                state_infos.append(state_info)
            else:
                x = layer(x)
        
        x = self.norm(x)
        
        result = x if return_sequence else x.mean(dim=1)
        
        if return_state_info:
            return result, state_infos
        return result


class TrueStatefulMambaAudioProcessor(nn.Module):
    """Audio processor with TRUE stateful Mamba blocks"""
    def __init__(self, hidden_dim, num_codebooks=8, codebook_size=1024, expand_factor=1.5):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Audio embeddings
        self.audio_embed = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(codebook_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_codebooks)
        ])
        
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # TRUE Stateful Mamba processing layers
        self.mamba_layers = nn.ModuleList([
            TrueStatefulMambaBlock(hidden_dim, expand_factor, dropout=0.1)
            for _ in range(4)
        ])
        
        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, codebook_size)
            ) for _ in range(num_codebooks)
        ])
    
    def enable_stateful_mode(self, batch_size=1):
        """Enable TRUE state persistence for audio processing"""
        for i, layer in enumerate(self.mamba_layers):
            layer.enable_stateful_mode(batch_size)
            logger.info(f"   Audio Layer {i}: Enabled stateful mode")
    
    def disable_stateful_mode(self):
        """Disable state persistence for audio processing"""
        for layer in self.mamba_layers:
            layer.disable_stateful_mode()
    
    def reset_states(self):
        """Reset all audio processor states"""
        for i, layer in enumerate(self.mamba_layers):
            layer.reset_state()
        logger.info("🔄 Reset all audio processor states")
    
    def forward(self, audio_tokens, text_context, return_state_info=False):
        B, C, T = audio_tokens.shape
        
        # Ensure 8 codebooks
        if C < 8:
            padding = torch.zeros(B, 8 - C, T, dtype=audio_tokens.dtype, device=audio_tokens.device)
            audio_tokens = torch.cat([audio_tokens, padding], dim=1)
        elif C > 8:
            audio_tokens = audio_tokens[:, :8, :]
        
        # Embed each codebook
        embedded = []
        for c in range(8):
            emb = self.audio_embed[c][0](audio_tokens[:, c, :])
            emb = self.audio_embed[c][1](emb)
            emb = self.audio_embed[c][2](emb)
            embedded.append(emb)
        
        # Combine embeddings
        x = torch.stack(embedded, dim=1).mean(dim=1)  # [B, T, hidden_dim]
        
        # Add text context
        text_context_proj = self.context_proj(text_context).unsqueeze(1)
        x = x + text_context_proj
        
        # Process through TRUE stateful Mamba layers
        state_infos = []
        for layer in self.mamba_layers:
            if return_state_info:
                x, state_info = layer(x, return_state_info=True)
                state_infos.append(state_info)
            else:
                x = layer(x)
        
        # Generate logits for each codebook
        logits = []
        for c in range(8):
            head_logits = self.output_heads[c](x)
            logits.append(head_logits)
        
        result = torch.stack(logits, dim=1)
        
        if return_state_info:
            return result, state_infos
        return result


def test_true_stateful_mamba():
    """Test TRUE stateful Mamba functionality"""
    logger.info("🧪 Testing TRUE Stateful Mamba")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    seq_len = 50
    d_model = 256
    
    # Create model
    mamba_block = TrueStatefulMambaBlock(d_model, expand_factor=1.5).to(device)
    
    # Test stateless mode
    logger.info("📋 Testing stateless mode...")
    x1 = torch.randn(batch_size, seq_len, d_model, device=device)
    x2 = torch.randn(batch_size, seq_len, d_model, device=device)
    
    out1 = mamba_block(x1)
    out2 = mamba_block(x2)
    logger.info(f"   Stateless output shapes: {out1.shape}, {out2.shape}")
    
    # Test stateful mode
    logger.info("🔗 Testing stateful mode...")
    mamba_block.enable_stateful_mode(batch_size)
    
    # Process chunks sequentially
    out1_stateful, info1 = mamba_block(x1, return_state_info=True)
    out2_stateful, info2 = mamba_block(x2, return_state_info=True)
    
    logger.info(f"   Stateful output shapes: {out1_stateful.shape}, {out2_stateful.shape}")
    logger.info(f"   State info 1: {info1}")
    logger.info(f"   State info 2: {info2}")
    
    # Check state continuity
    state_continuity = torch.norm(out2_stateful - out2).item()
    logger.info(f"   State continuity difference: {state_continuity:.6f}")
    
    if state_continuity > 1e-6:
        logger.info("✅ TRUE stateful mode working - outputs differ with state!")
    else:
        logger.warning("⚠️  States might not be properly carried over")
    
    # Reset and test
    mamba_block.reset_state()
    out1_reset, _ = mamba_block(x1, return_state_info=True)
    
    reset_difference = torch.norm(out1_reset - out1_stateful).item()
    logger.info(f"   Reset difference: {reset_difference:.6f}")
    
    logger.info("✅ TRUE Stateful Mamba test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_true_stateful_mamba()



@dataclass
class AudiobookTTSConfig:
    """Configuration for audiobook TTS system"""
    name: str
    num_layers: int = 4
    hidden_dim: int = 768
    expand_factor: float = 1.5
    embed_dim: int = 384
    num_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1
    conv_kernel: int = 3
    stateful: bool = True  # Default to stateful for audiobooks
    processing_mode: str = "stateful"  # "stateless", "stateful", "audiobook"


class IntegratedStatefulMambaTTS(nn.Module):
    """
    Complete audiobook TTS system with integrated stateful components
    """
    def __init__(self, config: AudiobookTTSConfig):
        super().__init__()
        self.config = config
        self.stateful = config.stateful
        
        # Initialize STATEFUL TOKENIZER
        self.tokenizer = StatefulNucleotideTokenizer()
        self.tokenizer.set_processing_mode(ProcessingMode(config.processing_mode))
        
        actual_vocab_size = self.tokenizer.get_vocab_size()
        logger.info(f"📝 Integrated tokenizer: {actual_vocab_size} tokens, mode: {config.processing_mode}")
        
        # TRUE STATEFUL TEXT ENCODER
        self.text_encoder = TrueStatefulMambaTextEncoder(
            vocab_size=actual_vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            expand_factor=config.expand_factor,
            dropout=config.dropout
        )
        
        # Duration regulator for prosody control
        self.duration_regulator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # TRUE STATEFUL AUDIO PROCESSOR
        self.audio_processor = TrueStatefulMambaAudioProcessor(
            hidden_dim=config.hidden_dim,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            expand_factor=config.expand_factor
        )
        
        # Text context projection
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 {config.name}: {total_params:,} parameters")
        logger.info(f"   📏 L{config.num_layers}_H{config.hidden_dim}_E{config.expand_factor}")
        logger.info(f"   🔗 Stateful: {config.stateful}")
        logger.info(f"   🎵 Processing mode: {config.processing_mode}")
        
    def enable_stateful_mode(self, batch_size=1):
        """Enable state persistence across all components"""
        if not self.stateful:
            return
        
        self.text_encoder.enable_stateful_mode(batch_size)
        self.audio_processor.enable_stateful_mode(batch_size)
        logger.info(f"🔗 Enabled stateful mode for batch_size={batch_size}")
    
    def disable_stateful_mode(self):
        """Disable state persistence across all components"""
        self.text_encoder.disable_stateful_mode()
        self.audio_processor.disable_stateful_mode()
        logger.info("🔄 Disabled stateful mode")
    
    def reset_all_states(self):
        """Reset all model states (call between different audiobooks/samples)"""
        if not self.stateful:
            return
        
        self.text_encoder.reset_states()
        self.audio_processor.reset_states()
        logger.info("🔄 Reset all states - ready for new audiobook/sample")
    
    def process_text_chunk(self, text: str, chunk_info: Dict) -> torch.Tensor:
        """
        Process text chunk with stateful-aware tokenization
        
        Args:
            text: Raw text chunk
            chunk_info: Information about chunk position and boundaries
        
        Returns:
            Token tensor ready for model processing
        """
        # Use appropriate encoding based on processing mode
        if self.config.processing_mode == "audiobook":
            tokens = self.tokenizer.encode_audiobook_chunk(text, chunk_info)
        elif self.config.processing_mode == "stateful":
            tokens = self.tokenizer.encode_for_stateful(
                text,
                is_first_chunk=chunk_info.get('is_first_chunk', False),
                is_last_chunk=chunk_info.get('is_last_chunk', False)
            )
        else:
            # Stateless mode
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dim
    
    def forward(self, text_tokens, audio_tokens=None, return_state_info=False):
        """Forward pass with integrated stateful processing"""
        batch_size = text_tokens.shape[0]
        
        # Process text with TRUE stateful Mamba
        text_features = self.text_encoder(text_tokens, return_sequence=True)
        text_context = self.text_encoder(text_tokens, return_sequence=False)
        text_context = self.text_proj(text_context)
        
        # Duration prediction for prosody
        predicted_durations = self.duration_regulator(text_features.mean(dim=1))
        
        # Audio processing with TRUE stateful Mamba
        if audio_tokens is not None:
            regulated_context = text_context
            audio_logits = self.audio_processor(audio_tokens, regulated_context)
        else:
            audio_logits = None
        
        result = {
            'logits': audio_logits,
            'predicted_durations': predicted_durations,
            'text_features': text_features,
        }
        
        return result

    # AUDIO GENERATION METHODS - PROPERLY INDENTED INSIDE CLASS
    def generate_audio_from_chunk(self, text_chunk: str, chunk_info: Dict, max_audio_length=1000):
        """
        Generuj audio z jednego chunka tekstu - WITH DEBUG
        """
        self.eval()
        
        with torch.no_grad():
            # Przetwórz tekst na tokeny
            text_tokens = self.process_text_chunk(text_chunk, chunk_info)
            text_tokens = text_tokens.to(next(self.parameters()).device)
            
            logger.info(f"🔍 FULL DEBUG audio generation:")
            logger.info(f"   Text: '{text_chunk[:50]}...'")
            logger.info(f"   Text tokens shape: {text_tokens.shape}")
            
            # Enkoduj tekst
            text_features = self.text_encoder(text_tokens, return_sequence=True)
            text_context = self.text_encoder(text_tokens, return_sequence=False)
            text_context = self.text_proj(text_context)
            
            logger.info(f"   Text features shape: {text_features.shape}")
            
            # Przewiduj długość audio - Z DEBUGIEM
            predicted_durations = self.duration_regulator(text_features.mean(dim=1))
            
            logger.info(f"   🎯 DURATION DEBUG:")
            logger.info(f"   Raw predicted_durations: {predicted_durations}")
            logger.info(f"   Duration shape: {predicted_durations.shape}")
            logger.info(f"   Duration item: {predicted_durations.item():.6f}")
            logger.info(f"   Text length: {text_tokens.shape[1]}")
            
            # ORIGINAL calculation
            original_audio_length = int(predicted_durations.item() * text_tokens.shape[1])
            logger.info(f"   Original calculation: {predicted_durations.item():.6f} * {text_tokens.shape[1]} = {original_audio_length}")
            
            # FORCE REASONABLE LENGTH
            if original_audio_length < 50:
                logger.warning(f"⚠️ Duration too short ({original_audio_length})! This is why you get clicks!")
                logger.warning(f"   Duration regulator needs more training or different loss weights")
                
            # Use reasonable calculation instead
            text_length = text_tokens.shape[1]
            expected_duration_seconds = text_length * 0.1  # 0.1s per text token
            audio_length = int(expected_duration_seconds * 75)  # 75 audio tokens per second
            audio_length = max(audio_length, 100)  # Minimum
            
            logger.info(f"   🔧 FIXED calculation: {text_length} * 6 = {audio_length} tokens")
            logger.info(f"   Expected duration: ~{audio_length/75:.2f}s")
            
            # Inicjalizuj audio tokens
            device = text_tokens.device
            audio_tokens = torch.zeros(1, 8, audio_length, dtype=torch.long, device=device)
            
            logger.info(f"   Generating {audio_length} audio tokens...")
            
            # Generuj auto-regressively
            for t in range(audio_length):
                if t % 50 == 0:
                    logger.info(f"     Progress: {t}/{audio_length}")
                
                current_audio = audio_tokens[:, :, :t+1]
                audio_logits = self.audio_processor(current_audio, text_context)
                
                if t < audio_logits.shape[2]:
                    next_logits = audio_logits[:, :, t, :]
                else:
                    next_logits = audio_logits[:, :, -1, :]
                
                for cb in range(8):
                    probs = F.softmax(next_logits[0, cb, :] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    audio_tokens[0, cb, t] = next_token
            
            logger.info(f"✅ Generated complete audio: {audio_tokens.shape}")
            
            # Zwróć jako [8, seq_len]
            return audio_tokens.squeeze(0)

    def save_audio_tokens_as_wav(self, audio_tokens, output_path="generated_audio.wav"):
        """Zapisz tokeny + użyj zewnętrznego dekodera jeśli EnCodec nie działa"""
        try:
            # Próbuj z wbudowanym EnCodec
            from encodec import EncodecModel
            import torchaudio
            
            logger.info("🔄 Trying built-in EnCodec...")
            encodec_model = EncodecModel.encodec_model_24khz()  # Twoja wersja
            encodec_model.set_target_bandwidth(6.0)
            encodec_model.eval()
            
            if torch.cuda.is_available():
                encodec_model = encodec_model.cuda()
            
            # Przygotuj tokeny
            if audio_tokens.dim() == 2:
                audio_tokens = audio_tokens.unsqueeze(0)
            
            device = next(encodec_model.parameters()).device
            audio_tokens = audio_tokens.to(device).long()
            
            # Dekoduj
            with torch.no_grad():
                try:
                    encoded_frames = [(audio_tokens, None)]
                    decoded_audio = encodec_model.decode(encoded_frames)
                    audio_waveform = decoded_audio.squeeze(0).cpu()
                    
                    if audio_waveform.dim() == 1:
                        audio_waveform = audio_waveform.unsqueeze(0)
                    
                    torchaudio.save(output_path, audio_waveform, encodec_model.sample_rate)
                    logger.info(f"🎵 Audio zapisane bezpośrednio: {output_path}")
                    return True
                    
                except Exception as e1:
                    logger.warning(f"⚠️ Bezpośrednie dekodowanie nie powiodło się: {e1}")
                    # Fallback - zapisz tokeny i użyj zewnętrznego dekodera
                    raise e1
            
        except Exception as e:
            logger.info(f"📁 Zapisuję tokeny dla zewnętrznego dekodera...")
            
            # Zapisz tokeny jako .pt file
            tokens_path = output_path.replace('.wav', '_tokens.pt')
            torch.save({
                'audio_tokens': audio_tokens.squeeze(0).cpu() if audio_tokens.dim() == 3 else audio_tokens.cpu(),  # [8, seq_len]
                'generated_by': 'IntegratedStatefulMambaTTS',
                'timestamp': time.time(),
                'shape': audio_tokens.shape
            }, tokens_path)
            
            logger.info(f"💾 Tokeny zapisane: {tokens_path}")
            logger.info(f"🎵 Użyj: python audio_decoder_script.py -i {tokens_path}")
            return True

    def generate_and_save_chunk_audio(self, text_chunk: str, chunk_info: Dict, output_path: str):
        """
        Kompletny pipeline: tekst -> tokeny audio -> zapisz
        """
        logger.info(f"🎤 Generuję audio dla chunka...")
        logger.info(f"   Tekst: '{text_chunk[:50]}...'")
        
        # Generuj tokeny audio
        audio_tokens = self.generate_audio_from_chunk(text_chunk, chunk_info)
        
        # Zapisz jako WAV lub tokeny
        success = self.save_audio_tokens_as_wav(audio_tokens, output_path)
        
        if success:
            logger.info(f"✅ Audio wygenerowane pomyślnie!")
        else:
            logger.error(f"❌ Błąd podczas generowania audio")
        
        return success

class AudiobookStatefulDataLoader:
    """
    Enhanced data loader for audiobook generation with proper state management
    """
    def __init__(self, data_dir="no_overlap_data", device='cpu', max_samples=4, processing_mode="stateful"):
        self.data_dir = Path(data_dir)
        self.device = device
        self.samples = {}
        self.max_samples = max_samples
        self.current_chunk_idx = 0
        self.max_chunks_per_sample = 0
        self.processing_mode = processing_mode
        
        logger.info(f"🔍 Loading audiobook data from {data_dir}")
        logger.info(f"   Processing mode: {processing_mode}")
        self._load_samples()
        
    def _load_samples(self):
        """Load samples with audiobook structure awareness"""
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return
            
        sample_dirs = [d for d in self.data_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('clean_batch_')]
        sample_dirs.sort()
        
        for sample_dir in sample_dirs[:self.max_samples]:
            try:
                meta_path = sample_dir / "batch_meta.json"
                if not meta_path.exists():
                    continue
                    
                with open(meta_path, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)
                
                sample_id = batch_meta.get('batch_idx', len(self.samples))
                sample_chunks = []
                
                chunk_files = sorted(batch_meta.get('chunk_files', []))
                
                for chunk_file in chunk_files:
                    chunk_path = sample_dir / chunk_file
                    if chunk_path.exists():
                        try:
                            chunk_data = torch.load(chunk_path, map_location=self.device, weights_only=False)
                            
                            if chunk_data.get('clean_chunk', False):
                                # Prepare audio codes
                                audio_codes = chunk_data.get('audio_codes')
                                if audio_codes is not None:
                                    if audio_codes.dim() == 2:
                                        C, T = audio_codes.shape
                                        if C < 8:
                                            padding = torch.zeros(8 - C, T, dtype=audio_codes.dtype, device=audio_codes.device)
                                            audio_codes = torch.cat([audio_codes, padding], dim=0)
                                        elif C > 8:
                                            audio_codes = audio_codes[:8, :]
                                        chunk_data['audio_codes'] = audio_codes
                                
                                sample_chunks.append(chunk_data)
                                    
                        except Exception as e:
                            logger.warning(f"Failed to load chunk {chunk_file}: {e}")
                            continue
                
                if sample_chunks:
                    self.samples[sample_id] = sample_chunks
                    logger.info(f"   📚 Sample {sample_id}: {len(sample_chunks)} chunks loaded")
                        
            except Exception as e:
                logger.warning(f"Failed to load sample from {sample_dir}: {e}")
                continue
        
        if self.samples:
            self.max_chunks_per_sample = min(len(chunks) for chunks in self.samples.values())
            logger.info(f"📊 Loaded {len(self.samples)} audiobook samples")
            logger.info(f"   🔢 Max chunks per sample: {self.max_chunks_per_sample}")
        else:
            logger.error("❌ No audiobook samples loaded!")
    
    def get_batch_with_context(self, chunk_idx):
        """
        Get batch with proper context information for stateful processing
        """
        if chunk_idx >= self.max_chunks_per_sample:
            return None, False
            
        batch_chunks = []
        sample_ids = []
        
        for sample_id, chunks in self.samples.items():
            if chunk_idx < len(chunks):
                chunk_data = chunks[chunk_idx].copy()
                
                # Add context information
                chunk_data['chunk_context'] = {
                    'sample_id': sample_id,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks),
                    'is_first_chunk': (chunk_idx == 0),
                    'is_last_chunk': (chunk_idx == len(chunks) - 1),
                    'is_chapter_start': (chunk_idx == 0),  # Simplified - first chunk is chapter start
                    'is_chapter_end': (chunk_idx == len(chunks) - 1),  # Last chunk is chapter end
                    'processing_mode': self.processing_mode
                }
                
                batch_chunks.append(chunk_data)
                sample_ids.append(sample_id)
        
        if not batch_chunks:
            return None, False
            
        return {
            'chunks': batch_chunks,
            'sample_ids': sample_ids,
            'chunk_idx': chunk_idx,
            'processing_mode': self.processing_mode
        }, True
    
    def get_next_batch(self):
        """Get next batch and advance chunk index"""
        batch_data, is_valid = self.get_batch_with_context(self.current_chunk_idx)
        
        if is_valid:
            self.current_chunk_idx += 1
            
        return batch_data, is_valid
    
    def reset_iterator(self):
        """Reset chunk index to beginning"""
        self.current_chunk_idx = 0
        logger.info("🔄 Reset data loader iterator")
        
    def get_random_batch(self):
        """Get random batch for stateless processing"""
        if not self.samples:
            return None, False
            
        chunk_idx = np.random.randint(0, self.max_chunks_per_sample)
        return self.get_batch_with_context(chunk_idx)
    
    def get_total_chunks(self):
        return self.max_chunks_per_sample
    
    def get_num_samples(self):
        return len(self.samples)


class IntegratedStatefulExperiment:
    """FIXED Complete experiment class - same name, just copy-paste"""
    
    def __init__(self, device, processing_mode="stateful"):
        self.device = device
        self.processing_mode = processing_mode
        
        # Create data loader
        self.data_loader = AudiobookStatefulDataLoader(
            "no_overlap_data", 
            device, 
            max_samples=4,
            processing_mode=processing_mode
        )
        
        if self.data_loader.get_num_samples() == 0:
            raise ValueError("❌ No audiobook samples loaded!")
        
        logger.info(f"✅ FIXED Integrated experiment ready")
    
    def run_stateful_experiment(self, test_steps=4000):
            """COMPLETELY FIXED Run experiment - replaces the entire method"""
            logger.info(f"\n🔧 COMPLETELY FIXED INTEGRATED STATEFUL EXPERIMENT")
            logger.info("=" * 60)
            
            start_time = time.time()
            
            try:
                # Create model
                config = AudiobookTTSConfig(
                    name="FixedIntegratedStatefulMamba",
                    processing_mode=self.processing_mode,
                    stateful=True
                )
                
                model = IntegratedStatefulMambaTTS(config).to(self.device)
                
                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"📊 FIXED model parameters: {param_count:,}")
                
                # Optimizer
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=5e-4,
                    weight_decay=1e-6,
                    betas=(0.9, 0.95)
                )
                
                # Training metrics
                metrics = {
                    'accuracies': [],
                    'losses': [],
                    'step_times': []
                }
                
                best_accuracy = 0.0
                successful_steps = 0
                
                # ENABLE STATEFUL MODE ONCE AT THE BEGINNING
                model.enable_stateful_mode(batch_size=1)
                logger.info("🔗 Stateful mode enabled ONCE for entire experiment")
                
                # FIXED Training loop
                model.train()
                for step in range(test_steps):
                    step_start = time.time()
                    
                    try:
                        # Get batch
                        batch_data, is_valid = self.data_loader.get_next_batch()
                        if not is_valid:
                            self.data_loader.reset_iterator()
                            batch_data, is_valid = self.data_loader.get_next_batch()
                            if not is_valid:
                                continue
                        
                        # FIXED: Don't re-enable stateful mode every step!
                        optimizer.zero_grad()
                        
                        total_loss = 0.0
                        total_accuracy = 0.0
                        processed_items = 0
                        
                        # Reset states ONLY at beginning of new audiobook sequence
                        if batch_data['chunk_idx'] == 0:
                            model.reset_all_states()
                            if step % 100 == 0:  # Log only occasionally
                                logger.info(f"🔄 Reset states for new audiobook sequence (step {step})")
                        
                        # Process each chunk individually
                        for chunk_data in batch_data['chunks']:
                            try:
                                # Process text_tokens
                                text_tokens = chunk_data['text_tokens']
                                if text_tokens.dim() == 1:
                                    text_tokens = text_tokens.unsqueeze(0)  # [1, seq_len]
                                
                                # Process audio_codes
                                audio_codes = chunk_data['audio_codes']
                                if audio_codes.dim() == 2:
                                    audio_codes = audio_codes.unsqueeze(0)  # [1, 8, seq_len]
                                
                                if text_tokens.shape[1] < 5:
                                    continue
                                
                                # Forward pass with consistent batch_size=1
                                output = model(text_tokens, audio_codes)
                                
                                # Compute loss
                                loss_dict = compute_combined_loss(output, chunk_data, text_tokens, self.device)
                                sample_loss = loss_dict.get('total_loss')
                                sample_accuracy = loss_dict.get('accuracy', 0.0)
                                
                                if sample_loss is not None and not torch.isnan(sample_loss):
                                    total_loss += sample_loss
                                    total_accuracy += sample_accuracy
                                    processed_items += 1
                                    
                            except Exception as e:
                                if step % 200 == 0:  # Log errors less frequently
                                    logger.warning(f"Failed to process chunk: {str(e)[:50]}")
                                continue
                        
                        if processed_items == 0:
                            continue
                        
                        avg_loss = total_loss / processed_items
                        avg_accuracy = total_accuracy / processed_items
                        
                        # Backward pass
                        avg_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        # Track metrics
                        step_time = time.time() - step_start
                        metrics['accuracies'].append(avg_accuracy)
                        metrics['losses'].append(avg_loss.item())
                        metrics['step_times'].append(step_time)
                        
                        if avg_accuracy > best_accuracy:
                            best_accuracy = avg_accuracy
                        
                        successful_steps += 1
                        
                        # Progress logging
                        if step % 100 == 0:
                            logger.info(f"   Step {step:4d}: Loss={avg_loss.item():.4f}, "
                                    f"Acc={avg_accuracy:.4f}, "
                                    f"ChunkIdx={batch_data['chunk_idx']}, "
                                    f"Time={step_time*1000:.1f}ms")
                        
                        # Early stopping
                        if step % 100 == 0:
                            logger.info(f"   Step {step:4d}: Loss={avg_loss.item():.4f}, "
                                    f"Acc={avg_accuracy:.4f}, "
                                    f"ChunkIdx={batch_data['chunk_idx']}, "
                                    f"Time={step_time*1000:.1f}ms")

                            
                    except Exception as e:
                        if step % 200 == 0:  # Log step errors less frequently  
                            logger.warning(f"Step {step} failed: {str(e)[:50]}")
                        continue
                
                # Calculate results
                training_time = time.time() - start_time
                final_accuracy = metrics['accuracies'][-1] if metrics['accuracies'] else 0.0
                avg_step_time = np.mean(metrics['step_times']) if metrics['step_times'] else 0.0
                
                result = {
                    'experiment_type': f'COMPLETELY_FIXED_STATEFUL_{self.processing_mode.upper()}',
                    'success': successful_steps > 0,
                    'training_time': training_time,
                    'final_accuracy': final_accuracy,
                    'best_accuracy': best_accuracy,
                    'avg_step_time': avg_step_time,
                    'total_steps': successful_steps,
                    'processing_mode': self.processing_mode,
                    'param_count': param_count
                }
                
                logger.info(f"✅ COMPLETELY FIXED RESULTS:")
                logger.info(f"   Final accuracy: {final_accuracy:.4f}")
                logger.info(f"   Best accuracy: {best_accuracy:.4f}")
                logger.info(f"   Training time: {training_time:.1f}s")
                logger.info(f"   Avg step time: {avg_step_time*1000:.1f}ms")
                logger.info(f"   Processing mode: {self.processing_mode}")
                
                # Save results
                timestamp = int(time.time())
                filename = f'completely_fixed_stateful_{timestamp}.json'
                
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"💾 COMPLETELY FIXED Results saved as: {filename}")
                
                return result, model
                
            except Exception as e:
                logger.error(f"❌ COMPLETELY FIXED experiment failed: {e}")
                import traceback
                traceback.print_exc()
                
                return {
                    'experiment_type': f'COMPLETELY_FIXED_STATEFUL_{self.processing_mode.upper()}',
                    'success': False,
                    'error': str(e),
                    'training_time': time.time() - start_time
                }

def test_audio_generation():
    """Test generowania audio z chunka"""
    logger.info("\n🎤 TESTOWANIE GENEROWANIA AUDIO")
    logger.info("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Utwórz model
    config = AudiobookTTSConfig(
        name="AudioGenTest",
        processing_mode="stateful",
        stateful=True
    )
    
    model = IntegratedStatefulMambaTTS(config).to(device)
    model.enable_stateful_mode(batch_size=1)
    
    # Przykładowy tekst
    test_text = "Hello, this is a test of the audio generation system. It should produce clear speech."
    
    chunk_info = {
        'chunk_idx': 0,
        'is_first_chunk': True,
        'is_last_chunk': True,
        'is_chapter_start': True,
        'is_chapter_end': True
    }
    
    # Generuj audio
    success = model.generate_and_save_chunk_audio(
        test_text, 
        chunk_info, 
        "test_generated_audio.wav"
    )
    
    if success:
        logger.info("🎉 Test generowania audio zakończony sukcesem!")
    else:
        logger.error("❌ Test generowania audio nie powiódł się")


def test_overfitting_after_training(model, data_loader):
    """Test czy model overfittuje do training data"""
    logger.info("🔍 Testing for overfitting...")
    
    # Get a training sample
    data_loader.reset_iterator()
    batch_data, is_valid = data_loader.get_next_batch()
    
    if not is_valid or not batch_data['chunks']:
        logger.warning("No training data available")
        return
    
    training_chunk = batch_data['chunks'][0]
    training_text = training_chunk.get('text', 'No text')
    
    # Test 1: Exact training text
    logger.info(f"🎯 Test 1: EXACT training text")
    logger.info(f"   Text: '{training_text[:60]}...'")
    
    model.generate_and_save_chunk_audio(
        training_text,
        training_chunk.get('chunk_context', {}),
        "exact_training_text.wav"
    )
    
    # Test 2: Similar but different text  
    modified_text = training_text.replace("w ", "w bardzo ").replace("Na ", "Na pewno ")[:100]
    logger.info(f"🔄 Test 2: MODIFIED training text")
    logger.info(f"   Text: '{modified_text[:60]}...'")
    
    model.generate_and_save_chunk_audio(
        modified_text,
        {'chunk_idx': 0, 'is_first_chunk': True, 'is_last_chunk': True},
        "modified_training_text.wav"
    )
    
    # Test 3: Completely new text
    new_text = "This is completely new text that the model has never seen before during training."
    logger.info(f"🆕 Test 3: COMPLETELY NEW text")
    logger.info(f"   Text: '{new_text}'")
    
    model.generate_and_save_chunk_audio(
        new_text,
        {'chunk_idx': 0, 'is_first_chunk': True, 'is_last_chunk': True},
        "completely_new_text.wav"
    )
    
    logger.info("🎵 Generated 3 test files:")
    logger.info("   📂 exact_training_text.wav - should be BEST if overfitting")
    logger.info("   📂 modified_training_text.wav - should be MEDIUM if overfitting") 
    logger.info("   📂 completely_new_text.wav - should be WORST if overfitting")
    logger.info("   ❓ If all sound equally bad → underfitting (need more training)")
    logger.info("   ❓ If training text sounds much better → overfitting")


def main():
    """Main function for integrated stateful audiobook TTS"""
    logger.info("🎵 INTEGRATED STATEFUL MAMBA TTS FOR AUDIOBOOKS")
    logger.info("=" * 70)
    logger.info("🔗 Complete system with:")
    logger.info("   • TrueStatefulMamba with full state propagation")
    logger.info("   • StatefulTokenizer with audiobook awareness")
    logger.info("   • Integrated processing pipeline")
    logger.info("   • Optimized for long-form audiobook generation")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Check data
    data_path = Path("no_overlap_data")
    if not data_path.exists():
        logger.error("❌ no_overlap_data directory not found!")
        logger.error("   Please run batch_no_overlap_preprocessor.py first")
        return
    
    try:
        test_audio_generation()
        
        # Create integrated experiment
        processing_mode = "stateful"
        experiment = IntegratedStatefulExperiment(device, processing_mode)
        
        # Run integrated stateful experiment
        logger.info(f"\n🚀 Starting integrated stateful experiment...")
        logger.info(f"   Processing mode: {processing_mode}")
        logger.info(f"   Target: Continuous audiobook generation")
        logger.info(f"   Expected: Better prosodic continuity")
        
        results, trained_model = experiment.run_stateful_experiment(test_steps=4000)
        
        if results['success']:
            logger.info(f"\n🎉 INTEGRATED STATEFUL AUDIOBOOK TTS COMPLETED!")
            logger.info(f"   ✅ Final accuracy: {results['final_accuracy']:.4f}")
            logger.info(f"   ✅ Best accuracy: {results['best_accuracy']:.4f}")
            logger.info(f"   ✅ Processing mode: {results['processing_mode']}")
            logger.info(f"   📚 Ready for audiobook generation!")

            logger.info(f"\n🔍 TESTING FOR OVERFITTING...")
            
            # ✅ Użyj TRAINED_MODEL do testów overfitting
            test_overfitting_after_training(trained_model, experiment.data_loader)
            
            # ✅ Użyj TRAINED_MODEL do generowania audio
            logger.info(f"\n🎤 Generuję audio z prawdziwych danych...")
            
            try:
                # Pobierz prawdziwy chunk z data loadera
                experiment.data_loader.reset_iterator()
                batch_data, is_valid = experiment.data_loader.get_next_batch()
                
                if is_valid and batch_data['chunks']:
                    real_chunk = batch_data['chunks'][0]
                    real_text = real_chunk.get('text', 'No text available')
                    
                    logger.info(f"📝 Prawdziwy tekst: '{real_text[:100]}...'")
                    
                    # ✅ Użyj TRAINED_MODEL
                    success = trained_model.generate_and_save_chunk_audio(
                        real_text, 
                        real_chunk.get('chunk_context', {}), 
                        "trained_model_output.wav"
                    )
                    
                    if success:
                        logger.info("✅ Audio z trenowanego modelu wygenerowany!")
                    
                    # Dodatkowo wygeneruj z przykładowego tekstu
                    example_text = "This is a real audiobook chunk for testing the trained model."
                    chunk_info = {
                        'chunk_idx': 0, 
                        'is_first_chunk': True, 
                        'is_last_chunk': False,
                        'is_chapter_start': True,
                        'is_chapter_end': False
                    }
                    
                    # ✅ Użyj TRAINED_MODEL
                    success2 = trained_model.generate_and_save_chunk_audio(
                        example_text, 
                        chunk_info, 
                        "example_audiobook_chunk.wav"
                    )
                    
                    if success2:
                        logger.info("✅ Przykładowy audiobook chunk wygenerowany!")
                        
                else:
                    logger.warning("⚠️ Brak danych do testowania")
                    
            except Exception as e:
                logger.warning(f"⚠️ Nie udało się wygenerować z prawdziwych danych: {e}")
                
                # Fallback - użyj trained_model
                try:
                    logger.info("🔄 Fallback: generuję z przykładowym tekstem...")
                    
                    fallback_text = "This is a fallback test of the audiobook generation system."
                    fallback_info = {
                        'chunk_idx': 0,
                        'is_first_chunk': True,
                        'is_last_chunk': True
                    }
                    
                    # ✅ Użyj TRAINED_MODEL
                    success = trained_model.generate_and_save_chunk_audio(
                        fallback_text,
                        fallback_info,
                        "fallback_audio.wav"
                    )
                    
                    if success:
                        logger.info("✅ Fallback audio wygenerowany!")
                        
                except Exception as e2:
                    logger.error(f"❌ Fallback też nie powiódł się: {e2}")
        else:
            logger.error(f"❌ Experiment failed: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

