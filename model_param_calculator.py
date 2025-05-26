#!/usr/bin/env python3
"""
Full Mamba Model Parameter Calculator
====================================
Dokładne wyliczenie parametrów dla Twojego Full Mamba modelu
"""

import torch
import torch.nn as nn

class ParameterCounter:
    """Helper class to count parameters with detailed breakdown"""
    
    def __init__(self):
        self.breakdown = {}
        self.total = 0
    
    def count_module(self, name, module):
        """Count parameters in a module"""
        params = sum(p.numel() for p in module.parameters())
        self.breakdown[name] = params
        self.total += params
        return params
    
    def count_linear(self, name, in_features, out_features, bias=True):
        """Count Linear layer parameters"""
        params = in_features * out_features
        if bias:
            params += out_features
        self.breakdown[name] = params
        self.total += params
        return params
    
    def count_embedding(self, name, num_embeddings, embedding_dim):
        """Count Embedding layer parameters"""
        params = num_embeddings * embedding_dim
        self.breakdown[name] = params
        self.total += params
        return params
    
    def count_conv1d(self, name, in_channels, out_channels, kernel_size, groups=1, bias=True):
        """Count Conv1d parameters"""
        params = (in_channels // groups) * out_channels * kernel_size
        if bias:
            params += out_channels
        self.breakdown[name] = params
        self.total += params
        return params
    
    def print_breakdown(self):
        """Print detailed parameter breakdown"""
        print("📊 DETAILED PARAMETER BREAKDOWN:")
        print("=" * 50)
        
        for name, params in self.breakdown.items():
            print(f"{name:35s}: {params:>10,} params")
        
        print("=" * 50)
        print(f"{'TOTAL':35s}: {self.total:>10,} params")
        print(f"{'SIZE (MB)':35s}: {self.total * 4 / (1024*1024):>10.1f} MB")


def calculate_full_mamba_block_params(d_model, expand_factor=2):
    """Calculate parameters for one FullMambaBlock"""
    counter = ParameterCounter()
    
    d_inner = d_model * expand_factor
    
    # Main projections
    counter.count_linear("in_proj", d_model, d_inner * 2, bias=False)
    counter.count_conv1d("conv1d", d_inner, d_inner, 3, groups=d_inner, bias=False)
    counter.count_linear("x_proj", d_inner, d_inner, bias=False)
    counter.count_linear("dt_proj", d_inner, d_inner, bias=True)
    counter.count_linear("out_proj", d_inner, d_model, bias=False)
    
    # LayerNorm parameters (approximation)
    counter.breakdown["layernorm"] = d_model * 2  # weight + bias
    counter.total += d_model * 2
    
    return counter.total, counter.breakdown


def calculate_full_mamba_model_params():
    """Calculate parameters for complete FullMambaTTSModel"""
    
    # Model configuration
    vocab_size = 131  # From nucleotide tokenizer
    embed_dim = 384
    hidden_dim = 512
    num_codebooks = 8
    codebook_size = 1024
    style_dim = 128
    
    counter = ParameterCounter()
    
    print("🔄 Full Mamba TTS Model Parameter Analysis")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_codebooks: {num_codebooks}")
    print(f"  codebook_size: {codebook_size}")
    print()
    
    # 1. TEXT ENCODER
    print("1️⃣ FullMambaTextEncoder:")
    
    # Embeddings
    counter.count_embedding("text_embedding", vocab_size, embed_dim)
    counter.count_embedding("pos_encoding", 2048, embed_dim)
    
    # 6 Mamba blocks
    mamba_block_params, _ = calculate_full_mamba_block_params(embed_dim)
    counter.breakdown["text_mamba_blocks (6x)"] = mamba_block_params * 6
    counter.total += mamba_block_params * 6
    
    # Final norm
    counter.breakdown["text_final_norm"] = embed_dim * 2
    counter.total += embed_dim * 2
    
    text_encoder_total = (counter.breakdown["text_embedding"] + 
                         counter.breakdown["pos_encoding"] + 
                         counter.breakdown["text_mamba_blocks (6x)"] + 
                         counter.breakdown["text_final_norm"])
    print(f"  Total: {text_encoder_total:,} params")
    print()
    
    # 2. DURATION REGULATOR
    print("2️⃣ FullMambaDurationRegulator:")
    
    # Input projection
    counter.count_linear("duration_input_proj", embed_dim + style_dim, 256)
    
    # 1 Mamba block
    mamba_duration_params, _ = calculate_full_mamba_block_params(256)
    counter.breakdown["duration_mamba_block"] = mamba_duration_params
    counter.total += mamba_duration_params
    
    # Prediction heads
    counter.count_linear("duration_predictor_1", 256, 128)
    counter.count_linear("duration_predictor_2", 128, 1)
    counter.count_linear("confidence_predictor_1", 256, 64)
    counter.count_linear("confidence_predictor_2", 64, 1)
    
    duration_total = (counter.breakdown["duration_input_proj"] + 
                     counter.breakdown["duration_mamba_block"] + 
                     counter.breakdown["duration_predictor_1"] + 
                     counter.breakdown["duration_predictor_2"] + 
                     counter.breakdown["confidence_predictor_1"] + 
                     counter.breakdown["confidence_predictor_2"])
    print(f"  Total: {duration_total:,} params")
    print()
    
    # 3. BACKWARD MAMBA STYLE EXTRACTOR
    print("3️⃣ BackwardMambaStyleExtractor:")
    
    # Input projection
    counter.count_linear("style_input_proj", hidden_dim, 256)
    
    # 1 Backward Mamba block
    mamba_style_params, _ = calculate_full_mamba_block_params(256)
    counter.breakdown["style_mamba_block"] = mamba_style_params
    counter.total += mamba_style_params
    
    # Style projection
    counter.count_linear("style_proj_1", 256, style_dim * 2)
    counter.count_linear("style_proj_2", style_dim * 2, style_dim)
    
    style_total = (counter.breakdown["style_input_proj"] + 
                  counter.breakdown["style_mamba_block"] + 
                  counter.breakdown["style_proj_1"] + 
                  counter.breakdown["style_proj_2"])
    print(f"  Total: {style_total:,} params")
    print()
    
    # 4. AUDIO PROCESSOR
    print("4️⃣ FullMambaAudioProcessor:")
    
    # Audio embeddings (8 codebooks)
    audio_embed_params = 0
    for i in range(num_codebooks):
        embed_params = counter.count_embedding(f"audio_embed_{i}", codebook_size, hidden_dim)
        # LayerNorm for each embedding
        counter.breakdown[f"audio_embed_norm_{i}"] = hidden_dim * 2
        counter.total += hidden_dim * 2
        audio_embed_params += embed_params + (hidden_dim * 2)
    
    # Context projection
    counter.count_linear("context_proj", hidden_dim, hidden_dim)
    
    # 4 Mamba blocks
    mamba_audio_params, _ = calculate_full_mamba_block_params(hidden_dim)
    counter.breakdown["audio_mamba_blocks (4x)"] = mamba_audio_params * 4
    counter.total += mamba_audio_params * 4
    
    # Output heads (8 codebooks)
    output_heads_params = 0
    for i in range(num_codebooks):
        # LayerNorm + Linear + Linear for each head
        head_params = (hidden_dim * 2 +  # LayerNorm
                      hidden_dim * (hidden_dim // 2) +  # Linear 1
                      (hidden_dim // 2) * codebook_size)  # Linear 2
        counter.breakdown[f"output_head_{i}"] = head_params
        counter.total += head_params
        output_heads_params += head_params
    
    audio_total = (audio_embed_params + 
                  counter.breakdown["context_proj"] + 
                  counter.breakdown["audio_mamba_blocks (4x)"] + 
                  output_heads_params)
    print(f"  Total: {audio_total:,} params")
    print()
    
    # 5. MISCELLANEOUS
    print("5️⃣ Miscellaneous:")
    
    # Text projection
    counter.count_linear("text_proj_1", embed_dim, hidden_dim)
    counter.breakdown["text_proj_norm"] = hidden_dim * 2
    counter.total += hidden_dim * 2
    
    # Default style parameter
    counter.breakdown["default_style"] = style_dim
    counter.total += style_dim
    
    misc_total = (counter.breakdown["text_proj_1"] + 
                 counter.breakdown["text_proj_norm"] + 
                 counter.breakdown["default_style"])
    print(f"  Total: {misc_total:,} params")
    print()
    
    # SUMMARY
    print("📊 COMPONENT SUMMARY:")
    print("=" * 50)
    print(f"Text Encoder:        {text_encoder_total:>12,} params ({text_encoder_total/counter.total*100:5.1f}%)")
    print(f"Duration Regulator:  {duration_total:>12,} params ({duration_total/counter.total*100:5.1f}%)")
    print(f"Style Extractor:     {style_total:>12,} params ({style_total/counter.total*100:5.1f}%)")
    print(f"Audio Processor:     {audio_total:>12,} params ({audio_total/counter.total*100:5.1f}%)")
    print(f"Miscellaneous:       {misc_total:>12,} params ({misc_total/counter.total*100:5.1f}%)")
    print("=" * 50)
    print(f"TOTAL:               {counter.total:>12,} params")
    print(f"Model Size:          {counter.total * 4 / (1024*1024):>12.1f} MB")
    print()
    
    # Comparison with other models
    print("📈 COMPARISON:")
    print("=" * 50)
    print(f"Your Full Mamba:     {counter.total/1e6:>8.1f}M params")
    print(f"Tacotron 2:          {28.0:>8.1f}M params")
    print(f"FastSpeech:          {30.0:>8.1f}M params")
    print(f"VITS:                {33.0:>8.1f}M params")
    print(f"F5-TTS:              {315.0:>8.1f}M params")
    print()
    
    return counter


def verify_with_actual_model():
    """Verify calculation by building actual model"""
    print("🔍 VERIFICATION with actual model:")
    print("=" * 50)
    
    try:
        # Try to import and create actual model
        # This would require your actual model code
        print("⚠️  To verify, run this in your training environment:")
        print("```python")
        print("model = FullMambaTTSModel(")
        print("    vocab_size=131,")
        print("    embed_dim=384,")
        print("    hidden_dim=512,")
        print("    num_codebooks=8,")
        print("    codebook_size=1024")
        print(")")
        print("total_params = sum(p.numel() for p in model.parameters())")
        print("print(f'Actual model: {total_params:,} parameters')")
        print("```")
        
    except Exception as e:
        print(f"Cannot create actual model: {e}")


if __name__ == "__main__":
    # Calculate parameters
    counter = calculate_full_mamba_model_params()
    
    # Print detailed breakdown
    print("\n" + "="*60)
    counter.print_breakdown()
    
    # Verification instructions
    print("\n" + "="*60)
    verify_with_actual_model()