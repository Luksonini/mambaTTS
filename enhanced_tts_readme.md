# Enhanced Prosody-Aware Mamba-Conv TTS System

🚀 **Advanced Text-to-Speech system with improved architecture and prosody modeling**

## 🎯 Key Features

- **Forward Mamba + Backward Convolutions**: Best of both worlds - global context and local patterns
- **FiLM Conditioning**: Advanced style modulation for prosody control
- **Depthwise Separable Convolutions**: Efficient style extraction
- **Learnable Stopping Decisions**: Adaptive generation control
- **Enhanced Duration Prediction**: Style-aware temporal modeling
- **Comprehensive Training Pipeline**: Advanced monitoring and logging

## 📁 File Structure

```
enhanced_tts_system/
├── modules.py                 # Enhanced prosody components
├── main_tts_model.py         # Core TTS architecture
├── training_system.py        # Training pipeline
├── main_runner.py            # Main system runner
├── nucleotide_tokenizer.py   # Text tokenization
└── README.md                 # This file
```

## 🔧 Architecture Components

### **Enhanced Prosody Modules** (`modules.py`)
- **AudioStyleExtractor**: Depthwise separable convolutions for efficient style extraction
- **FilmConditionedDurationPredictor**: FiLM modulation for style-aware duration prediction
- **NormalizedVarianceAdaptor**: LayerNorm stabilized pitch/energy control
- **EfficientLengthRegulator**: Tensor-optimized sequence expansion
- **AdaptiveGenerationController**: Learnable stopping decisions with proper projections

### **Main TTS Model** (`main_tts_model.py`)
- **MambaConvTextEncoder**: Forward Mamba + Backward multi-scale convolutions
- **ProsodyAwareMambaConvModel**: Main audio generation with hybrid architecture
- **ImprovedProsodyAwareTTS**: Complete system integration with all components

### **Training System** (`training_system.py`)
- **EnhancedDataExtractor**: Advanced data extraction with prosody information
- **EnhancedTTSTrainer**: Comprehensive training with multiple loss components
- **EnhancedGenerationTester**: Generation testing with multiple configurations

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchaudio
pip install encodec
pip install soundfile matplotlib numpy
```

### Required Files

- `speech.mp3` - Training audio file
- `speech_transcription.json` - Transcription with character alignments

### Training a New Model

```bash
# Basic training
python main_runner.py train --audio speech.mp3 --transcription speech_transcription.json

# Extended training with custom parameters
python main_runner.py train \
    --audio speech.mp3 \
    --transcription speech_transcription.json \
    --steps 25000 \
    --learning-rate 2e-4 \
    --test-after-training
```

### Testing an Existing Model

```bash
python main_runner.py test \
    --model best_model.pt \
    --audio speech.mp3 \
    --transcription speech_transcription.json
```

### Complete Pipeline (Train + Test)

```bash
python main_runner.py pipeline \
    --audio speech.mp3 \
    --transcription speech_transcription.json \
    --steps 20000
```

### Analyzing Training Results

```bash
python main_runner.py analyze --model best_model.pt
```

## 📊 Advanced Usage

### Custom Training Parameters

```bash
python main_runner.py train \
    --audio speech.mp3 \
    --transcription speech_transcription.json \
    --steps 30000 \
    --learning-rate 1e-4 \
    --vocab-size 1024 \
    --test-after-training \
    --log-level DEBUG
```

### CPU-only Training

```bash
python main_runner.py train \
    --audio speech.mp3 \
    --transcription speech_transcription.json \
    --cpu-only
```

## 🔧 Architecture Details

### **Hybrid Mamba-Conv Design**

```
TextEncoder:
  Forward Branch:  Mamba layers → Global context & state carrying
  Backward Branch: Multi-scale convolutions → Local patterns & parallel processing
  Fusion: Combines both branches for optimal representation

AudioModel:
  Style Conditioning: FiLM modulation throughout the network
  Duration Prediction: Style-aware with confidence estimation
  Variance Adaptor: Pitch/energy control with LayerNorm stabilization
  Generation Control: Learnable stopping decisions
```

### **Enhanced Components**

1. **AudioStyleExtractor**
   ```python
   # Depthwise separable convolutions
   depthwise_conv → pointwise_conv → BatchNorm → ReLU
   # Reduces parameters while maintaining effectiveness
   ```

2. **FiLM Conditioning**
   ```python
   # Feature-wise Linear Modulation
   γ, β = style_to_film(style_embedding)
   modulated_features = γ * text_features + β
   # Non-linear style conditioning vs simple addition
   ```

3. **Learnable Projections**
   ```python
   # Proper parameter management (not random weights!)
   text_projected = self.text_context_proj(text_context)
   audio_projected = self.audio_context_proj(audio_state)
   ```

## 📈 Training Monitoring

The system provides comprehensive training monitoring:

### **Real-time Metrics**
- Token accuracy
- Component losses (duration, style, pitch, energy, stopping)
- Gradient norms
- Learning rate schedule
- GPU memory usage
- Step times

### **Detailed Progress Reports**
- Trend analysis with moving averages
- ETA estimation
- Component-wise loss breakdown
- Memory usage tracking

### **Training Plots**
Automatically generated visualization plots:
- Loss curves
- Accuracy trends
- Learning rate schedule
- Component losses
- Training efficiency metrics

## 🎵 Generation Features

### **Iterative Refinement**
- Multiple refinement iterations with adaptive thresholds
- Confidence-based token updates
- Quality-based early stopping

### **Adaptive Generation**
- Dynamic stopping decisions based on content quality
- Learnable stopping thresholds
- Multiple iteration configurations for testing

### **Style-Aware Generation**
- Reference audio style extraction
- Style conditioning throughout generation
- Prosody-aware duration prediction

## 📁 Output Files

### **Training Outputs**
- `best_model.pt` - Best trained model checkpoint
- `checkpoint_*.pt` - Periodic training checkpoints
- `enhanced_training_plots_*.png` - Training visualization
- `enhanced_tts_system.log` - Complete training log

### **Generation Outputs**
- `enhanced_test_*_*iter.wav` - Generated audio samples
- Multiple iterations for quality comparison

## 🛠️ Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Use CPU-only mode
   python main_runner.py train --cpu-only --audio speech.mp3 --transcription speech_transcription.json
   ```

2. **File Not Found Errors**
   ```bash
   # Check file paths
   ls -la speech.mp3 speech_transcription.json
   ```

3. **Import Errors**
   ```bash
   # Ensure all files are in the same directory
   # Check that nucleotide_tokenizer.py exists
   ```

### **Performance Optimization**

1. **Reduce Memory Usage**
   - Lower batch size in data extraction
   - Reduce number of training steps
   - Use shorter audio fragments

2. **Speed Up Training**
   - Use mixed precision training
   - Reduce model dimensions
   - Use fewer refinement iterations

## 🧪 System Requirements

### **Minimum Requirements**
- Python 3.8+
- PyTorch 1.12+
- 8GB RAM
- 4GB storage

### **Recommended Requirements**
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU with 8GB+ VRAM
- 16GB RAM
- 10GB storage

## 📝 License & Citation

This enhanced TTS system builds upon state-of-the-art research in:
- Mamba/State Space Models for sequence modeling
- FastSpeech2 for duration and variance prediction
- FiLM conditioning for style transfer
- Depthwise separable convolutions for efficiency

## 🤝 Contributing

To contribute improvements:
1. Focus on modular design - each component in its own file
2. Maintain comprehensive logging and monitoring
3. Add proper error handling and validation
4. Include training plots and analysis
5. Document architectural decisions

## 🎯 Future Enhancements

Potential areas for improvement:
- Multi-speaker support
- Real-time generation optimization
- Advanced prosody analysis
- Integration with external vocoders
- Support for multiple languages

---

**🎉 Enhanced Prosody-Aware Mamba-Conv TTS System**  
*Advanced architecture with FiLM conditioning and improved components*
