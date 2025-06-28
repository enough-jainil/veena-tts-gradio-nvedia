# 🎵 Veena TTS - Gradio Web Interface

A beautiful and intuitive web interface for the **Veena Text-to-Speech model** developed by Maya Research. This Gradio application provides an easy-to-use interface for generating high-quality speech in Hindi, English, and code-mixed text.

## ✨ Features

- **🌍 Multi-lingual Support**: Hindi, English, and code-mixed text
- **🎭 4 Distinct Voices**: Choose from Kavya, Agastya, Maitri, and Vinaya
- **🎛️ Advanced Controls**: Adjust temperature and top-p for speech variation
- **📱 Responsive UI**: Modern, mobile-friendly interface
- **⚡ Real-time Generation**: Quick speech synthesis with progress tracking
- **🎧 Audio Playback**: Instant playback and download capabilities
- **📝 Quick Examples**: Pre-loaded example texts in different languages

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- At least 8GB GPU memory for 4-bit quantized model

### Installation

1. **Clone or download the application files**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

## 🎯 Usage Guide

### Basic Usage

1. **Enter Text**: Type or paste your text in Hindi, English, or code-mixed format
2. **Choose Voice**: Select from 4 available speakers:

   - 🎭 **Kavya** - Expressive female voice
   - 🎯 **Agastya** - Clear male voice
   - 💫 **Maitri** - Friendly female voice
   - 🎪 **Vinaya** - Warm male voice

3. **Adjust Settings** (optional):

   - **Temperature**: Controls creativity/variation (0.1-1.0)
   - **Top-p**: Controls focus/coherence (0.1-1.0)

4. **Generate**: Click "🎵 Generate Speech" and wait for processing

### Example Texts

The interface includes quick example buttons for:

- **Hindi**: Complex sentences in Devanagari script
- **English**: Standard English text
- **Code-mixed**: Hindi-English mixed sentences
- **Simple**: Basic test phrases

### Advanced Tips

- **Short texts** (1-2 sentences) work best for quality
- **Lower temperature** (0.2-0.4) for more consistent speech
- **Higher temperature** (0.6-0.8) for more expressive speech
- **Different speakers** have unique characteristics - experiment!

## 🔧 Technical Details

### Model Information

- **Architecture**: 3B parameter Llama-based transformer
- **Audio Quality**: 24kHz SNAC neural codec
- **Quantization**: 4-bit NF4 for efficient inference
- **Languages**: Hindi, English with code-mixing support
- **Latency**: Sub-80ms on H100, ~200ms on RTX 4090

### System Requirements

**Minimum:**

- GPU: 6GB VRAM (RTX 3060, RTX 4060)
- RAM: 8GB system RAM
- Storage: 10GB free space

**Recommended:**

- GPU: 12GB+ VRAM (RTX 4070, RTX 4080, RTX 4090)
- RAM: 16GB+ system RAM
- Storage: 20GB+ free space

### Performance Notes

- First run will download ~8GB of model files
- Initial model loading takes 30-60 seconds
- Subsequent generations are much faster
- GPU required for reasonable performance

### Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce batch size or use smaller quantization
2. **Model download fails**: Check internet connection and disk space
3. **Audio not playing**: Ensure browser supports WAV playback
4. **Slow generation**: Verify GPU is being used

**Solutions:**

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU memory
nvidia-smi

# Clear cache if needed
python -c "import torch; torch.cuda.empty_cache()"
```

## 🌐 Deployment Options

### Local Development

```bash
python app.py
```

### Production Deployment

```bash
# With custom port and sharing
python app.py --port 7860 --share

# Or modify the launch parameters in app.py
demo.launch(
    server_name="0.0.0.0",
    server_port=8000,
    share=True  # Enable public sharing
)
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

## 📝 Model Variants

This interface supports multiple Veena model variants:

- **maya-research/Veena** - Original model (default)
- **Prince-1/Veena-Onnx** - ONNX optimized version
- **hashvibe007/Veena-mlx-4Bit** - MLX optimized for Apple Silicon
- **Prince-1/Veena-Onnx-Int4** - INT4 quantized ONNX
- **Prince-1/Veena-RKllm** - Rockchip RK3588 optimized

To use a different variant, modify the model name in `app.py`:

```python
model = AutoModelForCausalLM.from_pretrained(
    "Prince-1/Veena-Onnx",  # Change this line
    # ... other parameters
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project follows the Apache 2.0 license of the original Veena model.

## 🙏 Acknowledgments

- **Maya Research** - For developing the amazing Veena TTS model
- **Hugging Face** - For hosting and model infrastructure
- **Gradio Team** - For the excellent web interface framework
- **Community** - For quantizations and optimizations

## 🔗 Links

- [🤗 Veena Model Hub](https://huggingface.co/maya-research/Veena)
- [📚 Model Documentation](https://huggingface.co/maya-research/Veena/blob/main/README.md)
- [🐛 Report Issues](https://github.com/your-repo/issues)
- [💬 Discussions](https://huggingface.co/maya-research/Veena/discussions)

---

**Made with ❤️ for the Indian Language AI Community**
