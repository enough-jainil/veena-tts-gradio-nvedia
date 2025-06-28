# 🎵 Veena TTS - Gradio Web Interface

![Image of gui](./img/image.png)

A beautiful and intuitive web interface for the **Veena Text-to-Speech model** developed by Maya Research. This Gradio application provides an easy-to-use interface for generating high-quality speech in Hindi, English, and code-mixed text.

## ✨ Features

- **🌍 Multi-lingual Support**: Hindi, English, and code-mixed text
- **🎭 11 Distinct Voices**: Choose from multiple professionally trained speakers
- **🎛️ Advanced Controls**: Adjust temperature and top-p for speech variation
- **📱 Responsive UI**: Modern, mobile-friendly interface
- **⚡ Real-time Generation**: Quick speech synthesis with progress tracking
- **🎧 Audio Playback**: Instant playback and download capabilities
- **📝 Quick Examples**: Pre-loaded example texts in different languages

## 🚀 Quick Start

### Prerequisites

#### Mainly i tested in these idk about the others so try and let me know 💖

| Requirement                  | Recommended Version                                                         |
| ---------------------------- | --------------------------------------------------------------------------- |
| Python                       | **3.9 – 3.12**                                                              |
| PyTorch                      | **≥ 2.2.0** (built with CUDA 12.1)                                          |
| CUDA Toolkit (NVIDIA driver) | **CUDA 12.1** runtime (Driver ≥ 545)                                        |
| NVIDIA GPU                   | RTX 30-series/40-series <br> **≥ 6 GB VRAM** (≥ 12 GB strongly recommended) |
| System RAM                   | ≥ 16 GB                                                                     |

On Linux/Windows the official PyTorch wheels already include the necessary CUDA libraries – a separate toolkit install is **not** required. If you are running on a laptop GPU with <6 GB VRAM or on CPU-only hardware you can still launch the app (see the _CPU fallback_ section) but generation will be slow.

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
2. **Choose Voice**: Select from 11 available speakers:

   - 🎭 **Kavya** - Expressive female voice
   - ✨ **Apsara** - Celestial female voice
   - 🎯 **Agastya** - Sage male voice
   - 🎪 **Vinaya** - Warm male voice
   - 💫 **Maitri** - Friendly female voice
   - 🌸 **Charu** - Graceful female voice
   - 🏔️ **Ishana** - Noble male voice
   - 🌟 **Kyra** - Bright female voice
   - 🎨 **Mohini** - Enchanting female voice
   - 🌊 **Varun** - Strong male voice
   - 🌙 **Soumya** - Gentle unisex voice

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

### Dependency Matrix

| Package                                  | Tested Version    |
| ---------------------------------------- | ----------------- |
| `torch`                                  | 2.2.2 + CUDA 12.1 |
| `torchaudio`                             | 2.2.2             |
| `transformers`                           | 4.52.4            |
| `bitsandbytes`                           | 0.43.0            |
| `snac`                                   | 0.1.5             |
| `gradio`                                 | 4.27.0            |
| `soundfile`, `numpy`, `scipy`, `librosa` | latest            |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### GPU vs CPU behaviour

The application auto-detects a CUDA-capable GPU:

- **GPU available →** loads Veena in 4-bit NF4 quantisation using `bitsandbytes` (`device_map="auto"`). This is memory-efficient (≈5 GB VRAM) and fast.
- **No GPU detected →** falls back to full-precision (FP16/FP32) _CPU_ inference. Expect **~50× slower** generation and ≈12 – 16 GB RAM usage. A warning is printed at startup.

No manual switches are required – the logic is handled in `app.py`.

### Performance Notes

- First run will download ~8GB of model files
- Initial model loading takes 30-60 seconds
- Subsequent generations are much faster
- GPU required for reasonable performance

**🔬 Tested configuration:** Windows 11 (22H2) • Python 3.10 • NVIDIA RTX 3070 Laptop GPU (8 GB VRAM) • Driver 546.xx • CUDA 12.1 wheels. Other comparable Ampere/ADA GPUs should behave similarly.

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

The Veena TTS model is available in multiple optimized variants for different hardware platforms and use cases. Each variant is specifically optimized for performance, memory efficiency, or platform compatibility.

### 🏆 Available Variants Overview

| Model Variant                  | Size    | Optimization   | Platform        | Use Case               |
| ------------------------------ | ------- | -------------- | --------------- | ---------------------- |
| **maya-research/Veena**        | ~7.6 GB | Original       | NVIDIA GPU      | Development & Research |
| **Prince-1/Veena-Onnx**        | ~7.6 GB | ONNX Runtime   | Cross-platform  | Production Deployment  |
| **hashvibe007/Veena-mlx-4Bit** | ~1.9 GB | MLX 4-bit      | Apple Silicon   | macOS M1/M2/M3         |
| **Prince-1/Veena-Onnx-Int4**   | ~4.1 GB | INT4 Quantized | Cross-platform  | Edge Devices           |
| **Prince-1/Veena-RKllm**       | ~7.6 GB | RKLLM          | Rockchip RK3588 | Embedded Systems       |

---

### 🔥 1. maya-research/Veena (Original)

**🎯 Best for: Development, Research, and High-End GPUs**

- **Architecture**: 3B parameter Llama-based transformer
- **Model Size**: ~7.6 GB (model files)
- **Optimization**: Standard PyTorch with 4-bit NF4 quantization
- **Platform**: NVIDIA GPUs (CUDA 11.8+)
- **Memory**: ~5-6 GB VRAM with quantization

**Key Features:**

- ✅ Original model with full precision
- ✅ 4 distinct voices (Kavya, Agastya, Maitri, Vinaya)
- ✅ 24kHz SNAC neural codec
- ✅ Sub-80ms latency on H100 GPUs
- ✅ Supports Hindi, English, and code-mixed text

**Installation:**

```bash
pip install transformers torch torchaudio snac bitsandbytes
```

**Usage:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "maya-research/Veena",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
```

---

### ⚡ 2. Prince-1/Veena-Onnx

**🎯 Best for: Production Deployment and Cross-Platform Compatibility**

- **Architecture**: ONNX Runtime optimized
- **Model Size**: ~7.6 GB
- **Optimization**: ONNX Runtime with graph optimizations
- **Platform**: Cross-platform (Windows, Linux, macOS)
- **Memory**: Optimized memory usage with ONNX Runtime

**Key Features:**

- ✅ ONNX Runtime optimization for faster inference
- ✅ Cross-platform compatibility
- ✅ Reduced memory footprint
- ✅ Better CPU performance compared to original
- ✅ Production-ready deployment

**Installation:**

```bash
pip install onnxruntime-genai torch torchaudio snac
```

**Usage:**

```python
# Note: Requires ONNX Runtime GenAI for inference
# This variant is optimized for production deployment
```

---

### 🍎 3. hashvibe007/Veena-mlx-4Bit

**🎯 Best for: Apple Silicon (M1/M2/M3) Macs**

- **Architecture**: MLX optimized with 4-bit quantization
- **Model Size**: ~1.9 GB (highly compressed)
- **Optimization**: Apple MLX framework with 4-bit quantization
- **Platform**: macOS with Apple Silicon (M1/M2/M3)
- **Memory**: ~2-3 GB unified memory

**Key Features:**

- ✅ Optimized for Apple Silicon chips
- ✅ 4-bit quantization for minimal memory usage
- ✅ Native macOS integration
- ✅ Fastest inference on Apple Silicon
- ✅ Energy efficient

**Installation:**

```bash
pip install mlx-lm
```

**Usage:**

```python
from mlx_lm import load, generate

model, tokenizer = load("hashvibe007/Veena-mlx-4Bit")

# Generate speech
response = generate(model, tokenizer, prompt=prompt, verbose=True)
```

---

### 🔧 4. Prince-1/Veena-Onnx-Int4

**🎯 Best for: Edge Devices and Resource-Constrained Environments**

- **Architecture**: ONNX Runtime with INT4 quantization
- **Model Size**: ~4.1 GB
- **Optimization**: INT4 quantization for minimal memory usage
- **Platform**: Cross-platform edge deployment
- **Memory**: ~2-3 GB VRAM/RAM

**Key Features:**

- ✅ INT4 quantization for 50% size reduction
- ✅ Edge device optimization
- ✅ Lower memory requirements
- ✅ Maintains good quality despite compression
- ✅ Fast inference on limited hardware

**Installation:**

```bash
pip install onnxruntime-genai torch torchaudio snac
```

**Usage:**

```python
# Optimized for edge deployment with minimal memory usage
# Ideal for embedded systems and IoT devices
```

---

### 🏭 5. Prince-1/Veena-RKllm

**🎯 Best for: Rockchip RK3588 and ARM-based Embedded Systems**

- **Architecture**: RKLLM optimized for Rockchip NPU
- **Model Size**: ~7.6 GB
- **Optimization**: Rockchip Neural Processing Unit (NPU) acceleration
- **Platform**: Rockchip RK3588 (Orange Pi 5, etc.)
- **Memory**: Optimized for ARM architecture

**Key Features:**

- ✅ Native Rockchip RK3588 NPU acceleration
- ✅ ARM architecture optimization
- ✅ Hardware-accelerated inference
- ✅ Embedded system deployment
- ✅ Industrial IoT applications

**Installation:**

```bash
# Requires Rockchip RKLLM toolkit
# Specific to RK3588-based hardware
```

---

### 🚀 Performance Comparison

| Variant       | GPU Latency      | CPU Latency     | Memory Usage | Quality    |
| ------------- | ---------------- | --------------- | ------------ | ---------- |
| **Original**  | ~80ms (H100)     | ~4000ms         | ~5-6 GB      | ⭐⭐⭐⭐⭐ |
| **ONNX**      | ~100ms (H100)    | ~2000ms         | ~4-5 GB      | ⭐⭐⭐⭐⭐ |
| **MLX 4-bit** | ~150ms (M3)      | ~200ms (M3)     | ~2-3 GB      | ⭐⭐⭐⭐   |
| **ONNX INT4** | ~120ms (RTX4090) | ~2500ms         | ~2-3 GB      | ⭐⭐⭐⭐   |
| **RKllm**     | ~200ms (RK3588)  | ~300ms (RK3588) | ~4-5 GB      | ⭐⭐⭐⭐   |

### 🔄 Switching Between Variants

To use a different variant in your application, modify the model name in `app.py`:

```python
# Original model (default)
model = AutoModelForCausalLM.from_pretrained(
    "maya-research/Veena",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

# ONNX variant
model = AutoModelForCausalLM.from_pretrained(
    "Prince-1/Veena-Onnx",
    trust_remote_code=True,
)

# MLX variant (macOS only)
from mlx_lm import load
model, tokenizer = load("hashvibe007/Veena-mlx-4Bit")
```

### 📋 Choosing the Right Variant

**Choose based on your requirements:**

- **🏆 High-end NVIDIA GPU**: Use `maya-research/Veena` (original)
- **🌐 Cross-platform deployment**: Use `Prince-1/Veena-Onnx`
- **🍎 Apple Silicon Mac**: Use `hashvibe007/Veena-mlx-4Bit`
- **📱 Edge devices/IoT**: Use `Prince-1/Veena-Onnx-Int4`
- **🏭 Rockchip RK3588**: Use `Prince-1/Veena-RKllm`

### 📖 Additional Resources

- [🤗 Original Model](https://huggingface.co/maya-research/Veena)
- [⚡ ONNX Version](https://huggingface.co/Prince-1/Veena-Onnx)
- [🍎 MLX Version](https://huggingface.co/hashvibe007/Veena-mlx-4Bit)
- [🔧 INT4 Version](https://huggingface.co/Prince-1/Veena-Onnx-Int4)
- [🏭 RKllm Version](https://huggingface.co/Prince-1/Veena-RKllm)

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
