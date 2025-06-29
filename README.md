# 🎵 Veena TTS - Enhanced Gradio Web Interface

![Veena TTS Interface](./img/image.png)

> **✨ RECENTLY UPDATED**: Fixed all major issues including Windows compatibility, correct model loading, SNAC decoding, and comprehensive example generation from Jupyter notebook!

A beautiful and intuitive web interface for the **Veena Text-to-Speech model** developed by Maya Research. This enhanced Gradio application provides an easy-to-use interface for generating high-quality speech in Hindi, English, and code-mixed text with full Windows support and automatic setup.

## 🆕 **Latest Updates & Fixes**

### ✅ **Major Fixes Applied**

- **🔧 Fixed Model Name**: Corrected from `maya-research/veena` to `maya-research/veena-tts`
- **🔧 Fixed SNAC Decoding**: Corrected token de-interleaving order for proper audio generation
- **🔧 Fixed Speaker Tokens**: Updated to use `<|spk_{speaker}|>` format from Jupyter notebook
- **🔧 Windows Support**: Resolved sentencepiece compilation issues on Windows
- **🔧 Port Handling**: Added automatic port detection to avoid conflicts
- **🔧 Enhanced Error Handling**: Better error messages and troubleshooting guidance

### 🚀 **New Features**

- **📚 Jupyter Notebook Integration**: Complete example generation from the official notebook
- **🪟 Windows Setup Script**: Automated Windows-specific installation (`setup_windows.py`)
- **🧪 Comprehensive Testing**: Full test suite with dependency checking (`test_generation.py`)
- **⚙️ Enhanced Launcher**: Improved `launch.py` with auto-dependency installation
- **🎭 Example Generation**: Generate all official examples (Hindi, English, Code-mixed, Bollywood)
- **🔧 Debug Tools**: Speaker token verification and audio testing tools

## ✨ **Core Features**

- **🌍 Multi-lingual Support**: Hindi, English, and code-mixed text
- **🎭 Multiple Voices**: Choose from 4 core speakers + 7 additional voices
- **🎛️ Advanced Controls**: Adjust temperature and top-p for speech variation
- **📱 Responsive UI**: Modern, mobile-friendly interface with enhanced styling
- **⚡ Real-time Generation**: Quick speech synthesis with detailed progress tracking
- **🎧 Audio Playback**: Instant playback and download capabilities
- **📝 Quick Examples**: Pre-loaded example texts in different languages
- **🔧 Built-in Testing**: Comprehensive debugging and testing tools

## 🚀 **Quick Start**

### **Method 1: Windows Users (Recommended)**

```bash
# Clone or download the project
git clone <your-repo-url>
cd veena

# Run Windows setup (handles all dependencies)
python setup_windows.py

# Launch the app
python launch.py
```

### **Method 2: Universal Installation**

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### **Method 3: Comprehensive Testing**

```bash
# Run full test suite
python test_generation.py

# If tests pass, launch the app
python app.py
```

## 🖥️ **System Requirements**

### **Minimum Requirements**

| Component   | Requirement                 |
| ----------- | --------------------------- |
| **Python**  | 3.8 - 3.13 (tested on 3.13) |
| **OS**      | Windows 10/11, Linux, macOS |
| **RAM**     | 8GB+ (16GB recommended)     |
| **Storage** | 10GB free space             |

### **Recommended for GPU**

| Component   | Specification           |
| ----------- | ----------------------- |
| **GPU**     | NVIDIA RTX 30/40 series |
| **VRAM**    | 6GB+ (8GB+ recommended) |
| **CUDA**    | 11.8+ or 12.1+          |
| **Drivers** | Latest NVIDIA drivers   |

### **CPU Fallback**

- **Processor**: Modern multi-core CPU
- **RAM**: 16GB+ (model uses ~12GB)
- **Note**: ⚠️ CPU inference is ~50x slower than GPU

## 📦 **Installation Guide**

### **Windows Installation**

1. **Ensure Python 3.8+ is installed**
2. **Run the Windows setup script**:

   ```bash
   python setup_windows.py
   ```

   This script will:

   - Install all required dependencies
   - Handle sentencepiece compilation issues
   - Test model loading
   - Provide troubleshooting guidance

3. **Launch the application**:
   ```bash
   python launch.py
   ```

### **Linux/macOS Installation**

```bash
# Install PyTorch with CUDA support (if you have GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

### **Manual Dependency Installation**

```bash
# Core dependencies
pip install torch>=2.0.0 torchaudio>=2.0.0
pip install transformers>=4.35.0 gradio>=4.0.0
pip install snac bitsandbytes>=0.41.0
pip install soundfile numpy accelerate scipy librosa

# sentencepiece is optional - auto-installed by transformers if needed
```

## 🎯 **Usage Guide**

### **Basic Usage**

1. **Launch the app**:

   ```bash
   python app.py
   # or
   python launch.py
   ```

2. **Open browser**: Navigate to `http://localhost:8000`

   - If port 8000 is busy, the app will automatically find an available port

3. **Enter text**: Type Hindi, English, or code-mixed text

4. **Choose voice**: Select from available speakers:

   - **Core speakers**: kavya, agastya, maitri, vinaya
   - **Extended speakers**: apsara, charu, ishana, kyra, mohini, varun, soumya

5. **Generate speech**: Click "🎵 Generate Speech"

### **Advanced Features**

#### **🎭 Example Generation**

Click "🎭 Generate All Examples (Jupyter Notebook)" to create:

- Hindi literature examples
- English professional text
- Code-mixed (Hinglish) conversations
- Bollywood dialogue samples

#### **🔧 Debug Tools**

- **Test Speaker Tokens**: Verify all speaker tokens are properly recognized
- **Test All Speakers Audio**: Generate test audio for all available voices
- **Speaker Selection Debug**: Real-time speaker selection verification

#### **⚙️ Advanced Settings**

- **Temperature** (0.1-1.0): Controls speech creativity/variation
- **Top-p** (0.1-1.0): Controls focus/coherence
- **Real-time progress**: Detailed generation progress tracking

## 🔧 **Technical Details**

### **Model Information**

- **Model**: `maya-research/veena-tts` (corrected from previous versions)
- **Architecture**: 3B parameter Llama-based transformer
- **Audio Codec**: 24kHz SNAC neural codec
- **Quantization**: 4-bit NF4 for GPU efficiency
- **Latency**: Sub-80ms on H100, ~200ms on RTX 4090

### **Performance Optimizations**

- **GPU**: Automatic 4-bit quantization with bitsandbytes
- **CPU**: Full-precision fallback with memory optimization
- **Memory**: ~5GB VRAM (GPU) or ~12GB RAM (CPU)
- **Caching**: Automatic model caching for faster subsequent loads

### **File Structure**

```
veena/
├── app.py              # Main Gradio application (✅ FIXED)
├── launch.py           # Enhanced launcher with auto-setup
├── setup_windows.py    # Windows-specific setup script (🆕 NEW)
├── test_generation.py  # Comprehensive test suite (✅ ENHANCED)
├── requirements.txt    # Dependencies (✅ UPDATED)
├── README.md          # This file (✅ UPDATED)
└── img/               # Interface screenshots
```

## 🔍 **Troubleshooting**

### **Common Issues & Solutions**

#### **🔧 Port Already in Use**

```bash
# Error: [Errno 10048] error while attempting to bind on address
# Solution: App will auto-find available port, or set manually:
GRADIO_SERVER_PORT=7860 python app.py
```

#### **🔧 sentencepiece Installation Failed (Windows)**

```bash
# Run the Windows setup script:
python setup_windows.py

# Or install manually:
pip install sentencepiece --only-binary=:all:
```

#### **🔧 CUDA Out of Memory**

```bash
# Check GPU memory:
nvidia-smi

# Clear cache:
python -c "import torch; torch.cuda.empty_cache()"

# Use CPU if needed:
python app.py  # Will auto-detect and fallback to CPU
```

#### **🔧 Model Download Issues**

```bash
# Check internet connection and disk space
# Model files are ~5GB total

# Test download:
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('maya-research/veena-tts', trust_remote_code=True)"
```

#### **🔧 Audio Generation Fails**

```bash
# Run comprehensive tests:
python test_generation.py

# Check speaker tokens:
# Use the "Test Speaker Tokens" button in the interface
```

### **Debugging Commands**

```bash
# Check Python version
python --version

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check dependencies
python -c "import transformers, gradio, snac, soundfile; print('All imports OK')"

# Run test suite
python test_generation.py
```

## 🎭 **Speaker Voices**

### **Core Speakers** (From Jupyter Notebook)

- **🎭 Kavya**: Expressive female voice - Best for dramatic content
- **🎯 Agastya**: Sage male voice - Ideal for professional/formal content
- **💫 Maitri**: Friendly female voice - Perfect for conversational tone
- **🎪 Vinaya**: Warm male voice - Great for storytelling

### **Extended Speakers** (Additional Options)

- **✨ Apsara**: Celestial female voice
- **🌸 Charu**: Graceful female voice
- **🏔️ Ishana**: Noble male voice
- **🌟 Kyra**: Bright female voice
- **🎨 Mohini**: Enchanting female voice
- **🌊 Varun**: Strong male voice
- **🌙 Soumya**: Gentle unisex voice

## 📚 **Example Texts**

### **Hindi Literature**

```
बचपन की यादें हमेशा दिल के सबसे करीब होती हैं, खासकर जब वे गर्मियों की छुट्टियों से जुड़ी हों।
```

### **English Professional**

```
The rise of generative AI is transforming industries by enabling faster content creation and automation.
```

### **Code-mixed (Hinglish)**

```
अगर आपने अभी तक client को final proposal नहीं भेजा है, then please do it by evening.
```

### **Bollywood Dialogue**

```
कभी-कभी ज़िन्दगी हमें वहाँ ले आती है जहाँ हमें अपने दिल की नहीं, दूसरों की खुशी के लिए जीना पड़ता है।
```

## 🌐 **Deployment Options**

### **Local Development**

```bash
python app.py
# Runs on http://localhost:8000
```

### **Custom Port**

```bash
# Set custom port via environment variable
GRADIO_SERVER_PORT=7860 python app.py

# Or edit app.py:
demo.launch(server_port=7860)
```

### **Public Sharing**

```bash
# Enable public sharing
demo.launch(share=True)
```

### **Docker Deployment**

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

## 🧪 **Testing & Validation**

### **Run Test Suite**

```bash
python test_generation.py
```

**Test Coverage**:

- ✅ Import validation
- ✅ Model loading
- ✅ Speaker token verification
- ✅ Basic audio generation
- ✅ Gradio integration
- ✅ Multilingual support
- ✅ Example generation

### **Performance Benchmarks**

| Hardware   | Model Load | Generation | Quality    |
| ---------- | ---------- | ---------- | ---------- |
| RTX 4090   | ~30s       | ~200ms     | ⭐⭐⭐⭐⭐ |
| RTX 3070   | ~45s       | ~400ms     | ⭐⭐⭐⭐⭐ |
| RTX 3060   | ~60s       | ~600ms     | ⭐⭐⭐⭐⭐ |
| CPU (16GB) | ~90s       | ~10s       | ⭐⭐⭐⭐⭐ |

## 📝 **Model Variants**

This application supports multiple model variants:

### **🏆 maya-research/veena-tts** (Default - ✅ FIXED)

- **Size**: ~5GB
- **Platform**: NVIDIA GPU + CPU fallback
- **Optimization**: 4-bit quantization
- **Best for**: General use with GPU

### **Other Variants** (Available)

- **Prince-1/Veena-Onnx**: Cross-platform ONNX
- **hashvibe007/Veena-mlx-4Bit**: Apple Silicon optimized
- **Prince-1/Veena-Onnx-Int4**: Edge devices
- **Prince-1/Veena-RKllm**: Rockchip RK3588

## 🤝 **Contributing**

We welcome contributions! Here's how to help:

### **Reporting Issues**

1. Run `python test_generation.py` first
2. Include system information (OS, Python version, GPU)
3. Provide full error messages
4. Share example text that fails

### **Feature Requests**

- Additional language support
- New speaker voices
- Performance optimizations
- UI/UX improvements

### **Development Setup**

```bash
git clone <your-repo>
cd veena
python setup_windows.py  # or pip install -r requirements.txt
python test_generation.py
```

## 🔗 **Links & Resources**

### **Official Resources**

- [🤗 Veena TTS Model](https://huggingface.co/maya-research/veena-tts)
- [📚 Model Documentation](https://huggingface.co/maya-research/veena-tts/blob/main/README.md)
- [💬 Model Discussions](https://huggingface.co/maya-research/veena-tts/discussions)

### **Community Resources**

- [📖 Jupyter Notebook Examples](https://huggingface.co/maya-research/veena-tts/blob/main/examples.ipynb)
- [🎯 Performance Benchmarks](https://huggingface.co/maya-research/veena-tts/blob/main/benchmarks.md)
- [🔧 Advanced Configuration](https://huggingface.co/maya-research/veena-tts/blob/main/config.md)

## 📄 **License**

This project follows the Apache 2.0 license of the original Veena model.

## 🙏 **Acknowledgments**

- **Maya Research** - For developing the amazing Veena TTS model
- **Hugging Face** - For hosting and model infrastructure
- **Gradio Team** - For the excellent web interface framework
- **Windows Community** - For testing and feedback on Windows compatibility
- **Contributors** - For bug reports, feature requests, and improvements

## 📊 **Changelog**

### **v2.0.0 - Enhanced Windows Support** (Latest)

- ✅ Fixed correct model name (`maya-research/veena-tts`)
- ✅ Fixed SNAC token decoding order
- ✅ Fixed speaker token format (`<|spk_{speaker}|>`)
- ✅ Added Windows setup script
- ✅ Enhanced error handling and troubleshooting
- ✅ Added comprehensive test suite
- ✅ Added Jupyter notebook example generation
- ✅ Fixed port conflict handling
- ✅ Improved documentation and setup instructions

### **v1.0.0 - Initial Release**

- Basic Gradio interface
- GPU/CPU support
- Multiple speaker voices
- Multi-language support

---

**🎵 Made with ❤️ for the Indian Language AI Community**

> **Need Help?** Run `python test_generation.py` to diagnose issues, or check the troubleshooting section above!
