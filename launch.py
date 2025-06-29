#!/usr/bin/env python3
"""
Veena TTS Gradio App Launcher
=============================================================================
Enhanced launcher with automatic environment setup and dependency management
"""

import os
import sys
import subprocess
import importlib.util
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (very slow)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - will install dependencies")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    requirements = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "transformers>=4.35.0",
        "huggingface_hub[hf_xet]>=0.17.0",
        "gradio>=4.0.0",
        "snac",
        "bitsandbytes>=0.41.0",
        "soundfile>=0.12.1",
        "numpy>=1.21.0",
        "accelerate>=0.20.0",
        "scipy>=1.9.0",
        "librosa>=0.9.0",
        "sentencepiece>=0.1.99"
    ]
    
    for requirement in requirements:
        try:
            print(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {requirement}: {e}")
            return False
    
    print("✅ All dependencies installed successfully!")
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("gradio", "Gradio"),
        ("snac", "SNAC Codec"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy")
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append((package, name))
            print(f"❌ {name} not found")
        else:
            print(f"✅ {name} available")
    
    return len(missing_packages) == 0, missing_packages

def show_system_info():
    """Display system information"""
    print("🖥️  System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   CPU: {platform.processor()}")
    
    # Memory info (if available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
    except ImportError:
        print("   RAM: Unable to detect (install psutil for memory info)")

def main():
    """Main launcher function"""
    print("🎵 Veena TTS - Gradio App Launcher")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Show system info
    show_system_info()
    
    # Check CUDA
    has_cuda = check_cuda()
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\n❌ Missing dependencies: {[pkg[1] for pkg in missing]}")
        install_deps = input("\n📦 Install missing dependencies? (y/n): ").lower().strip()
        
        if install_deps in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually.")
                sys.exit(1)
        else:
            print("❌ Cannot continue without required dependencies.")
            sys.exit(1)
    
    # Launch the app
    print("\n🚀 Launching Veena TTS Gradio App...")
    print("   - Model: maya-research/veena-tts")
    print("   - Interface: http://localhost:8000")
    print("   - Use Ctrl+C to stop the server")
    
    if not has_cuda:
        print("\n⚠️  WARNING: Running on CPU will be very slow!")
        print("   Consider using a GPU-enabled environment for better performance.")
        proceed = input("   Continue anyway? (y/n): ").lower().strip()
        if proceed not in ['y', 'yes']:
            print("Cancelled.")
            sys.exit(0)
    
    print("\n" + "=" * 50)
    
    # Import and launch the app
    try:
        from app import demo
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=False,
            show_error=True,
            inbrowser=True
        )
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("   Make sure app.py is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 