#!/usr/bin/env python3
"""
Veena TTS Gradio App Launcher
Simple launcher script with setup checks and error handling
"""

# -----------------------------
# Standard Library Imports
# -----------------------------
import sys
import subprocess
import os
from pathlib import Path

# -----------------------------
# Constants
# -----------------------------
# Default CUDA wheel index used by PyTorch >=2.2 (cu121).  Adjust here for
# future CUDA versions if necessary.
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
# We deliberately keep versions un-pinned so that the most recent compatible
# wheels are pulled in.
PYTORCH_PACKAGES = [
    "torch",        # core
    "torchaudio",   # needed by requirements.txt
]

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 6:
                print("⚠️  Warning: Less than 6GB GPU memory may cause issues")
                return "warning"
            return True
        else:
            print("⚠️  Warning: No GPU detected. Performance will be very slow.")
            return "cpu"
    except ImportError:
        print("⚠️  PyTorch not installed. Cannot check GPU status.")
        return "unknown"

def detect_nvidia_gpu():
    """Return True if `nvidia-smi` detects at least one CUDA-capable GPU."""
    try:
        subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

def ensure_cuda_torch():
    """Install/upgrade PyTorch with CUDA support when an NVIDIA GPU is present.

    If PyTorch is missing *or* is the CPU-only build, automatically installs the
    latest wheels that include CUDA 12.1 runtimes using the official PyTorch
    index.  Re-imports `torch` afterwards so the caller can continue using it.
    """

    gpu_present = detect_nvidia_gpu()

    try:
        import torch  # noqa: F401
        has_cuda = torch.cuda.is_available()
    except ImportError:
        torch = None  # type: ignore
        has_cuda = False

    if not gpu_present:
        # Nothing extra to do when no NVIDIA GPU is available.
        return

    if has_cuda:
        # All good – we already have a CUDA-enabled build.
        return

    print("⚙️  NVIDIA GPU detected but current PyTorch build lacks CUDA support.")
    print("   Installing CUDA wheels – this may take a few minutes…")

    # Construct pip install command
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--index-url",
        PYTORCH_CUDA_INDEX,
        *PYTORCH_PACKAGES,
    ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"❌  Failed to install CUDA-enabled PyTorch: {e}")
        print("   Falling back to existing (CPU) build – the app will run on CPU and be slow.")
        return

    # Re-import torch so callers get the CUDA build
    import importlib
    globals()["torch"] = importlib.reload(__import__("torch"))

    if torch.cuda.is_available():
        print("✅  PyTorch now has CUDA support enabled.")
    else:
        print("⚠️  PyTorch installation completed but CUDA is still unavailable. Continuing with CPU fallback.")

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
        
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_app_file():
    """Check if main app file exists"""
    app_file = Path("app.py")
    if not app_file.exists():
        print("❌ Error: app.py not found")
        return False
    return True

def launch_app():
    """Launch the Gradio app"""
    print("🚀 Starting Veena TTS Gradio App...")
    print("   This may take 30-60 seconds for first-time model download...")
    print("   Once started, open http://localhost:8000 in your browser")
    print("   Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Import and run the app
        from app import demo
        demo.launch()
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all requirements are installed")
        print("2. Check if you have enough GPU memory")
        print("3. Try running: python app.py directly")

def main():
    """Main launcher function"""
    print("🎵 Veena TTS - Gradio App Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if app file exists
    if not check_app_file():
        sys.exit(1)
    
    # Ensure we have a CUDA-enabled torch build before we do anything else
    ensure_cuda_torch()

    # Re-run the GPU check **after** potential installation/upgrade
    gpu_status = check_gpu()
    
    # Ask user if they want to install requirements
    try:
        install_deps = input("\n📦 Install/update requirements? (y/n): ").lower().strip()
        if install_deps in ['y', 'yes', '']:
            if not install_requirements():
                print("❌ Failed to install requirements. Please install manually:")
                print("   pip install -r requirements.txt")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        sys.exit(0)
    
    # Show GPU warning if needed
    if gpu_status == "cpu":
        try:
            continue_cpu = input("\n⚠️  Continue without GPU? This will be very slow. (y/n): ").lower().strip()
            if continue_cpu not in ['y', 'yes']:
                print("👋 Exiting. Please ensure GPU drivers and CUDA are installed.")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            sys.exit(0)
    
    # Launch the app
    print("\n" + "="*50)
    launch_app()

if __name__ == "__main__":
    main() 