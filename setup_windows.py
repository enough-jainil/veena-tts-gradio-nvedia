#!/usr/bin/env python3
"""
Windows-specific setup script for Veena TTS
=============================================================================
Handles common Windows installation issues, especially with sentencepiece
"""

import subprocess
import sys
import os
import platform

def install_with_pip(package, extra_args=None):
    """Install package with pip, with optional extra arguments"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(package)
    
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False

def install_sentencepiece_windows():
    """Try multiple methods to install sentencepiece on Windows"""
    print("🪟 Installing sentencepiece on Windows...")
    
    # Method 1: Try pre-built wheel
    print("🔧 Method 1: Trying pre-built wheel...")
    if install_with_pip("sentencepiece", ["--only-binary=:all:"]):
        print("✅ Installed sentencepiece using pre-built wheel")
        return True
    
    # Method 2: Try specific version
    print("🔧 Method 2: Trying specific version...")
    if install_with_pip("sentencepiece==0.1.99"):
        print("✅ Installed sentencepiece version 0.1.99")
        return True
    
    # Method 3: Try conda-forge if conda is available
    print("🔧 Method 3: Trying conda-forge...")
    try:
        subprocess.check_call(["conda", "install", "-c", "conda-forge", "sentencepiece", "-y"])
        print("✅ Installed sentencepiece using conda-forge")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Method 4: Skip sentencepiece and warn user
    print("⚠️  Could not install sentencepiece automatically")
    print("   The app may still work as transformers can handle tokenization")
    return False

def install_build_tools():
    """Install Microsoft C++ Build Tools if needed"""
    print("🔨 Installing Microsoft C++ Build Tools...")
    print("   This is needed for compiling packages like sentencepiece")
    
    # Try to install via chocolatey if available
    try:
        subprocess.check_call(["choco", "install", "visualstudio2022buildtools", "--package-parameters", "--add Microsoft.VisualStudio.Workload.VCTools"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("⚠️  Could not install build tools automatically")
    print("   Please install Visual Studio Build Tools manually if needed:")
    print("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    return False

def main():
    """Main setup function for Windows"""
    print("🪟 Veena TTS - Windows Setup")
    print("=" * 40)
    
    if platform.system() != "Windows":
        print("❌ This script is for Windows only!")
        sys.exit(1)
    
    print(f"🐍 Python version: {sys.version}")
    print(f"🖥️  Platform: {platform.platform()}")
    
    # Install basic requirements first
    print("\n📦 Installing basic requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing basic requirements: {e}")
        print("   Continuing with sentencepiece installation...")
    
    # Try to install sentencepiece
    print("\n🔤 Installing sentencepiece...")
    if not install_sentencepiece_windows():
        print("\n🤔 Sentencepiece installation failed. Let's test if the app works anyway...")
        
        # Test tokenizer loading
        try:
            print("🧪 Testing tokenizer loading...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)
            print("✅ Tokenizer works without explicit sentencepiece!")
        except Exception as e:
            print(f"❌ Tokenizer test failed: {e}")
            
            # Offer to install build tools
            install_tools = input("\n🔨 Install Microsoft C++ Build Tools? This may help. (y/n): ").lower().strip()
            if install_tools in ['y', 'yes']:
                install_build_tools()
                
                # Try sentencepiece again
                print("\n🔄 Retrying sentencepiece installation...")
                install_sentencepiece_windows()
    
    print("\n🚀 Setup complete! You can now run:")
    print("   python app.py")
    print("   or")
    print("   python launch.py")

if __name__ == "__main__":
    main() 