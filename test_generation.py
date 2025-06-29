#!/usr/bin/env python3
"""
Comprehensive test script for Veena TTS generation
=============================================================================
Tests all functionality from the Jupyter notebook including:
- Model loading and initialization
- Speaker token verification
- Audio generation with different speakers
- Example generation from notebook
- File output verification
"""

import sys
import os
import tempfile
import soundfile as sf
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        import gradio as gr
        print(f"✅ Gradio: {gr.__version__}")
        
        from snac import SNAC
        print("✅ SNAC codec")
        
        import soundfile as sf
        print("✅ SoundFile")
        
        import numpy as np  
        print("✅ NumPy")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n🧪 Testing model loading...")
    
    try:
        from app import load_models, model, tokenizer, snac_model
        
        print("📦 Loading models...")
        load_models()
        
        if model is None:
            print("❌ Model failed to load")
            return False
        
        if tokenizer is None:
            print("❌ Tokenizer failed to load")
            return False
            
        if snac_model is None:
            print("❌ SNAC model failed to load")
            return False
        
        print("✅ All models loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_speaker_tokens():
    """Test speaker token functionality"""
    print("\n🧪 Testing speaker tokens...")
    
    try:
        from app import test_speaker_tokens
        
        result = test_speaker_tokens()
        print(result)
        
        # Check if any working speakers were found
        if "✅ Good tokenization: 0/" in result:
            print("❌ No working speakers found")
            return False
        
        print("✅ Speaker token test completed")
        return True
        
    except Exception as e:
        print(f"❌ Speaker token test error: {e}")
        return False

def test_basic_generation():
    """Test basic speech generation"""
    print("\n🧪 Testing basic speech generation...")
    
    try:
        from app import generate_speech_simple, load_models, speakers
        
        # Load models
        load_models()
        
        # Test text
        test_text = "Hello, this is a test of the Veena text to speech system."
        
        # Test with each core speaker
        for speaker in speakers[:2]:  # Test first 2 speakers only
            print(f"🎭 Testing speaker: {speaker}")
            
            try:
                audio = generate_speech_simple(test_text, speaker)
                
                if audio is None:
                    print(f"❌ Generation failed for {speaker}")
                    return False
                
                # Check audio properties
                if len(audio) == 0:
                    print(f"❌ Empty audio generated for {speaker}")
                    return False
                
                # Create temporary file to verify
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                sf.write(temp_filename, audio, 24000)
                
                # Verify file
                if not os.path.exists(temp_filename):
                    print(f"❌ Output file not created for {speaker}")
                    return False
                
                file_size = os.path.getsize(temp_filename)
                if file_size == 0:
                    print(f"❌ Empty output file for {speaker}")
                    return False
                
                print(f"✅ Speaker {speaker}: {file_size} bytes, {len(audio)} samples")
                
                # Clean up
                os.remove(temp_filename)
                
            except Exception as e:
                print(f"❌ Error with speaker {speaker}: {e}")
                return False
        
        print("✅ Basic generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic generation test error: {e}")
        return False

def test_gradio_integration():
    """Test Gradio integration"""
    print("\n🧪 Testing Gradio integration...")
    
    try:
        from app import generate_speech, load_models
        
        # Load models
        load_models()
        
        # Test with Gradio wrapper
        def dummy_progress(progress, desc=""):
            print(f"⏳ Progress: {progress:.1%} - {desc}")
        
        # Test generation
        result = generate_speech(
            "Testing Gradio integration.", 
            "kavya", 
            progress=dummy_progress
        )
        
        if result[0] is None:
            print(f"❌ Gradio integration test failed: {result[1]}")
            return False
        
        audio_file, status = result
        
        # Verify file exists
        if not os.path.exists(audio_file):
            print("❌ Generated audio file not found")
            return False
        
        file_size = os.path.getsize(audio_file)
        print(f"✅ Gradio integration test passed: {file_size} bytes")
        
        # Clean up
        os.remove(audio_file)
        return True
        
    except Exception as e:
        print(f"❌ Gradio integration test error: {e}")
        return False

def test_examples_generation():
    """Test examples generation from Jupyter notebook"""
    print("\n🧪 Testing examples generation...")
    
    try:
        from app import generate_examples
        
        results = generate_examples()
        
        if not results:
            print("❌ No examples generated")
            return False
        
        success_count = 0
        total_count = len(results)
        
        for name, filepath, status in results:
            if filepath and os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ {name}: {file_size} bytes")
                success_count += 1
                # Clean up
                os.remove(filepath)
            else:
                print(f"❌ {name}: {status}")
        
        print(f"📊 Examples generation: {success_count}/{total_count} successful")
        
        if success_count == 0:
            print("❌ No examples generated successfully")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Examples generation test error: {e}")
        return False

def test_multilingual():
    """Test multilingual capabilities"""
    print("\n🧪 Testing multilingual capabilities...")
    
    try:
        from app import generate_speech_simple, load_models
        
        load_models()
        
        test_cases = [
            ("Hindi", "आज मैंने एक नई तकनीक सीखी।", "kavya"),
            ("English", "Today I learned a new technology.", "agastya"),
            ("Code-mixed", "मैं बहुत excited हूं!", "maitri")
        ]
        
        for lang, text, speaker in test_cases:
            try:
                print(f"🌍 Testing {lang} with {speaker}")
                audio = generate_speech_simple(text, speaker)
                
                if audio is None or len(audio) == 0:
                    print(f"❌ {lang} generation failed")
                    return False
                
                print(f"✅ {lang}: {len(audio)} samples")
                
            except Exception as e:
                print(f"❌ {lang} test error: {e}")
                return False
        
        print("✅ Multilingual test passed")
        return True
        
    except Exception as e:
        print(f"❌ Multilingual test error: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🧪 Veena TTS Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Loading", test_model_loading),
        ("Speaker Tokens", test_speaker_tokens),
        ("Basic Generation", test_basic_generation),
        ("Gradio Integration", test_gradio_integration),
        ("Examples Generation", test_examples_generation),
        ("Multilingual", test_multilingual),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Veena TTS setup is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1) 