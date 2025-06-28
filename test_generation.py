#!/usr/bin/env python3
"""
Simple test script to verify Veena TTS generation works correctly
"""

import sys
import os
sys.path.append('.')

from app import generate_speech, load_models, SPEAKERS

def test_basic_generation():
    """Test basic speech generation"""
    print("🧪 Testing Veena TTS generation...")
    
    # Load models
    print("📦 Loading models...")
    load_models()
    print("✅ Models loaded")
    
    # Test text
    test_text = "Hello, this is a test."
    
    # Test with kavya speaker
    print(f"🎭 Testing with speaker: kavya")
    print(f"📝 Text: {test_text}")
    
    # Simple progress function
    def dummy_progress(progress, desc=""):
        print(f"⏳ Progress: {progress:.1%} - {desc}")
    
    # Generate speech
    result = generate_speech(test_text, "kavya", progress=dummy_progress)
    
    if result[0] is None:
        print(f"❌ Generation failed: {result[1]}")
        return False
    else:
        audio_file, status = result
        print(f"✅ Generation successful!")
        print(f"📁 Audio file: {audio_file}")
        print(f"📊 Status: {status}")
        
        # Check if file exists
        if os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file)
            print(f"📏 File size: {file_size} bytes")
            
            if file_size > 0:
                print("✅ File created successfully with content!")
                return True
            else:
                print("❌ File created but is empty!")
                return False
        else:
            print("❌ File not found!")
            return False

if __name__ == "__main__":
    try:
        success = test_basic_generation()
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n💥 Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        sys.exit(1) 