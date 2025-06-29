import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
import soundfile as sf
import numpy as np
import tempfile
import os
import atexit
from datetime import datetime
import time

# Global variables for model components
model = None
tokenizer = None
snac_model = None

# List to keep track of temporary files for cleanup
temp_files = []

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
        
    try:
        import gradio
    except ImportError:
        missing_deps.append("gradio")
        
    try:
        from snac import SNAC
    except ImportError:
        missing_deps.append("snac")
        
    try:
        import soundfile
    except ImportError:
        missing_deps.append("soundfile")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def cleanup_temp_files():
    """Clean up temporary audio files"""
    global temp_files
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
    temp_files.clear()

def periodic_cleanup():
    """Clean up old temporary files (keep only the last 5)"""
    global temp_files
    if len(temp_files) > 5:
        # Remove older files
        for file_path in temp_files[:-5]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        temp_files = temp_files[-5:]

# Register cleanup function to run on exit
atexit.register(cleanup_temp_files)

# Control token IDs (fixed for Veena)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

# Available speaker voices from the Jupyter notebook
speakers = ["kavya", "agastya", "maitri", "vinaya"]

# Extended speakers from the model (keeping the fancy display names)
SPEAKERS = {
    "kavya": "🎭 Kavya - Expressive female voice",
    "agastya": "🎯 Agastya - Sage male voice", 
    "maitri": "💫 Maitri - Friendly female voice",
    "vinaya": "🎪 Vinaya - Warm male voice",
    # Additional speakers (if available in the model)
    "apsara": "✨ Apsara - Celestial female voice",
    "charu": "🌸 Charu - Graceful female voice",
    "ishana": "🏔️ Ishana - Noble male voice",
    "kyra": "🌟 Kyra - Bright female voice",
    "mohini": "🎨 Mohini - Enchanting female voice",
    "varun": "🌊 Varun - Strong male voice",
    "soumya": "🌙 Soumya - Gentle unisex voice"
}

def load_models():
    """Load the Veena TTS model and SNAC decoder"""
    global model, tokenizer, snac_model
    
    if model is None:
        print("Loading Veena TTS model...")
        
        try:
            has_gpu = torch.cuda.is_available()

            if has_gpu:
                print("✅ CUDA available, loading quantized model...")
                # Model configuration for fast 4-bit GPU inference
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    "maya-research/veena-tts",
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                # Move SNAC to GPU for faster decode
                snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda()
            else:
                print("⚠️  No CUDA - loading CPU model (will be slow)...")

                # Load the full-precision model on CPU (bitsandbytes requires CUDA)
                model = AutoModelForCausalLM.from_pretrained(
                    "maya-research/veena-tts",
                    trust_remote_code=True,
                )

                snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

            print("📝 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check your internet connection")
            print("2. Make sure you have enough disk space")
            print("3. Try running: python setup_windows.py")
            raise e

def decode_snac_tokens(snac_tokens, snac_model):
    """De-interleave and decode SNAC tokens to audio - FIXED VERSION from Jupyter notebook"""
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None

    # De-interleave tokens into 3 hierarchical levels (CORRECTED ORDER)
    codes_lvl = [[] for _ in range(3)]
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    for i in range(0, len(snac_tokens), 7):
        # Level 0: Coarse (1 token)
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        # Level 1: Medium (2 tokens) - FIXED ORDER
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+2] - llm_codebook_offsets[4])
        # Level 2: Fine (4 tokens) - FIXED ORDER  
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+4] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])

    # Convert to tensors for SNAC decoder
    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_model.device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            raise ValueError("Invalid SNAC token values")
        hierarchical_codes.append(tensor)

    # Decode with SNAC
    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)
    return audio_hat.squeeze().cpu().numpy()

def generate_speech_simple(text, speaker="kavya", temperature=0.4, top_p=0.9):
    """Generate speech from text using the specified speaker voice - Simple version from Jupyter notebook"""
    # Prepare input with speaker token
    prompt = f"<|spk_{speaker}|> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # Construct the full sequence
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]
    input_ids = torch.tensor([input_tokens], device=model.device)

    # Generate audio tokens
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
        )

    # Extract SNAC tokens
    generated_ids = output[0][len(input_tokens):].tolist()
    snac_tokens = [token_id for token_id in generated_ids if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)]

    if not snac_tokens:
        raise ValueError("No audio tokens were generated.")

    # Decode audio from SNAC tokens
    audio = decode_snac_tokens(snac_tokens, snac_model)
    return audio

def generate_examples():
    """Generate example audio files like in the Jupyter notebook"""
    load_models()
    
    examples = [
        ("Hindi", "बचपन की यादें हमेशा दिल के सबसे करीब होती हैं, खासकर जब वे गर्मियों की छुट्टियों से जुड़ी हों।", "kavya"),
        ("English", "The rise of generative AI is transforming industries by enabling faster content creation and automation.", "agastya"),
        ("Code-mixed", "अगर आपने अभी तक client को final proposal नहीं भेजा है, then please do it by evening.", "maitri"),
        ("Formal", "According to the latest report, sustainable energy initiatives have gained significant momentum across urban regions.", "vinaya"),
        ("Bollywood Hindi", "कभी-कभी ज़िन्दगी हमें वहाँ ले आती है जहाँ हमें अपने दिल की नहीं, दूसरों की खुशी के लिए जीना पड़ता है।", "kavya"),
        ("Bollywood English", "They said love knows no borders, no religion, no boundaries — but the world isn't always ready to accept that truth.", "agastya")
    ]
    
    results = []
    
    for name, text, speaker in examples:
        try:
            print(f"Generating {name} with {speaker}...")
            audio = generate_speech_simple(text, speaker=speaker)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"veena_{name.lower().replace(' ', '_')}_") as temp_file:
                temp_filename = temp_file.name
            
            sf.write(temp_filename, audio, 24000)
            temp_files.append(temp_filename)
            
            results.append((name, temp_filename, f"✅ Generated {name} successfully"))
            print(f"Generated '{name}' -> {temp_filename}")
            
        except Exception as e:
            results.append((name, None, f"❌ Error generating {name}: {str(e)}"))
            print(f"Failed to generate {name}: {e}")
    
    return results

def test_speaker_tokens():
    """Test function to verify speaker tokens are recognized"""
    load_models()
    
    # All speakers from SPEAKERS dictionary (matches special_tokens_map.json)
    all_speakers = list(SPEAKERS.keys())
    
    test_results = [f"🔍 TESTING ALL {len(all_speakers)} OFFICIAL SPEAKER TOKENS:\n"]
    
    working_speakers = []
    broken_speakers = []
    
    for speaker in all_speakers:
        token = f"<|spk_{speaker}|>"  # Fixed format
        try:
            tokens = tokenizer.encode(token, add_special_tokens=False)
            decoded = tokenizer.decode(tokens)
            
            # Check if it's properly tokenized (should be a single special token)
            if len(tokens) == 1 and decoded.strip() == token:
                status = "✅ GOOD TOKENIZATION"
                working_speakers.append(speaker)
            else:
                status = "⚠️ POOR TOKENIZATION" 
                broken_speakers.append(speaker)
                
            test_results.append(f"{status} {speaker}: {token} -> tokens: {tokens} -> decoded: '{decoded}'")
        except Exception as e:
            test_results.append(f"❌ ERROR {speaker}: {token} -> ERROR: {str(e)}")
            broken_speakers.append(speaker)
    
    test_results.extend([
        f"\n📊 TOKENIZATION SUMMARY:",
        f"✅ Good tokenization: {len(working_speakers)}/{len(all_speakers)} ({', '.join(working_speakers)})",
        f"❌ Poor tokenization: {len(broken_speakers)}/{len(all_speakers)} ({', '.join(broken_speakers)})",
        f"\n💡 All {len(all_speakers)} speakers should have good tokenization - use 'Test All Speakers Audio' to compare voices!"
    ])
    
    return "\n".join(test_results)

def test_all_speakers_audio():
    """Test audio generation with all speakers using a short phrase"""
    test_text = "Hello, this is a test."
    all_speakers = list(SPEAKERS.keys())  # Use all speakers from the dictionary
    
    results = [f"🎵 TESTING AUDIO GENERATION FOR ALL {len(all_speakers)} OFFICIAL SPEAKERS:\n"]
    
    working_count = 0
    failed_count = 0
    
    for speaker in all_speakers:
        try:
            audio, status = generate_speech(test_text, speaker, progress=lambda x, desc="": None)
            if audio is not None:
                results.append(f"✅ {speaker}: Audio generated successfully")
                working_count += 1
            else:
                results.append(f"❌ {speaker}: {status}")
                failed_count += 1
        except Exception as e:
            results.append(f"❌ {speaker}: ERROR - {str(e)}")
            failed_count += 1
    
    results.extend([
        f"\n📊 SUMMARY:",
        f"✅ Working: {working_count}/{len(all_speakers)} speakers",
        f"❌ Failed: {failed_count}/{len(all_speakers)} speakers",
        f"\n💡 All {len(all_speakers)} speakers are officially supported - compare the different voices!"
    ])
    return "\n".join(results)

def generate_speech(text, speaker="kavya", temperature=0.4, top_p=0.9, progress=gr.Progress()):
    """Generate speech from text using specified speaker voice - Enhanced version"""
    
    if not text.strip():
        return None, "⚠️ Please enter some text to convert to speech."
    
    try:
        progress(0.1, desc="Loading models...")
        load_models()
        
        progress(0.3, desc="Preparing input...")
        
        # Debug: Print speaker information
        print(f"🎭 DEBUG: Selected speaker = '{speaker}'")
        print(f"🎭 DEBUG: Available speakers = {list(SPEAKERS.keys())}")
        
        # Prepare input with speaker token - use correct format from Jupyter notebook
        prompt = f"<|spk_{speaker}|> {text}"
        print(f"🎭 DEBUG: Full prompt = '{prompt}'")
        
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"🎭 DEBUG: Prompt tokens = {prompt_tokens[:10]}...")  # Show first 10 tokens

        # Construct full sequence: [HUMAN] <spk_speaker> text [/HUMAN] [AI] [SPEECH]
        input_tokens = [
            START_OF_HUMAN_TOKEN,
            *prompt_tokens,
            END_OF_HUMAN_TOKEN,
            START_OF_AI_TOKEN,
            START_OF_SPEECH_TOKEN
        ]

        input_ids = torch.tensor([input_tokens], device=model.device)

        # Calculate max tokens based on text length
        max_tokens = min(int(len(text) * 1.3) * 7 + 21, 1024)  # Increased limit from notebook
        
        progress(0.5, desc="Generating audio tokens...")

        # Generate audio tokens using settings from Jupyter notebook
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
            )

        progress(0.7, desc="Processing audio tokens...")
        
        # Extract SNAC tokens
        generated_ids = output[0][len(input_tokens):].tolist()
        snac_tokens = [
            token_id for token_id in generated_ids
            if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
        ]

        if not snac_tokens:
            return None, "❌ No audio tokens generated. Please try a different text or settings."

        progress(0.9, desc="Decoding to audio...")
        
        # Decode audio using the fixed function
        audio = decode_snac_tokens(snac_tokens, snac_model)
        
        if audio is None:
            return None, "❌ Failed to decode audio tokens."
        
        # Create temporary file for audio using proper temp file handling
        try:
            # Create a temporary file that will be accessible to Gradio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"veena_{speaker}_") as temp_file:
                temp_filename = temp_file.name
            
            # Write audio to the temporary file
            sf.write(temp_filename, audio, 24000)
            
            # Add to cleanup list
            temp_files.append(temp_filename)
            
            # Periodic cleanup of old files
            periodic_cleanup()
            
            # Small delay to ensure file is fully written
            time.sleep(0.1)
            
            progress(1.0, desc="Audio generated successfully!")
            
            # Verify file exists and is readable
            if not os.path.exists(temp_filename):
                return None, "❌ Error: Generated audio file not found."
            
            # Get file size for verification
            file_size = os.path.getsize(temp_filename)
            print(f"🎵 Generated audio file: {temp_filename} ({file_size} bytes)")
            
            # Return the file path and success message
            return temp_filename, f"✅ Audio generated successfully using {SPEAKERS[speaker]} voice! ({file_size} bytes, {len(snac_tokens)} tokens)"
            
        except Exception as e:
            return None, f"❌ Error saving audio file: {str(e)}"
        
    except Exception as e:
        return None, f"❌ Error generating speech: {str(e)}"

# Example texts for different languages
EXAMPLE_TEXTS = {
    "Hindi": "आज मैंने एक नई तकनीक के बारे में सीखा जो कृत्रिम बुद्धिमत्ता का उपयोग करके मानव जैसी आवाज़ उत्पन्न कर सकती है।",
    "English": "Today I learned about a new technology that uses artificial intelligence to generate human-like voices.",
    "Code-mixed": "मैं तो पूरा presentation prepare कर चुका हूं! कल रात को ही मैंने पूरा code base चेक किया।",
    "Simple": "Hello, this is a test of the Veena text to speech system."
}

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
.speaker-radio .wrap {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.control-panel {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}
.footer-info {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    background: #e9ecef;
    border-radius: 8px;
    font-size: 0.9rem;
    color: #6c757d;
}
.warning-notice {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    color: #856404;
    font-size: 0.9rem;
}
"""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(css=custom_css, title="Veena TTS - Maya Research", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎵 Veena TTS</h1>
            <h2>Multi-lingual Text-to-Speech for Indian Languages</h2>
            <p>Advanced neural TTS supporting Hindi, English, and code-mixed text</p>
            <p><em>Developed by Maya Research • Powered by Llama Architecture</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Text input
                text_input = gr.Textbox(
                    label="📝 Enter text to convert to speech",
                    placeholder="Type in Hindi, English, or code-mixed text...",
                    lines=4,
                    max_lines=8
                )
                
                # Example buttons
                gr.Markdown("**Quick Examples:**")
                with gr.Row():
                    example_buttons = []
                    for lang, text in EXAMPLE_TEXTS.items():
                        btn = gr.Button(f"{lang}", size="sm", variant="secondary")
                        example_buttons.append((btn, text))
                
                # Speaker selection
                speaker_choice = gr.Radio(
                    choices=[(desc, key) for key, desc in SPEAKERS.items()],
                    label="🎭 Choose Voice",
                    value="kavya",
                    elem_classes=["speaker-radio"]
                )
                
                # Important notice about speakers
                gr.Markdown("""
                **✅ Speaker Info:** Model includes **11 official voices** with diverse characteristics. 
                All speakers are officially supported by Maya Research - use the debug tools below to test and compare the different voices.
                """, elem_classes=["warning-notice"])
            
            with gr.Column(scale=1):
                # Control panel
                gr.Markdown("### 🎛️ Audio Controls")
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.4,
                    step=0.1,
                    label="🌡️ Temperature (Creativity)",
                    info="Higher = more varied speech"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="🎯 Top-p (Focus)",
                    info="Lower = more focused speech"
                )
                
                generate_btn = gr.Button(
                    "🎵 Generate Speech",
                    variant="primary",
                    size="lg"
                )
                
                # Debug section
                gr.Markdown("### 🔧 Debug Tools")
                with gr.Row():
                    test_tokens_btn = gr.Button(
                        "🧪 Test Speaker Tokens",
                        variant="secondary",
                        size="sm"
                    )
                    test_audio_btn = gr.Button(
                        "🎵 Test All Speakers Audio",
                        variant="secondary", 
                        size="sm"
                    )
                
                # Examples section
                gr.Markdown("### 📚 Generate Examples")
                generate_examples_btn = gr.Button(
                    "🎭 Generate All Examples (Jupyter Notebook)",
                    variant="primary",
                    size="sm"
                )
        
        # Output section
        with gr.Row():
            with gr.Column():
                audio_output = gr.Audio(
                    label="🔊 Generated Audio",
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                    show_share_button=False
                )
                
                status_output = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    lines=2
                )
                
                debug_output = gr.Textbox(
                    label="🔧 Debug Output",
                    interactive=False,
                    lines=6,
                    visible=True,  # Show by default for debugging
                    value="🎭 Default speaker: kavya (Click 'Test Speaker Tokens' to verify all 11 speakers)"
                )
        
        # Set up example button callbacks
        for btn, text in example_buttons:
            btn.click(fn=lambda t=text: t, outputs=[text_input])
        
        # Wrapper function for better error handling and UI updates
        def generate_speech_wrapper(text, speaker, temperature, top_p, progress=gr.Progress()):
            """Wrapper function to handle generation with better UI feedback"""
            try:
                print(f"🎭 Starting generation: speaker={speaker}, text_length={len(text)}")
                progress(0.1, desc="Starting generation...")
                
                # Call the actual generation function
                result = generate_speech(text, speaker, temperature, top_p, progress)
                
                if result[0] is None:
                    print(f"❌ Generation failed: {result[1]}")
                    return None, result[1]
                else:
                    print(f"✅ Generation successful: {result[1]}")
                    return result[0], result[1]
                    
            except Exception as e:
                error_msg = f"❌ Wrapper error: {str(e)}"
                print(error_msg)
                return None, error_msg
        
        # Set up generation
        generate_btn.click(
            fn=generate_speech_wrapper,
            inputs=[text_input, speaker_choice, temperature, top_p],
            outputs=[audio_output, status_output]
        )
        
        # Set up debug tools
        def show_debug_and_test():
            result = test_speaker_tokens()
            return result
            
        def debug_speaker_selection(selected_speaker):
            debug_info = f"🎭 Radio button value: '{selected_speaker}'\n"
            debug_info += f"🎭 Type: {type(selected_speaker)}\n"
            debug_info += f"🎭 Available options: {list(SPEAKERS.keys())}\n"
            debug_info += f"🎭 Is valid? {'✅' if selected_speaker in SPEAKERS else '❌'}"
            return debug_info
            
        test_tokens_btn.click(
            fn=show_debug_and_test,
            outputs=[debug_output]
        )
        
        test_audio_btn.click(
            fn=test_all_speakers_audio,
            outputs=[debug_output]
        )
        
        # Example generation functionality
        def generate_examples_ui():
            """Generate examples and return status"""
            try:
                results = generate_examples()
                status_lines = []
                for name, filepath, status in results:
                    status_lines.append(f"{name}: {status}")
                    if filepath:
                        status_lines.append(f"  📁 {filepath}")
                return "🎭 Example Generation Results:\n\n" + "\n".join(status_lines)
            except Exception as e:
                return f"❌ Error generating examples: {str(e)}"
        
        generate_examples_btn.click(
            fn=generate_examples_ui,
            outputs=[debug_output]
        )
        
        # Debug speaker selection when it changes
        speaker_choice.change(
            fn=debug_speaker_selection,
            inputs=[speaker_choice],
            outputs=[debug_output]
        )
        
        # Footer with information
        gr.HTML("""
        <div class="footer-info">
            <h3>🌟 About Veena TTS</h3>
            <p><strong>Architecture:</strong> 3B parameter Llama-based transformer | <strong>Audio Quality:</strong> 24kHz SNAC codec</p>
            <p><strong>Languages:</strong> Hindi, English, Code-mixed | <strong>Latency:</strong> Ultra-low sub-80ms on H100</p>
            <p><strong>Voices:</strong> 11 official speakers with diverse characteristics</p>
            <p><strong>Speakers:</strong> kavya, apsara, agastya, vinaya, maitri, charu, ishana, kyra, mohini, varun, soumya</p>
            <br>
            <p>🔗 <a href="https://huggingface.co/maya-research/veena-tts" target="_blank">Model Hub</a> | 
               📄 <a href="https://huggingface.co/maya-research/veena-tts/blob/main/README.md" target="_blank">Documentation</a> |
               🧪 Use debug tools above to test all speakers</p>
        </div>
        """)
    
    return interface

# Create the demo interface globally so it can be imported
demo = create_interface()

# Launch the application
if __name__ == "__main__":
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,  # Using port 8000 as specified in memory
        share=False,
        show_error=True
    ) 