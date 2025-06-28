import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
import soundfile as sf
import numpy as np
import tempfile
import os
from datetime import datetime

# Global variables for model components
model = None
tokenizer = None
snac_model = None

# Control token IDs (fixed for Veena)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

# Available speakers with descriptions
SPEAKERS = {
    "kavya": "🎭 Kavya - Expressive female voice",
    "agastya": "🎯 Agastya - Clear male voice", 
    "maitri": "💫 Maitri - Friendly female voice",
    "vinaya": "🎪 Vinaya - Warm male voice"
}

def load_models():
    """Load the Veena TTS model and SNAC decoder"""
    global model, tokenizer, snac_model
    
    if model is None:
        print("Loading Veena TTS model...")
        
        # Model configuration for 4-bit inference
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "maya-research/Veena",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("maya-research/Veena", trust_remote_code=True)
        
        # Initialize SNAC decoder
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        if torch.cuda.is_available():
            snac_model = snac_model.cuda()
        
        print("Models loaded successfully!")

def decode_snac_tokens(snac_tokens, snac_model):
    """De-interleave and decode SNAC tokens to audio"""
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None

    # De-interleave tokens into 3 hierarchical levels
    codes_lvl = [[] for _ in range(3)]
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    for i in range(0, len(snac_tokens), 7):
        # Level 0: Coarse (1 token)
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        # Level 1: Medium (2 tokens)
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
        # Level 2: Fine (4 tokens)
        codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])

    # Convert to tensors for SNAC decoder
    # Get device from SNAC model parameters
    try:
        device = next(snac_model.parameters()).device
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            raise ValueError("Invalid SNAC token values")
        hierarchical_codes.append(tensor)

    # Decode with SNAC
    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)

    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()

def generate_speech(text, speaker="kavya", temperature=0.4, top_p=0.9, progress=gr.Progress()):
    """Generate speech from text using specified speaker voice"""
    
    if not text.strip():
        return None, "⚠️ Please enter some text to convert to speech."
    
    try:
        progress(0.1, desc="Loading models...")
        load_models()
        
        progress(0.3, desc="Preparing input...")
        
        # Debug: Print speaker information
        print(f"🎭 DEBUG: Selected speaker = '{speaker}'")
        print(f"🎭 DEBUG: Available speakers = {list(SPEAKERS.keys())}")
        
        # Prepare input with speaker token
        prompt = f"<spk_{speaker}> {text}"
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
        max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)
        
        progress(0.5, desc="Generating audio tokens...")

        # Generate audio tokens
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
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
        
        # Decode audio
        audio = decode_snac_tokens(snac_tokens, snac_model)
        
        if audio is None:
            return None, "❌ Failed to decode audio tokens."
        
        # Create temporary file for audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = f"veena_output_{speaker}_{timestamp}.wav"
        sf.write(temp_file, audio, 24000)
        
        progress(1.0, desc="Audio generated successfully!")
        
        return temp_file, f"✅ Audio generated successfully using {SPEAKERS[speaker]} voice!"
        
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
        
        # Output section
        with gr.Row():
            with gr.Column():
                audio_output = gr.Audio(
                    label="🔊 Generated Audio",
                    type="filepath",
                    interactive=False
                )
                
                status_output = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    lines=2
                )
        
        # Set up example button callbacks
        for btn, text in example_buttons:
            btn.click(fn=lambda t=text: t, outputs=[text_input])
        
        # Set up generation
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, speaker_choice, temperature, top_p],
            outputs=[audio_output, status_output]
        )
        
        # Footer with information
        gr.HTML("""
        <div class="footer-info">
            <h3>🌟 About Veena TTS</h3>
            <p><strong>Architecture:</strong> 3B parameter Llama-based transformer | <strong>Audio Quality:</strong> 24kHz SNAC codec</p>
            <p><strong>Languages:</strong> Hindi, English, Code-mixed | <strong>Latency:</strong> Ultra-low sub-80ms on H100</p>
            <p><strong>Voices:</strong> 4 distinct speakers with unique characteristics</p>
            <br>
            <p>🔗 <a href="https://huggingface.co/maya-research/Veena" target="_blank">Model Hub</a> | 
               📄 <a href="https://huggingface.co/maya-research/Veena/blob/main/README.md" target="_blank">Documentation</a> |
               ⭐ <a href="https://github.com" target="_blank">Star on GitHub</a></p>
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