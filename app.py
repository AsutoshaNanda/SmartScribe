# Clean Imports
import os
import gc
import torch
import numpy as np
import uuid
import pycountry
import yt_dlp
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from huggingface_hub import login
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Setup YouTube Cookies from Environment
def setup_cookies():
    """Write cookies from environment variable to cookies.txt file"""
    cookies_content = os.getenv('YOUTUBE_COOKIES')
    if cookies_content:
        with open('cookies.txt', 'w') as f:
            f.write(cookies_content)
        print("✅ Cookies loaded successfully")
        return True
    else:
        print("⚠️ No cookies found in environment - YouTube downloads may fail")
        return False

# Call cookie setup when app starts
setup_cookies()

# Hugging Face Login (optional - uses environment variable)
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token, add_to_git_credential=True)

# Model names
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
QWEN = "Qwen/Qwen3-4B-Instruct-2507"
PHI = "microsoft/Phi-4-mini-instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
Gemma = 'google/gemma-3-4b-it'

# YouTube Download Function
def _download_if_youtube(source):
    if "youtube.com" in source or "youtu.be" in source:
        unique = str(uuid.uuid4())[:8]
        filename = f"audio_{unique}.%(ext)s"
        
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": filename,
            "quiet": True,
            "extractor_args": {"youtube": {"player_client": ["default"]}},
            "cookiefile": "cookies.txt",
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=True)
            return ydl.prepare_filename(info)
    else:
        return source

# Convert to WAV
def _to_wav(path):
    unique = str(uuid.uuid4())[:8]
    wav_path = f"audio_{unique}.wav"
    AudioSegment.from_file(path).export(wav_path, format="wav")
    return wav_path

# Transcription Function
def transcription_whisper(source):
    torch.cuda.empty_cache()
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel('medium', device=device, compute_type=compute)
    file_path = _download_if_youtube(source)
    wav_path = _to_wav(file_path)
    segments, info = model.transcribe(wav_path)
    result = []
    formatted_output = "**TRANSCRIPTION**\n" + "="*50 + "\n\n"
    
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
        formatted_output += f"[{seg.start:.2f}s - {seg.end:.2f}s]\n{seg.text.strip()}\n\n"
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return formatted_output, result

# System Prompts
system_prompt = """
You are an expert assistant that generates clear, concise, and well-structured
Minutes of Meeting (MOM) documents from raw meeting transcripts.
Your output must be in clean Markdown format (without code blocks) and must include:
- **Meeting Summary:** A brief overview of the meeting context, agenda, and participants (if mentioned).
- **Key Discussion Points:** Major topics, decisions, or debates.
- **Takeaways:** Important insights, conclusions, and agreements.
- **Action Items:** Actionable tasks with responsible owners and deadlines
(e.g., "John will prepare the project plan by Monday").

Guidelines:
- Write in professional, easy-to-read language.
- Summarize meaningfully; avoid filler words or irrelevant content.
- Omit transcription artifacts (e.g., "um", "okay", "yeah").
- Do not include timestamps.
- Maintain a formal and factual tone while being concise.
- Focus entirely on clarity, structure, and readability.
"""

def user_prompt_for(source):
    formatted_output, segments = transcription_whisper(source)
    transcript_text = " ".join(seg["text"] for seg in segments)
    user_prompt = f"""
Please write well-structured **Minutes of Meeting (MOM)** in Markdown format (without code blocks), including:
- **Summary:** Include attendees, location, and date if mentioned.
- **Key Discussion Points:** List the main topics or discussions.
- **Takeaways:** Summaries of conclusions or insights.
- **Action Items:** Tasks with clear owners and deadlines.

Transcription:
{transcript_text}
"""
    return user_prompt

def messages_for(source):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt_for(source)}
    ]
    return messages

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4'
)

# Generate MOM
def generate(model_name, source):
    messages = messages_for(source)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config)
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=5000)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    mom_output = result
    if '<|start_header_id|>assistant<|end_header_id|>' in mom_output:
        mom_output = mom_output.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
    elif 'assistant' in mom_output:
        parts = mom_output.split('assistant')
        if len(parts) > 1:
            mom_output = parts[-1]
    
    mom_output = mom_output.replace('<|eot_id|>', '').replace('<|end_header_id|>', '').strip()
    
    if '**Minutes of Meeting' in mom_output:
        mom_output = mom_output.split('**Minutes of Meeting')[1]
        mom_output = '**Minutes of Meeting' + mom_output
    elif '**MINUTES' in mom_output:
        mom_output = mom_output.split('**MINUTES')[1]
        mom_output = '**MINUTES' + mom_output
    
    del model, inputs, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    yield mom_output.strip()

# Translation Functions
def valid_language(lang):
    return bool(
        pycountry.languages.get(name=lang.capitalize()) or
        pycountry.languages.get(alpha_2=lang.lower()) or
        pycountry.languages.get(alpha_3=lang.lower())
    )

system_prompt_translate = "You are a translation assistant. Given a target language and some content, translate the content accurately into that language, preserving meaning, tone, and style, and return only the translated text. Also maintain proper format."

def user_prompt_translate(source, lang):
    if not valid_language(lang):
        return f"Invalid language: {lang}. Please provide a valid language name or code."
    
    transcript_text, _ = transcription_whisper(source)
    lines = transcript_text.split('\n')
    text_lines = []
    for line in lines:
        if line.startswith('**') or line.startswith('=') or line.startswith('[') or not line.strip():
            continue
        text_lines.append(line.strip())
    
    transcript_text = " ".join(text_lines)
    max_chars = 3000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "..."
    
    user_prompt = f"""Translate the following text into {lang}.

Instructions:
- Provide ONLY the translation in {lang}
- Do NOT add any explanations or comments
- Preserve the original meaning and tone
- Keep formatting simple and clean

Text to translate:
{transcript_text}

{lang} translation:"""
    return user_prompt

def messages_for_translate(source, lang):
    messages = [
        {'role': 'system', 'content': system_prompt_translate},
        {'role': 'user', 'content': user_prompt_translate(source, lang)}
    ]
    return messages

def translate_transcribe(model_name, source, lang):
    messages = messages_for_translate(source, lang)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config)
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=5000)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    translate_output = result
    if '<|start_header_id|>assistant<|end_header_id|>' in translate_output:
        translate_output = translate_output.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
    elif 'assistant' in translate_output:
        parts = translate_output.split('assistant')
        if len(parts) > 1:
            translate_output = parts[-1]
    
    translate_output = translate_output.replace('<|eot_id|>', '').replace('<|end_header_id|>', '').strip()
    if 'translation:' in translate_output.lower():
        translate_output = translate_output.split('translation:')[-1].strip()
    if "Here's an edited version:" in translate_output:
        translate_output = translate_output.split("Here's an edited version:")[0].strip()
    translate_output = translate_output.replace('assistant', '').strip()
    
    # Format into paragraphs
    sentences = translate_output.split('. ')
    paragraphs = []
    current_para = []
    sentence_count = 0
    
    for sentence in sentences:
        current_para.append(sentence.strip())
        sentence_count += 1
        if sentence_count >= 4:
            paragraphs.append('. '.join(current_para) + '.')
            current_para = []
            sentence_count = 0
    
    if current_para:
        paragraphs.append('. '.join(current_para) + ('.' if not current_para[-1].endswith('.') else ''))
    
    formatted_output = '\n\n'.join(paragraphs)
    
    del model, inputs, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    yield formatted_output

def translate_transcribe_gemma(Gemma, source, lang):
    messages = [{'role': 'user', 'content': user_prompt_translate(source, lang)}]
    tokenizer = AutoTokenizer.from_pretrained(Gemma, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to('cuda')
    model = AutoModelForCausalLM.from_pretrained(Gemma, device_map='auto', quantization_config=quant_config)
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=5000)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    translate_output = result
    if '<|start_header_id|>assistant<|end_header_id|>' in translate_output:
        translate_output = translate_output.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
    elif 'assistant' in translate_output:
        parts = translate_output.split('assistant')
        if len(parts) > 1:
            translate_output = parts[-1]
    
    translate_output = translate_output.replace('<|eot_id|>', '').replace('<|end_header_id|>', '').strip()
    if 'translation:' in translate_output.lower():
        translate_output = translate_output.split('translation:')[-1].strip()
    if "Here's an edited version:" in translate_output:
        translate_output = translate_output.split("Here's an edited version:")[0].strip()
    translate_output = translate_output.replace('assistant', '').strip()
    
    del model, inputs, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    yield translate_output

# Optimization Functions
def optimize(model_name, source):
    if model_name == 'LLAMA':
        result = generate(LLAMA, source)
    elif model_name == 'PHI':
        result = generate(PHI, source)
    elif model_name == 'QWEN':
        result = generate(QWEN, source)
    elif model_name == 'DEEPSEEK':
        result = generate(DEEPSEEK, source)
    
    for chunk in result:
        yield chunk

def optimize_translate(model_name, source, lang):
    if model_name == 'LLAMA':
        translate = translate_transcribe(LLAMA, source, lang)
    elif model_name == 'PHI':
        translate = translate_transcribe(PHI, source, lang)
    elif model_name == 'QWEN':
        translate = translate_transcribe(QWEN, source, lang)
    elif model_name == 'DEEPSEEK':
        translate = translate_transcribe(DEEPSEEK, source, lang)
    elif model_name == 'Gemma':
        translate = translate_transcribe_gemma(Gemma, source, lang)
    
    for chunk_tr in translate:
        yield chunk_tr

# Helper function for file/link input
def get_source_input(file, link):
    if file is not None:
        return file.name if hasattr(file, 'name') else file
    return link if link else ""

# CSS Styling
css = """
#file-box {
    min-height: 500px !important;
}
#file-box button {
    height: 100% !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
}
#box {
    min-height: 550px !important;
}
"""

# Gradio Interface
with gr.Blocks(css=css) as ui:
    gr.Markdown("## Transcription & MOM Generator & Translator")
    
    gr.Markdown("""
    ### 📌 Note: YouTube Link Support
    Due to YouTube's bot protection, only **direct file uploads** are guaranteed to work. 
    YouTube links may fail without authentication cookies.
    
    **Workaround:** Upload your audio/video file directly for best results.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_file = gr.File(label="Upload Audio/Video", file_types=["audio", "video"], elem_id="file-box")
            input_link = gr.Textbox(label="YouTube Link (optional)", lines=2)
        
        with gr.Column(scale=2):
            output_transcription = gr.Textbox(label="Transcription", lines=25, elem_id='box')
            transcribe = gr.Button("Transcribe", variant="primary", scale=2)
        
        with gr.Column(scale=2):
            output_summary = gr.Textbox(label="MOM Output", lines=25, elem_id='box')
            summarize = gr.Button("Summarize", variant="secondary", scale=2)
        
        with gr.Column(scale=2):
            output_translate = gr.Textbox(label='Translation Output', lines=20)
            language_input = gr.Textbox(label="Target Language", value="English", lines=1)
            translate = gr.Button('Translate', scale=2)
    
    with gr.Row():
        model = gr.Dropdown(
            ["LLAMA", "PHI", "QWEN", "DEEPSEEK", 'Gemma'],
            label="Choose Your Model",
            value="LLAMA"
        )
    
    # Wrapper functions to handle generators properly
    def summarize_wrapper(model, file, link):
        source = get_source_input(file, link)
        for result in optimize(model, source):
            yield result
    
    def translate_wrapper(model, file, link, lang):
        source = get_source_input(file, link)
        for result in optimize_translate(model, source, lang):
            yield result
    
    # Event handlers with file/link support
    transcribe.click(
        fn=lambda file, link: transcription_whisper(get_source_input(file, link))[0],
        inputs=[input_file, input_link],
        outputs=[output_transcription]
    )
    
    summarize.click(
        fn=summarize_wrapper,
        inputs=[model, input_file, input_link],
        outputs=[output_summary]
    )
    
    translate.click(
        fn=translate_wrapper,
        inputs=[model, input_file, input_link, language_input],
        outputs=[output_translate]
    )

# Launch the app
if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
