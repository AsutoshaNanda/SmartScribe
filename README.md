
<div align="center">

# SmartScribe

  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://openai.com/research/whisper)
[![Faster-Whisper](https://img.shields.io/badge/Faster--Whisper-Audio-orange.svg)](https://github.com/guillaumekln/faster-whisper)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/spaces/itsasutosha/SmartScribe)
[![LLAMA](https://img.shields.io/badge/Meta-LLAMA-red.svg)](https://www.llama.com/)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)

**AI-Powered Audio Transcription, Meeting Minutes Generation, and Multi-Language Translation**


</div>

<div align="center">
<h2>📋 Table of Contents</h2>
<table>
  <tr>
    <td><a href="#features">✨ Features</a></td>
    <td><a href="#supported-models">🤖 Supported Models</a></td>
    <td><a href="#requirements">📦 Requirements</a></td>
    <td><a href="#installation">🔧 Installation</a></td>
  </tr>
  <tr>
    <td><a href="#configuration">⚙️ Configuration</a></td>
    <td><a href="#usage">🎮 Usage</a></td>
    <td><a href="#architecture">🏗️ Architecture</a></td>
    <td><a href="#troubleshooting">🐛 Troubleshooting</a></td>
  </tr>
  </tr>
</table>
</div>

---

## ✨ Features

### 🎙️ **Audio/Video Transcription**
- Convert YouTube links or local audio/video files to text
- Support for multiple audio formats (MP3, WAV, M4A, etc.)
- GPU-accelerated transcription using Faster-Whisper
- Timestamped transcription output

### 📝 **Minutes of Meeting Generation**
- Automatically generate structured MOM documents
- Professional summary with participants and date
- Key discussion points extraction
- Takeaways and conclusions identification
- Actionable items with clear ownership and deadlines
- Markdown-formatted output

### 🌍 **Multi-Language Translation**
- Translate transcriptions into any supported language
- Language validation using pycountry
- Clean, paragraph-formatted output
- Preserves original meaning and tone

### 🤖 **Multi-Model Support**
- LLAMA 3.2 3B Instruct
- PHI 4 Mini Instruct
- QWEN 3 4B Instruct
- DeepSeek R1 Distill Qwen 1.5B
- Google Gemma 3 4B IT

### 🖥️ **Interactive Web UI**
- Beautiful Gradio interface
- Drag-and-drop file upload
- YouTube link support
- Side-by-side input and output panels
- Model selection dropdown
- Real-time streaming responses

### ⚡ **Performance Optimization**
- 4-bit quantization for efficient inference
- GPU acceleration support
- Memory-efficient model loading
- Garbage collection and cache clearing

---

## 🤖 Supported Models

| Model | Provider | Size | Speed | Quality | Best For |
|-------|----------|------|-------|---------|----------|
| LLAMA | Meta | 3.2B | ⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| PHI | Microsoft | 4B | ⚡⚡ | ⭐⭐⭐⭐ | General |
| QWEN | Alibaba | 4B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast |
| DEEPSEEK | DeepSeek | 1.5B | ⚡⚡⚡ | ⭐⭐⭐ | Minimal Resources |
| Gemma | Google | 3-4B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Efficient |

---

## 📦 Requirements

### System Requirements
- **Python 3.8+**
- **CUDA-capable GPU** (recommended for transcription)
- **8GB+ RAM**
- **FFmpeg** for audio processing

### Python Dependencies
```
gradio>=4.0.0
torch>=2.0.0
transformers>=4.30.0
faster-whisper>=0.10.0
yt-dlp>=2023.0.0
pydub>=0.25.0
bitsandbytes>=0.41.0
accelerate>=0.20.0
pycountry>=23.0.0
huggingface-hub>=0.16.0
```

---

## 🔧 Local Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup HuggingFace Token
Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_token_here
```

Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### 4. Setup YouTube Cookies (Optional)
For YouTube link support, set environment variable or create `cookies.txt`:
```bash
export YOUTUBE_COOKIES="your_cookies_content"
```

Or create `cookies.txt` with Netscape HTTP Cookie format.

---

## ⚙️ Configuration

### Model Selection
Edit model paths in `app.py`:
```python
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
QWEN = "Qwen/Qwen3-4B-Instruct-2507"
PHI = "microsoft/Phi-4-mini-instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
Gemma = 'google/gemma-3-4b-it'
```

### Quantization Configuration
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4'
)
```

### Server Configuration
```python
ui.launch(server_name="0.0.0.0", server_port=7860)
```

---

## ☁️ Deployment

### HuggingFace Spaces

SmartScribe is deployed and available at: [https://huggingface.co/spaces/itsasutosha/SmartScribe](https://huggingface.co/spaces/itsasutosha/SmartScribe)

**Features:**
- ✅ Free to use
- ✅ No installation needed
- ✅ GPU-accelerated inference
- ✅ Persistent storage for temporary files
- ✅ Real-time streaming output

**To Deploy Your Own:**

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Create a new Space
3. Select "Gradio" as the framework
4. Upload your repository files
5. Add secrets in Space settings:
   - `HF_TOKEN`: Your HuggingFace token
   - `YOUTUBE_COOKIES`: (Optional) YouTube authentication cookies

6. Space will automatically build and deploy

---

## 🎮 Usage

### Quick Start - Live Demo

#### 🌐 Try Online
Visit the live application at: **[SmartScribe on HuggingFace Spaces](https://huggingface.co/spaces/itsasutosha/SmartScribe)**

No installation required! Just upload your audio/video or paste a YouTube link.

#### 1. Launch Application (Local Setup)
```bash
python app.py
```

The application will start at `http://0.0.0.0:7860`

#### 2. Using the Web UI

1. **Upload Content**:
   - Upload audio/video file directly, OR
   - Paste YouTube link

2. **Choose Operation**:
   - Click "Transcribe" to extract text from audio
   - Click "Summarize" to generate Minutes of Meeting
   - Click "Translate" for multi-language translation

3. **Select Model**: Choose preferred LLM from dropdown

4. **View Results**: See output in corresponding text areas

### Programmatic Usage

#### Transcribe Audio
```python
from app import transcription_whisper

formatted_output, segments = transcription_whisper("audio.mp3")
print(formatted_output)

# Access individual segments
for seg in segments:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
```

#### Generate Minutes of Meeting
```python
from app import optimize

for chunk in optimize("LLAMA", "audio.mp3"):
    print(chunk, end="", flush=True)
```

#### Translate Transcription
```python
from app import optimize_translate

for chunk in optimize_translate("LLAMA", "audio.mp3", "Spanish"):
    print(chunk, end="", flush=True)
```

---

## 🏗️ Architecture

### Component Overview

```
┌──────────────────────────────────────────────────────────┐
│            Gradio Web Interface (UI Layer)               │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌────────────────────┐  ┌────────────────┐              │
│  │ Audio/Video Input  │  │  Model Select  │              │
│  └────────────────────┘  └────────────────┘              │
│                                                            │
│  ┌────────────────────────────────────────────────┐      │
│  │  Transcription | MOM | Translation Output     │      │
│  └────────────────────────────────────────────────┘      │
├──────────────────────────────────────────────────────────┤
│          Multi-Module Processing Layer                   │
├─────────────────┬──────────────────┬──────────────────┤
│                 │                  │                  │
│  Transcription  │  MOM Generation  │   Translation   │
│  Module         │  Module          │   Module        │
│  ───────────    │  ──────────────  │   ────────────  │
│  • Download     │  • System Prompt │  • Language     │
│  • Convert      │  • User Prompt   │    Validation   │
│  • Transcribe   │  • Generation    │  • Extraction   │
│                 │                  │  • Translation  │
├─────────────────┴──────────────────┴──────────────────┤
│              LLM Integration Layer                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  LLAMA | PHI | QWEN | DEEPSEEK | Gemma                │
│  (with 4-bit Quantization & GPU Acceleration)          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `transcription_whisper()` | Convert audio to text | Audio file/URL | Formatted transcript |
| `user_prompt_for()` | Build MOM generation prompt | Audio source | User prompt string |
| `messages_for()` | Build message structure | Audio source | Message array |
| `generate()` | Route to LLM for MOM | Model, audio | Generator yielding output |
| `optimize()` | Execute MOM generation | Model, audio | Streaming MOM content |
| `user_prompt_translate()` | Build translation prompt | Audio, language | Translation prompt |
| `messages_for_translate()` | Build translation messages | Audio, language | Message array |
| `translate_transcribe()` | Execute translation | Model, audio, lang | Streaming translation |
| `optimize_translate()` | Route translation task | Model, audio, lang | Streaming result |
| `valid_language()` | Validate language code | Language string | Boolean |

---

## 🐛 Troubleshooting

### Issue: YouTube download fails
**Solution**: Update YouTube cookies or use direct file upload
```bash
export YOUTUBE_COOKIES="your_updated_cookies"
# or use direct file upload instead
```

### Issue: CUDA out of memory
**Solution**: Reduce model size or use CPU inference
```python
device = "cpu"  # Force CPU usage
```

### Issue: HuggingFace authentication failed
**Solution**: Verify HF_TOKEN in .env file
```bash
huggingface-cli login  # Interactive login
```

### Issue: Transcription is slow
**Solution**: Ensure CUDA is properly configured
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### Issue: Language validation fails
**Solution**: Use full language name or ISO code
```python
# Valid formats:
valid_language("English")  # Full name
valid_language("en")       # ISO 639-1 code
valid_language("eng")      # ISO 639-3 code
```

### Issue: Memory issues with large files
**Solution**: Reduce chunk size or break audio into segments
```python
# Process smaller chunks
segment_duration = 300  # 5 minutes per segment
```

### Issue: Generated MOM missing action items
**Solution**: Try different model or update system prompt
- Claude models typically produce better structured output
- QWEN is faster and generally reliable

---

## 📁 File Structure

```
smartscribe/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── cookies.txt                # YouTube cookies (optional)
├── README.md                  # This file
├── LICENSE                    # MIT License
└── .env                       # Environment variables (git-ignored)
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Citation

If you use SmartScribe in your project, please cite:
```bibtex
@software{smartscribe2025,
  author = {Asutosha Nanda},
  title = {SmartScribe},
  year = {2025},
  url = {https://huggingface.co/spaces/itsasutosha/SmartScribe}
}
```

---

<div align="center">

**Intelligent Audio Transcription & Meeting Documentation**  
Powered by Advanced LLMs and Faster-Whisper

Deployed on [HuggingFace Spaces](https://huggingface.co/spaces/itsasutosha/SmartScribe)

</div>
