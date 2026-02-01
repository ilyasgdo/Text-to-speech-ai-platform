# 🎵 Qwen3-TTS Studio

A modern Text-to-Speech application powered by **Qwen3-TTS** with a beautiful web interface.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-macOS%20|%20Linux-lightgrey.svg)

## ✨ Features

- 🎤 **Voice Clone** - Clone any voice from an audio sample
- ✨ **Voice Design** - Create voices from text descriptions
- 🦙 **Ollama Chat** - Talk to AI and hear the responses (requires Ollama)
- 🍎 **Apple Silicon Optimized** - Works great on M1/M2/M3 Macs
- 🎨 **Modern UI** - Beautiful Gradio web interface

## 🚀 Quick Start

### One-Command Install & Run

```bash
git clone https://github.com/YOUR_USERNAME/Text-to-speech-ai-platform.git
cd Text-to-speech-ai-platform
chmod +x setup.sh
./setup.sh
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Text-to-speech-ai-platform.git
cd Text-to-speech-ai-platform

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python qwen_tts_studio.py
```

## 📋 Requirements

- **Python 3.12+**
- **macOS** (Apple Silicon recommended) or **Linux**
- ~4GB disk space for models
- (Optional) **Ollama** for AI chat feature

## 🦙 Ollama Setup (Optional)

To use the "Talk with Ollama" feature:

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
brew services start ollama

# Pull a model
ollama pull qwen3:0.6b
```

## 📖 Usage

Once running, open http://localhost:7860 in your browser.

### Voice Clone Tab
1. Upload or record an audio sample
2. (Optional) Add transcription for better quality
3. Enter text to synthesize
4. Click "Generate"

### Voice Design Tab
1. Describe the voice you want
2. Enter text to synthesize
3. Click "Generate"

### Ollama Chat Tab
1. Select an Ollama model
2. (Optional) Upload custom voice
3. Type your message
4. Audio plays automatically!

## 🛠️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Device | CPU | Use CPU (stable) or MPS (faster on Mac) |
| Model | 0.6B Base | Choose between 0.6B (fast) or 1.7B (quality) |
| Language | French | Supports 10+ languages |

## 📁 Project Structure

```
Text-to-speech-ai-platform/
├── qwen_tts_studio.py   # Main application
├── qwen_tts_app.py      # CLI version
├── test_tts.py          # Simple test script
├── requirements.txt     # Python dependencies
├── setup.sh             # Install & run script
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

## 📄 License

MIT License - feel free to use this project for any purpose.

## 🙏 Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [Gradio](https://gradio.app/) for the web UI
- [Ollama](https://ollama.ai/) for local LLM integration
