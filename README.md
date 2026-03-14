# 🎙️ SDET Interview Coach

A fully local, privacy-preserving, terminal-based AI application designed to help you practice for SDET (Software Development Engineer in Test) interviews using voice-to-voice interaction.

Powered by **Whisper** (Speech-to-Text), **Piper** (Text-to-Speech), and **Ollama** (Local LLMs like Llama 3.1 & Qwen).

---

## 🚀 Key Features

- **Local & Private**: No audio or text data ever leaves your machine.
- **Voice-to-Voice Interaction**: Practice speaking your answers just like a real interview.
- **RAG (Retrieval-Augmented Generation)**: Ingest your own study materials (.pdf, .txt, .md) to ground the AI's questions in your specific resources.
- **Multi-Round Support**: Practice specifically for HR, Technical, Managerial, or CTO rounds.
- **Session Intelligence**: 
  - Real-time timer and progress tracking.
  - Active keyboard overrides (`done`, `quit`, `skip`) while the mic is live.
  - Smart handling of strict model templates (optimized for Llama 3.1 & Qwen).
- **Post-Interview Report**: Get an immediate JSON-formatted feedback report with scores, strengths, and areas for improvement.

---

## 🛠️ Tech Stack

- **Ollama**: Orchestrates local Large Language Models.
- **Faster-Whisper**: High-performance local speech-to-text.
- **Piper TTS**: Low-latency, ultra-fast local text-to-speech.
- **Sentence-Transformers**: Local embeddings for the document knowledge base.
- **Rich**: Beautiful terminal UI with progress bars and panels.

---

## 📦 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: [Download and install](https://ollama.com/)
- **FFmpeg**: Required for audio processing.

### 2. Clone and Setup Environment
```bash
git clone <repository-url>
cd sdet-interview-app

# Run the automated setup script
chmod +x setup.sh
./setup.sh
```
*The setup script will create a virtual environment, install dependencies, download the Piper binary, and fetch the default voice model.*

### 3. Pull LLM Models
For the best experience, pull these models via Ollama:
```bash
# Recommended: Fast & Snappy
ollama pull llama3.1:8b

# Advanced: High Intelligence (Slow)
ollama pull qwen3:30b
```

---

## 📖 How to Use

### Step 1: Index your study docs (Optional)
Place your `.pdf`, `.md`, or `.txt` files into `resources/<topic_name>/` and run:
```bash
python main.py --index
```

### Step 2: Start an Interview
```bash
python main.py
```
Follow the interactive wizard to select your **Model**, **Round**, **Topic**, and **Duration**.

### Step 3: During the Interview
When the `🎙 Recording…` status appears:
- **Speak** your answer naturally.
- **Silence Detection**: Stay silent for **7 seconds** (configurable) to auto-submit.
- **Keyboard Overrides**: Type these into the terminal while recording:
  - `done` + Enter: Instantly submit your current answer.
  - `skip` + Enter: Move to the next question.
  - `quit` + Enter: Gracefully end the interview early.

---

## 🔧 Troubleshooting & Performance

### "Thinking" Time
If the AI takes too long to respond (the "🤖 Thinking..." spinner), it is likely because the model is generating internal reasoning tokens.
- **Fix**: Use a smaller model like `llama3.1:8b`.
- **Note**: We have implemented a specialized "thought stripper" that hides `<think>` tags from being displayed or spoken, ensuring you only hear the interviewer's actual response.

### Empty Responses (Llama 3.1)
Llama 3 models are strict and require a user kickoff. We have optimized the `InterviewSession` logic to ensure the chat template is always correctly triggered, so the interviewer never stays silent.

### Microphone Issues
- Ensure your default system microphone is correctly selected.
- If you see a `PortAudio` error, ensure you have `ffmpeg` and the `sounddevice` dependencies installed correctly via `./setup.sh`.

---

## 📜 License
This project is open-source. Build, modify, and ace your interviews! 🚀


🎙️ SDET Interview Coach Walkthrough
This document provides a comprehensive overview of the SDET Interview Coach, a private, local AI application for practicing interviews.

🚀 Features & Capabilities
100% Local: No data sent to the cloud.
Voice-to-Voice: Interactive oral practice.
RAG-Powered: Grounds questions in your own study documents.
Feedback Engine: Generates professional scoring and recommendations.
🛠️ Setup Guide
Dependencies: Run 
./setup.sh
 to install everything automatically.
Ollama: Ensure Ollama is running (ollama serve).
Models: For high speed, use ollama pull llama3.1:8b.
📖 Using the App
Starting a Session
Run python main.py and follow the menu to choose your round (HR, Technical, etc.) and topic.

During the Interview
The app uses 7-second silence detection to know when you've finished speaking.
Type done: Instantly submit your answer without waiting for silence.
Type skip: Move to the next question.
Type quit: End the session and get your feedback report.
🔧 Optimized for Llama 3.1 & Qwen
We've implemented specialized fixes to make the app work perfectly with the latest local models:

Thought Stripping: Automatically removes internal reasoning (<think>) tags so you only hear the final answer.
Chat Template Fix: Uses a hidden "kickoff" message to ensure Llama 3 models always start the conversation properly.
Context Reordering: Keeps system messages behind the user turn to prevent "silent generation" bugs in strict models.
Good luck with your interview practice!


