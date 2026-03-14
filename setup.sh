#!/usr/bin/env bash
# =============================================================================
# SDET Interview Coach — One-time Setup Script
# macOS aarch64 (Apple Silicon) / M2 Mac Mini
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPER_DIR="$PROJECT_DIR/piper"
VOICES_DIR="$PROJECT_DIR/voices"
VENV_DIR="$PROJECT_DIR/venv"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        SDET Interview Coach — Setup                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ────────────────────────────────────────────────────
echo "▶ Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
  echo "✗ python3 not found. Install via: brew install python@3.11"; exit 1
fi

if ! command -v brew &>/dev/null; then
  echo "✗ Homebrew not found. Install from https://brew.sh"; exit 1
fi

if ! command -v ollama &>/dev/null; then
  echo "✗ Ollama not found. Install from https://ollama.com/download"; exit 1
fi

# Check ffmpeg (required by faster-whisper)
if ! command -v ffmpeg &>/dev/null; then
  echo "▶ Installing ffmpeg via Homebrew..."
  brew install ffmpeg
fi

# Check portaudio (required by sounddevice)
if ! brew list portaudio &>/dev/null 2>&1; then
  echo "▶ Installing portaudio via Homebrew..."
  brew install portaudio
fi

echo "✓ Prerequisites OK"

# ── 2. Python virtual environment ────────────────────────────────────────────
echo ""
echo "▶ Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "▶ Upgrading pip..."
pip install --upgrade pip --quiet

echo "▶ Installing Python dependencies (this may take a few minutes)..."
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

echo "✓ Python environment ready"

# ── 3. Piper TTS binary ───────────────────────────────────────────────────────
echo ""
echo "▶ Setting up Piper TTS..."
mkdir -p "$PIPER_DIR" "$VOICES_DIR"

PIPER_VERSION="2023.11.14-2"
PIPER_ARCHIVE="piper_macos_aarch64.tar.gz"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/${PIPER_ARCHIVE}"

if [ ! -f "$PIPER_DIR/piper" ]; then
  echo "  Downloading Piper binary (${PIPER_VERSION})..."
  curl -L --progress-bar "$PIPER_URL" -o "/tmp/$PIPER_ARCHIVE"
  tar -xzf "/tmp/$PIPER_ARCHIVE" -C "$PIPER_DIR" --strip-components=1
  rm "/tmp/$PIPER_ARCHIVE"
  chmod +x "$PIPER_DIR/piper"
  echo "  ✓ Piper binary installed"
else
  echo "  ✓ Piper binary already present — skipping"
fi

# ── 4. Piper voice model ──────────────────────────────────────────────────────
VOICE_MODEL="en_US-amy-medium"
VOICE_ONNX="$VOICES_DIR/${VOICE_MODEL}.onnx"
VOICE_JSON="$VOICES_DIR/${VOICE_MODEL}.onnx.json"
VOICE_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium"

if [ ! -f "$VOICE_ONNX" ]; then
  echo "  Downloading voice model: ${VOICE_MODEL}..."
  curl -L --progress-bar "${VOICE_BASE_URL}/en_US-amy-medium.onnx" -o "$VOICE_ONNX"
  curl -L --progress-bar "${VOICE_BASE_URL}/en_US-amy-medium.onnx.json" -o "$VOICE_JSON"
  echo "  ✓ Voice model downloaded"
else
  echo "  ✓ Voice model already present — skipping"
fi

# ── 5. Ollama model check ─────────────────────────────────────────────────────
echo ""
echo "▶ Checking Ollama for qwen3:30b..."
if ollama list 2>/dev/null | grep -q "qwen3:30b"; then
  echo "  ✓ qwen3:30b found"
else
  echo "  ⚠ qwen3:30b not found in ollama list."
  echo "  Run: ollama pull qwen3:30b"
  echo "  (Skipping — you can pull it manually and rerun if needed)"
fi

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup complete!                                       ║"
echo "║                                                          ║"
echo "║  To start your interview session:                        ║"
echo "║    source venv/bin/activate                              ║"
echo "║    python main.py                                        ║"
echo "║                                                          ║"
echo "║  To index your study documents:                          ║"
echo "║    python main.py --index                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
