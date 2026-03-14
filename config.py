"""
config.py — Central configuration for SDET Interview Coach
"""
import os

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
INDEX_DIR = os.path.join(RESOURCES_DIR, ".index")
PIPER_BIN = os.path.join(BASE_DIR, "piper", "piper")
VOICE_MODEL = os.path.join(BASE_DIR, "voices", "en_US-amy-medium.onnx")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:30b"

# ── Whisper ───────────────────────────────────────────────────────────────────
WHISPER_MODEL = "base.en"   # tiny.en / base.en / small.en / medium.en
WHISPER_DEVICE = "auto"     # auto → uses MPS on Apple Silicon

# ── Audio recording ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01    # RMS energy below this = silence
SILENCE_DURATION = 7.0      # seconds of silence before stopping recording
MAX_RECORD_SECONDS = 120    # safety cap per answer

# ── Interview rounds ──────────────────────────────────────────────────────────
ROUNDS = {
    "HR": {
        "name": "HR Round",
        "durations": [20],
        "description": "Behavioural, motivation, career goals, communication",
    },
    "Technical": {
        "name": "Technical Round",
        "durations": [30, 45],
        "description": "Hands-on testing skills, tools, frameworks, coding",
    },
    "Managerial": {
        "name": "Managerial Round",
        "durations": [30, 45],
        "description": "Leadership, delivery, cross-team collaboration, strategy",
    },
    "CTO": {
        "name": "CTO Round",
        "durations": [30, 45],
        "description": "Architecture, quality metrics, engineering culture",
    },
}

# ── Interview topics ──────────────────────────────────────────────────────────
TOPICS = {
    "agile-methodology": {
        "label": "Agile Methodology",
        "rounds": ["Technical", "Managerial", "CTO"],
        "keywords": ["scrum", "kanban", "sprint", "agile", "retrospective", "velocity"],
    },
    "api-testing": {
        "label": "API Testing",
        "rounds": ["Technical", "CTO"],
        "keywords": ["REST", "SOAP", "Postman", "RestAssured", "HTTP", "JSON", "status codes"],
    },
    "ci-cd": {
        "label": "CI/CD",
        "rounds": ["Technical", "Managerial", "CTO"],
        "keywords": ["Jenkins", "GitHub Actions", "pipeline", "Docker", "deployment", "CI/CD"],
    },
    "cucumber-bdd": {
        "label": "Cucumber BDD",
        "rounds": ["Technical"],
        "keywords": ["Gherkin", "Feature", "Scenario", "step definitions", "BDD", "Cucumber"],
    },
    "hr-round": {
        "label": "HR Round",
        "rounds": ["HR"],
        "keywords": ["strength", "weakness", "motivation", "teamwork", "conflict", "goals"],
    },
    "java-selenium": {
        "label": "Java & Selenium",
        "rounds": ["Technical"],
        "keywords": ["Selenium", "WebDriver", "TestNG", "JUnit", "Page Object", "Java", "locators"],
    },
    "manual-testing": {
        "label": "Manual Testing",
        "rounds": ["Technical", "Managerial"],
        "keywords": ["test case", "test plan", "bug report", "STLC", "smoke", "regression", "UAT"],
    },
    "sql": {
        "label": "SQL",
        "rounds": ["Technical"],
        "keywords": ["SELECT", "JOIN", "index", "stored procedure", "query", "database", "SQL"],
    },
    "sdet-common": {
        "label": "SDET Common Interview Questions",
        "rounds": ["Technical", "Managerial", "CTO"],
        "keywords": ["SDET", "automation", "framework design", "shift-left", "quality"],
    },
}

# Reverse map: round → eligible topics
ROUND_TOPICS = {}
for slug, meta in TOPICS.items():
    for r in meta["rounds"]:
        ROUND_TOPICS.setdefault(r, []).append(slug)

# ── Piper audio format ────────────────────────────────────────────────────────
PIPER_SAMPLE_RATE = 22050
PIPER_CHANNELS = 1

# ── Feedback rating scale ─────────────────────────────────────────────────────
RATING_LABELS = {
    (1, 3): "Needs Significant Work",
    (4, 5): "Below Expectations",
    (6, 7): "Meets Expectations",
    (8, 9): "Strong Performance",
    (10, 10): "Outstanding",
}

def rating_label(score: int) -> str:
    for (lo, hi), label in RATING_LABELS.items():
        if lo <= score <= hi:
            return label
    return "N/A"
