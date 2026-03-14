"""
interview_engine.py — Core interview orchestration

Manages the full lifecycle of an interview session:
  - System prompt construction (round + topic + RAG context)
  - Turn-by-turn Q&A with timer enforcement
  - Post-interview feedback generation
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import ROUNDS, TOPICS, rating_label
from knowledge_base import retrieve, topic_has_docs
from llm import OllamaClient


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    question: str
    answer: str
    elapsed_secs: float  # time into interview when question was asked


@dataclass
class FeedbackReport:
    round_name: str
    topic_label: str
    duration_mins: int
    turns: List[Turn]
    overall_score: int
    strengths: List[str]
    improvements: List[str]
    topic_scores: Dict[str, int]
    recommendations: str
    raw_feedback: str


# ── System prompt builders ────────────────────────────────────────────────────

def _build_system_prompt(
    round_key: str,
    topic_slug: str,
    duration_mins: int,
    rag_context: str = "",
) -> str:
    round_info = ROUNDS[round_key]
    topic_info = TOPICS[topic_slug]
    total_questions_hint = max(3, duration_mins // 5)

    prompt = f"""You are a senior {round_info['name']} interviewer conducting a {duration_mins}-minute interview for an SDET (Software Development Engineer in Test) position.

Interview details:
- Round: {round_info['name']}
- Topic focus: {topic_info['label']}
- Duration: {duration_mins} minutes
- Approximate questions to cover: {total_questions_hint}

Your interviewing style:
- Professional, focused, and constructive
- Ask one clear question at a time
- Listen carefully to the candidate's answer before asking your next question
- Ask follow-up or counter questions when the answer is incomplete, vague, or shows potential worth exploring further
- Keep time in mind — as time runs low, prioritise the most important remaining questions
- Do NOT reveal scores or judgements during the interview itself

Topic expertise you are testing:
{topic_info['label']} — key areas: {", ".join(topic_info['keywords'])}

{round_info['description']}
"""

    if rag_context:
        prompt += f"""
Reference material (from the candidate's own study resources — use to ground your questions and later feedback):
---
{rag_context}
---
"""

    prompt += """
Interview rules:
1. Start by warmly greeting the candidate and asking your first question.
2. After each candidate response, either ask a follow-up or move to the next topic.
3. When you determine the session is ending, say exactly: "INTERVIEW_COMPLETE" on its own line — nothing else after it. Do NOT say this until the session is truly over.
4. Never break character.
"""
    return prompt


def _build_feedback_prompt(
    round_key: str,
    topic_slug: str,
    duration_mins: int,
    turns: List[Turn],
) -> str:
    topic_info = TOPICS[topic_slug]
    transcript = "\n\n".join(
        f"Q: {t.question}\nA: {t.answer}"
        for t in turns
    )
    return f"""You are an expert SDET interview coach reviewing a completed {duration_mins}-minute {ROUNDS[round_key]['name']} interview on the topic: {topic_info['label']}.

Full interview transcript:
---
{transcript}
---

Please provide a structured feedback report in the following EXACT JSON format (no prose outside the JSON):
{{
  "overall_score": <integer 1-10>,
  "topic_scores": {{
    "<skill_area>": <integer 1-10>
  }},
  "strengths": ["<point>", "<point>", ...],
  "improvements": ["<point>", "<point>", ...],
  "recommendations": "<2-3 sentences of actionable advice and resources>"
}}

Be honest, specific, and constructive. Reference exact answers where helpful.
"""


# ── Interview Session ─────────────────────────────────────────────────────────

class InterviewSession:
    def __init__(
        self,
        round_key: str,
        topic_slug: str,
        duration_mins: int,
        llm: OllamaClient,
    ):
        self.round_key = round_key
        self.topic_slug = topic_slug
        self.duration_mins = duration_mins
        self.llm = llm

        self.turns: List[Turn] = []
        self.messages: List[Dict] = []
        self.start_time: Optional[float] = None
        self.finished = False

        # Pre-fetch RAG context (topic-level overview query)
        topic_label = TOPICS[topic_slug]["label"]
        rag_chunks = retrieve(topic_slug, f"overview of {topic_label}", top_k=3)
        rag_context = "\n\n".join(rag_chunks) if rag_chunks else ""
        self.has_docs = topic_has_docs(topic_slug)

        system_prompt = _build_system_prompt(
            round_key, topic_slug, duration_mins, rag_context
        )
        self.messages.append({"role": "system", "content": system_prompt})

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> str:
        """Begin the session. Returns the AI's opening question."""
        self.start_time = time.time()
        # Llama 3.1 and other strict chat templates require a user message to start
        self.messages.append({
            "role": "user", 
            "content": "Hi there! I am ready to begin the interview."
        })
        response = self.llm.chat(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def elapsed_secs(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def elapsed_mins(self) -> float:
        return self.elapsed_secs() / 60

    def time_remaining_secs(self) -> float:
        return max(0.0, self.duration_mins * 60 - self.elapsed_secs())

    def is_time_up(self) -> bool:
        return self.elapsed_secs() >= self.duration_mins * 60

    def submit_answer(self, answer: str) -> str:
        """
        Submit a candidate answer. Returns the AI's next question or closing remark.
        If the AI signals INTERVIEW_COMPLETE, sets self.finished = True.
        """
        elapsed = self.elapsed_secs()
        remaining = self.time_remaining_secs()

        # Inject time-awareness into the user turn
        time_note = ""
        if remaining < 60:
            time_note = " [Note to interviewer: less than 1 minute remaining — wrap up now.]"
        elif remaining < 5 * 60:
            time_note = f" [Note to interviewer: ~{int(remaining/60)} minutes remaining.]"

        # Retrieve context relevant to this specific answer for the next question
        if self.has_docs:
            follow_up_chunks = retrieve(self.topic_slug, answer, top_k=2)
            if follow_up_chunks:
                ctx = "\n\n".join(follow_up_chunks)
                self.messages.append({
                    "role": "system",
                    "content": f"[Retrieved context for next question]\n{ctx}"
                })

        user_msg = answer + time_note
        self.messages.append({"role": "user", "content": user_msg})

        response = self.llm.chat(self.messages)
        self.messages.append({"role": "assistant", "content": response})

        # Log the turn
        self.turns.append(Turn(
            question=self.messages[-4]["content"] if len(self.messages) >= 4
                     else self.messages[1]["content"],
            answer=answer,
            elapsed_secs=elapsed,
        ))

        if "INTERVIEW_COMPLETE" in response:
            self.finished = True

        return response

    def generate_feedback(self) -> FeedbackReport:
        """Ask the LLM to evaluate all answers and return a FeedbackReport."""
        import json as _json

        feedback_prompt = _build_feedback_prompt(
            self.round_key, self.topic_slug, self.duration_mins, self.turns
        )
        raw = self.llm.chat([
            {"role": "system", "content": "You are an expert SDET interview coach."},
            {"role": "user", "content": feedback_prompt},
        ])

        # Parse JSON from the response
        try:
            # Strip code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            data = _json.loads(cleaned.strip())
        except Exception:
            # Fallback: return minimal report
            data = {
                "overall_score": 5,
                "topic_scores": {},
                "strengths": ["Could not parse detailed feedback."],
                "improvements": ["Please review the raw feedback below."],
                "recommendations": "Review the transcript manually.",
            }

        return FeedbackReport(
            round_name=ROUNDS[self.round_key]["name"],
            topic_label=TOPICS[self.topic_slug]["label"],
            duration_mins=self.duration_mins,
            turns=self.turns,
            overall_score=data.get("overall_score", 5),
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
            topic_scores=data.get("topic_scores", {}),
            recommendations=data.get("recommendations", ""),
            raw_feedback=raw,
        )
