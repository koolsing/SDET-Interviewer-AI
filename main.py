"""
main.py — SDET Interview Coach
Entry point for the voice-based interview practice application.

Usage:
    python main.py              # start an interview session
    python main.py --index      # index documents in resources/
    python main.py --list-docs  # show indexed documents per topic
"""
import sys
import os
import time
import argparse
import threading

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.text import Text
from rich import box
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns

from config import ROUNDS, TOPICS, ROUND_TOPICS, DEFAULT_MODEL, rating_label
from llm import OllamaClient, get_client
from interview_engine import InterviewSession
from tts import speak, piper_available
from stt import listen_and_transcribe
from knowledge_base import index_all, list_docs, topic_has_docs

console = Console()

# ── ASCII Banner ──────────────────────────────────────────────────────────────
BANNER = r"""
  _____ ____  _____ _____   ___       _                  _
 / ____|  _ \| ____|_   _| |_ _|_ __ | |_ ___ _ ____   _(_) _____      __
 \___ \| | | |  _|   | |    | || '_ \| __/ _ \ '__\ \ / / |/ _ \ \ /\ / /
  ___) | |_| | |___  | |    | || | | | ||  __/ |   \ V /| |  __/\ V  V /
 |____/|____/|_____| |_|   |___|_| |_|\__\___|_|    \_/ |_|\___| \_/\_/
                     C O A C H
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_banner():
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    console.print(
        "[dim]  100% local • no data stored • powered by Whisper + Piper + Ollama[/dim]\n"
    )

def _spinner_while(fn, label: str):
    """Run fn() in a background thread while showing a spinner. Returns fn's result."""
    result = {}
    def _run():
        result["value"] = fn()

    t = threading.Thread(target=_run, daemon=True)
    with console.status(f"[bold yellow]{label}[/bold yellow]", spinner="dots"):
        t.start()
        t.join()
    return result.get("value")

def _format_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"

def _clean_response(text: str) -> str:
    """
    Strip Qwen3's internal <think>…</think> reasoning blocks and the
    INTERVIEW_COMPLETE sentinel from any LLM response before displaying
    or speaking it.
    """
    import re
    # Remove complete <think>...</think> blocks (including multi-line)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If the model started thinking without an opening tag, remove everything up to </think>
    text = re.sub(r"^.*?</think>\n*", "", text, flags=re.DOTALL)
    # Remove any remaining stray </think> (just in case)
    text = text.replace("</think>", "")
    # Remove the interview-complete sentinel
    text = text.replace("INTERVIEW_COMPLETE", "")
    return text.strip()

# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_index():
    console.print("\n[bold yellow]▶ Indexing documents in resources/...[/bold yellow]\n")
    # Trigger sentence-transformer download on first use
    from knowledge_base import _get_sentence_transformer
    _spinner_while(_get_sentence_transformer, "Loading embedding model (first run may download ~80 MB)…")

    results = {}
    for slug in TOPICS:
        count = _spinner_while(
            lambda s=slug: __import__("knowledge_base").index_topic(s, verbose=False),
            f"Indexing [{TOPICS[slug]['label']}]…"
        )
        if count:
            results[slug] = count

    if results:
        table = Table(title="Indexed Documents", box=box.ROUNDED, style="green")
        table.add_column("Topic", style="bold")
        table.add_column("Chunks", justify="right")
        for slug, count in results.items():
            table.add_row(TOPICS[slug]["label"], str(count))
        console.print(table)
    else:
        console.print("[yellow]No documents found. Drop .pdf/.txt/.md files into resources/<topic>/ and retry.[/yellow]")


def cmd_list_docs():
    docs = list_docs()
    if not docs:
        console.print("[yellow]No documents indexed yet. Run: python main.py --index[/yellow]")
        return

    table = Table(title="Indexed Documents by Topic", box=box.ROUNDED)
    table.add_column("Topic", style="bold cyan")
    table.add_column("Files")
    for slug, files in docs.items():
        table.add_row(TOPICS[slug]["label"], "\n".join(files))
    console.print(table)

# ── Setup wizard ──────────────────────────────────────────────────────────────

def _select_model(client: OllamaClient) -> str:
    models = client.list_models()
    if not models:
        console.print("[red]✗ No models found in Ollama. Is Ollama running?[/red]")
        console.print("  Start it with: [bold]ollama serve[/bold]")
        sys.exit(1)

    table = Table(title="Available Ollama Models", box=box.SIMPLE_HEAD)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model", style="bold")
    table.add_column("Default", justify="center")
    for i, m in enumerate(models, 1):
        default_mark = "✓" if m == DEFAULT_MODEL or (DEFAULT_MODEL not in models and i == 1) else ""
        table.add_row(str(i), m, f"[green]{default_mark}[/green]")
    console.print(table)

    choice = Prompt.ask(
        "Select model number (or press Enter for default)",
        default="1" if DEFAULT_MODEL not in models else str(models.index(DEFAULT_MODEL) + 1),
    )
    try:
        idx = int(choice) - 1
        selected = models[idx]
    except (ValueError, IndexError):
        selected = models[0]

    console.print(f"[green]✓ Using model:[/green] [bold]{selected}[/bold]\n")
    return selected


def _select_round() -> str:
    console.print(Panel.fit(
        "\n".join(
            f"  [bold]{i}.[/bold] {info['name']} — {info['description']}"
            for i, (key, info) in enumerate(ROUNDS.items(), 1)
        ),
        title="[bold cyan]Interview Rounds[/bold cyan]",
        border_style="cyan",
    ))
    round_keys = list(ROUNDS.keys())
    choice = IntPrompt.ask("Select round", default=1)
    selected = round_keys[min(max(choice, 1), len(round_keys)) - 1]
    console.print(f"[green]✓ Round:[/green] [bold]{ROUNDS[selected]['name']}[/bold]\n")
    return selected


def _select_topic(round_key: str) -> str:
    eligible_slugs = ROUND_TOPICS.get(round_key, list(TOPICS.keys()))
    rows = []
    for i, slug in enumerate(eligible_slugs, 1):
        info = TOPICS[slug]
        doc_mark = "[green]📄[/green]" if topic_has_docs(slug) else "[dim]—[/dim]"
        rows.append(f"  [bold]{i}.[/bold] {info['label']}  {doc_mark}")

    console.print(Panel.fit(
        "\n".join(rows) + "\n\n[dim]📄 = you have study docs indexed for this topic[/dim]",
        title="[bold cyan]Interview Topic[/bold cyan]",
        border_style="cyan",
    ))
    choice = IntPrompt.ask("Select topic", default=1)
    selected = eligible_slugs[min(max(choice, 1), len(eligible_slugs)) - 1]
    console.print(f"[green]✓ Topic:[/green] [bold]{TOPICS[selected]['label']}[/bold]\n")
    return selected


def _select_duration(round_key: str) -> int:
    durations = ROUNDS[round_key]["durations"]
    if len(durations) == 1:
        console.print(f"[green]✓ Duration:[/green] [bold]{durations[0]} minutes[/bold]\n")
        return durations[0]

    opts = " / ".join(f"[bold]{d}[/bold]" for d in durations)
    console.print(f"Available durations: {opts} minutes")
    choice = IntPrompt.ask("Select duration", choices=[str(d) for d in durations], default=str(durations[0]))
    console.print(f"[green]✓ Duration:[/green] [bold]{choice} minutes[/bold]\n")
    return int(choice)

# ── Interview loop ────────────────────────────────────────────────────────────

def _print_timer(session: InterviewSession):
    remaining = session.time_remaining_secs()
    elapsed = session.elapsed_secs()
    color = "red" if remaining < 120 else "yellow" if remaining < 300 else "green"
    console.print(
        f"[dim]  ⏱ Elapsed: {_format_time(elapsed)}  |  "
        f"[{color}]Remaining: {_format_time(remaining)}[/{color}][/dim]"
    )

def _show_question(text: str):
    clean = _clean_response(text)
    if clean:
        console.print(Panel(
            clean,
            title="[bold yellow]🎙 Interviewer[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))

def _get_candidate_answer(session: InterviewSession) -> str | None:
    """
    Record the candidate's answer via mic.
    Returns the transcribed text, or None if the user typed 'quit'/'skip'.
    """
    console.print(
        "\n[bold green]  Your turn:[/bold green] "
        "[dim]Speak your answer, then stay silent for 7 seconds to finish.[/dim]"
    )
    console.print(
        "[dim]  Or type: [bold]done[/bold]=finish speaking  "
        "[bold]skip[/bold]=skip question  [bold]quit[/bold]=end session[/dim]\n"
    )

    piper_ok = piper_available()
    answer_text = None

    # Start mic recording in background, allow keyboard override
    from multiprocessing.pool import ThreadPool as _Pool
    from queue import Queue

    q: Queue = Queue()
    stop_event = threading.Event()
    typed_command = None

    def _mic_task():
        try:
            text = listen_and_transcribe(stop_event=stop_event)
            q.put(("mic", text))
        except Exception as e:
            q.put(("error", str(e)))

    mic_thread = threading.Thread(target=_mic_task, daemon=True)

    with console.status("[bold green]🎙 Recording…[/bold green]", spinner="bouncingBar"):
        mic_thread.start()

        import select
        # Allow user to type an override while mic is running
        # We poll stdin using select so the spinner keeps rendering
        while mic_thread.is_alive():
            i, _, _ = select.select([sys.stdin], [], [], 0.1)
            if i:
                line = sys.stdin.readline().strip().lower()
                if line in ("quit", "exit", "end", "skip", "next", "done"):
                    typed_command = line
                    stop_event.set()
                    break

        mic_thread.join()   # wait for mic (silence detection stops it via stop_event)

    if not q.empty():
        kind, value = q.get()
        if kind == "mic":
            answer_text = value.strip()
        else:
            console.print(f"[red]Mic error: {value}[/red]")
            answer_text = ""
    else:
        answer_text = ""

    # Handle typed commands first
    if typed_command in ("quit", "exit", "end"):
        return None
    if typed_command in ("skip", "next"):
        return "__SKIP__"

    # Also check for keyboard commands embedded in transcription (if they spoke it instead of typing)
    if answer_text.lower() in ("quit", "exit", "end"):
        return None
    if answer_text.lower() in ("skip", "next"):
        return "__SKIP__"

    # Show what was transcribed
    if answer_text:
        console.print(Panel(
            f"[italic]{answer_text}[/italic]",
            title="[bold blue]🗣 You said[/bold blue]",
            border_style="blue",
            padding=(0, 2),
        ))
    else:
        if typed_command == "done":
            # They typed 'done' but didn't speak anything
            console.print(Panel(
                f"[italic](Typed: done)[/italic]",
                title="[bold blue]🗣 You said[/bold blue]",
                border_style="blue",
                padding=(0, 2),
            ))
            return "I have no more to add."
        else:
            console.print("[dim]  (no speech detected — skipping)[/dim]")
            return "__SKIP__"

    return answer_text


def run_interview(session: InterviewSession):
    tts_ok = piper_available()

    if not tts_ok:
        console.print("[yellow]⚠ Piper TTS not found — text-only mode (run setup.sh to enable voice)[/yellow]\n")

    # Opening question
    with console.status("[bold yellow]🤖 Starting your interview…[/bold yellow]", spinner="dots"):
        opening = session.start()

    opening_clean = _clean_response(opening)
    _show_question(opening_clean)
    _print_timer(session)
    if tts_ok:
        speak(opening_clean)

    # Main loop
    while not session.finished and not session.is_time_up():
        answer = _get_candidate_answer(session)

        if answer is None:
            # User wants to quit
            console.print("\n[bold yellow]Ending session early…[/bold yellow]")
            break

        if answer == "__SKIP__":
            console.print("[dim]  Question skipped.[/dim]")
            # Still need to submit something to keep conversation going
            answer = "I'd like to move on to the next question, please."

        # Get next question
        with console.status("[bold yellow]🤖 Thinking…[/bold yellow]", spinner="dots"):
            response = session.submit_answer(answer)

        response_clean = _clean_response(response)
        if not response_clean and not session.finished:
            # Fallback for models that might return empty strings if confused
            response_clean = "I see. Let's move on. Can you tell me more about your experience with this topic?"
        
        if session.is_time_up() or session.finished:
            _show_question(response_clean)
            if tts_ok:
                speak(response_clean)
            break

        _show_question(response_clean)
        _print_timer(session)
        if tts_ok:
            speak(response_clean)

    if session.is_time_up() and not session.finished:
        msg = "Time's up! Thank you for your time today. The interview is now complete."
        console.print(Panel(msg, title="[bold yellow]🎙 Interviewer[/bold yellow]", border_style="yellow"))
        if tts_ok:
            speak(msg)

# ── Feedback report renderer ──────────────────────────────────────────────────

def _render_feedback(report):
    console.print("\n")
    console.rule("[bold cyan]📋 Interview Feedback Report[/bold cyan]")
    console.print(
        f"\n  [bold]Round:[/bold] {report.round_name}  |  "
        f"[bold]Topic:[/bold] {report.topic_label}  |  "
        f"[bold]Duration:[/bold] {report.duration_mins} min\n"
    )

    # Overall score
    label = rating_label(report.overall_score)
    score_color = (
        "red" if report.overall_score <= 4
        else "yellow" if report.overall_score <= 6
        else "green"
    )
    console.print(Panel(
        f"[{score_color}][bold]{report.overall_score}/10[/bold][/{score_color}]  —  {label}",
        title="[bold]Overall Score[/bold]",
        border_style=score_color,
    ))

    # Topic scores
    if report.topic_scores:
        table = Table(title="Scores by Skill Area", box=box.ROUNDED)
        table.add_column("Skill Area", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Rating")
        for skill, score in report.topic_scores.items():
            sc = "green" if score >= 7 else "yellow" if score >= 5 else "red"
            table.add_row(skill, f"[{sc}]{score}/10[/{sc}]", rating_label(score))
        console.print(table)

    # Strengths
    console.print(Panel(
        "\n".join(f"  ✅ {s}" for s in report.strengths),
        title="[bold green]Strengths[/bold green]",
        border_style="green",
    ))

    # Areas for improvement
    console.print(Panel(
        "\n".join(f"  🔧 {s}" for s in report.improvements),
        title="[bold yellow]Areas for Improvement[/bold yellow]",
        border_style="yellow",
    ))

    # Recommendations
    if report.recommendations:
        console.print(Panel(
            report.recommendations,
            title="[bold cyan]Recommendations[/bold cyan]",
            border_style="cyan",
        ))

    # Transcript summary
    console.print(f"\n[dim]  Questions covered: {len(report.turns)}[/dim]")
    console.rule()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SDET Interview Coach")
    parser.add_argument("--index", action="store_true", help="Index documents in resources/")
    parser.add_argument("--list-docs", action="store_true", help="List indexed documents")
    args = parser.parse_args()

    _print_banner()

    if args.index:
        cmd_index()
        return

    if args.list_docs:
        cmd_list_docs()
        return

    # ── Check Ollama ──────────────────────────────────────────────────────────
    tmp_client = OllamaClient()
    if not tmp_client.is_running():
        console.print("[red]✗ Ollama is not running.[/red]")
        console.print("  Start it in another terminal: [bold]ollama serve[/bold]")
        sys.exit(1)

    # ── Setup wizard ──────────────────────────────────────────────────────────
    console.rule("[bold cyan]Session Setup[/bold cyan]")

    selected_model = _select_model(tmp_client)
    client = get_client(selected_model)
    client.switch_model(selected_model)

    round_key = _select_round()
    topic_slug = _select_topic(round_key)
    duration_mins = _select_duration(round_key)

    # ── Confirm ───────────────────────────────────────────────────────────────
    doc_note = (
        "[green]📄 Study docs loaded[/green]" if topic_has_docs(topic_slug)
        else "[dim]No study docs (add to resources/ and run --index)[/dim]"
    )
    console.print(Panel(
        f"  Model:     [bold]{selected_model}[/bold]\n"
        f"  Round:     [bold]{ROUNDS[round_key]['name']}[/bold]\n"
        f"  Topic:     [bold]{TOPICS[topic_slug]['label']}[/bold]\n"
        f"  Duration:  [bold]{duration_mins} minutes[/bold]\n"
        f"  Docs:      {doc_note}\n"
        f"  TTS:       {'[green]Piper ready[/green]' if piper_available() else '[yellow]Text-only (run setup.sh)[/yellow]'}",
        title="[bold]Ready to start?[/bold]",
        border_style="cyan",
    ))

    go = Prompt.ask("Start interview?", choices=["yes", "no"], default="yes")
    if go != "yes":
        console.print("[dim]Cancelled.[/dim]")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    session = InterviewSession(round_key, topic_slug, duration_mins, client)
    console.rule("[bold green]Interview Started[/bold green]")
    run_interview(session)

    # ── Feedback ──────────────────────────────────────────────────────────────
    if session.turns:
        console.rule("[bold cyan]Generating Feedback…[/bold cyan]")
        with console.status("[bold yellow]Analysing your answers…[/bold yellow]", spinner="dots"):
            report = session.generate_feedback()
        _render_feedback(report)
    else:
        console.print("[dim]No answers recorded — no feedback to generate.[/dim]")

    console.print("\n[bold cyan]Good luck with your interviews! 🚀[/bold cyan]\n")


if __name__ == "__main__":
    main()
