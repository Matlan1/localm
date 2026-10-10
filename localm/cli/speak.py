# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm speak``: turn text into a WAV file with a text-to-speech model."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click

from ._core import _complete_model_name, console, main

_PLAIN_PROGRESS_EVERY_S = 5.0


class _Progress:
    """Shows what ``speak`` is doing: a live status line on a terminal, a line
    every few seconds otherwise."""

    def __init__(self, status) -> None:
        self._status = status
        self._last_line: float | None = None

    def __call__(self, event: dict) -> None:
        stage = event.get("stage")
        if stage == "loading":
            text = "Loading the speech model..."
        elif stage == "waiting":
            text = "Waiting for another speech request..."
        else:
            text = f"Speaking: {float(event.get('seconds') or 0.0):.1f} s of audio"
        if self._status is not None:
            self._status.update(text)
            return
        if stage != "speaking":
            click.echo(text, err=True)
            return
        now = time.monotonic()
        if self._last_line is None or now - self._last_line >= _PLAIN_PROGRESS_EVERY_S:
            self._last_line = now
            click.echo(text, err=True)


@main.command("speak")
@click.argument("text", required=False)
@click.option("-o", "--output", "output", default=None,
              type=click.Path(dir_okay=False),
              help="The WAV file to write.")
@click.option("--file", "text_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Read the text to speak from a UTF-8 text file.")
@click.option("--model", "model", default=None, shell_complete=_complete_model_name,
              help="Registered text-to-speech model. Default: the only one.")
@click.option("--voice", "voice", default=None,
              help="'default', or the name of a WAV recording in the voices folder.")
@click.option("--voice-file", "voice_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="A WAV recording (at most 30 s) whose voice to imitate.")
@click.option("--language", "language", default=None,
              help="Language code such as 'en' or name such as 'english'. "
                   "Default: the model's own.")
@click.option("--seed", "seed", type=click.IntRange(0, 0xFFFFFFFE), default=None,
              help="Seed for a reproducible result. Default: random (printed).")
@click.option("--list-voices", "list_voices", is_flag=True,
              help="List the available voices and exit.")
def speak_cmd(text, output, text_file, model, voice, voice_file, language, seed,
              list_voices):
    """Speak TEXT into a WAV file with a text-to-speech model.

    Runs the model in this process (a Qwen3-TTS GGUF with its mmproj, added with
    `localm pull`). The server's POST /v1/audio/speech does the same. Named
    voices are WAV recordings saved as <name>.wav in the voices folder of the
    data directory.

    \b
    Example:
      localm speak "Hello there." -o hello.wav
    """
    from rich.markup import escape

    from ..inference import speech
    from ..inference.backends.base import PretokenizerUnsafeInputError
    from ..inference.backends.llamacpp import mtmd_gen

    if list_voices:
        for name in speech.list_voices():
            click.echo(name)
        console.print(f"[dim]Voices folder: {escape(str(speech.voices_dir()))}[/dim]")
        return
    if text_file:
        try:
            text = Path(text_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise click.ClickException(f"cannot read {text_file}: {e}") from e
    if not text or not text.strip():
        raise click.UsageError("give the text to speak as an argument or with --file")
    if not output:
        raise click.UsageError("name the WAV file to write with -o/--output")
    if voice_file and voice and voice != speech.DEFAULT_VOICE:
        raise click.UsageError("use either --voice or --voice-file, not both")
    try:
        mtmd_gen.resolve_language(language)
        target = speech.resolve_speech_model(model)
        if voice_file:
            reference = Path(voice_file).read_bytes()
        else:
            reference = speech.voice_reference(voice)
    except (speech.SpeechModelError, mtmd_gen.SpeechInputError, OSError) as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)

    status_ctx = console.status("Starting...") if console.is_terminal else None
    try:
        if status_ctx is not None:
            with status_ctx as status:
                out = speech.synthesize(target, text, language=language,
                                        reference_wav=reference, seed=seed,
                                        on_progress=_Progress(status))
        else:
            out = speech.synthesize(target, text, language=language,
                                    reference_wav=reference, seed=seed,
                                    on_progress=_Progress(None))
    except KeyboardInterrupt:
        speech.reset_speech()
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(130)
    except (mtmd_gen.SpeechInputError, PretokenizerUnsafeInputError,
            mtmd_gen.SpeechUnavailable, mtmd_gen.SpeechBudgetExceeded,
            speech.SpeechUnavailableError, RuntimeError) as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)

    dest = Path(output)
    tmp = dest.with_name(dest.name + ".part")
    try:
        tmp.write_bytes(out.wav)
        os.replace(tmp, dest)
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise click.ClickException(f"cannot write {output}: {e}") from e
    console.print(f"[green]Wrote[/green] {escape(str(dest))} "
                  f"({out.seconds:.2f} s of audio, seed {out.seed})")
