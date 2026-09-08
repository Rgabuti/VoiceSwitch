"""Поднимает настоящий воркер на настоящей модели и проверяет ответ.

Пропускается, если рантайм не установлен. Пути можно переопределить:
VOICESWITCH_RUNTIME (корень Runtime), VOICESWITCH_PYTHON (python с
transcribe_cpp), TRANSCRIBE_LIBRARY (dylib).
"""
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "asr_worker.py"
MARKER = "__VOICESWITCH_JSON__"
GGUF = "gigaam-v3-e2e-rnnt-Q8_0.gguf"
PHRASE = "Перенеси встречу с мастером на завтра и напиши клиенту."


def runtime_root() -> Path:
    custom = os.environ.get("VOICESWITCH_RUNTIME")
    if custom:
        return Path(custom)
    return Path.home() / "Library" / "Application Support" / "VoiceSwitch" / "Runtime"


class WorkerSmokeTest(unittest.TestCase):
    def setUp(self):
        root = runtime_root()
        self.library = Path(
            os.environ.get("TRANSCRIBE_LIBRARY") or root / "native" / "libtranscribe.dylib"
        )
        self.cache = root / "Models"
        self.python = os.environ.get("VOICESWITCH_PYTHON") or str(root / "venv" / "bin" / "python3")
        if not self.library.is_file() or not (self.cache / "gigaam" / GGUF).is_file():
            self.skipTest("рантайм transcribe.cpp не установлен")
        if not os.access(self.python, os.X_OK):
            self.skipTest("нет python рантайма")

    def make_wav(self) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="voiceswitch-test-"))
        aiff = directory / "phrase.aiff"
        wav = directory / "phrase.wav"
        try:
            subprocess.run(
                ["say", "-v", "Milena", "-o", str(aiff), PHRASE],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                check=True, capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("нет голоса Milena или afconvert")
        return wav

    def read_until(self, process, wanted: str, timeout_lines: int = 200) -> dict:
        for _ in range(timeout_lines):
            line = process.stdout.readline()
            if not line:
                self.fail("воркер закрылся раньше времени")
            if line.startswith(MARKER):
                message = json.loads(line[len(MARKER):])
                if message.get("type") == "error":
                    self.fail(f"воркер ответил ошибкой: {message.get('message')}")
                if message.get("type") == wanted:
                    return message
        self.fail(f"не дождались сообщения {wanted}")

    def test_transcribes_russian_phrase(self):
        wav = self.make_wav()
        env = {**os.environ, "TRANSCRIBE_LIBRARY": str(self.library), "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            [self.python, str(WORKER), "--serve", "--engine", "gigaam", "--cache", str(self.cache)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, env=env,
        )
        self.addCleanup(process.kill)
        self.addCleanup(process.stdout.close)
        self.addCleanup(process.stdin.close)

        ready = self.read_until(process, "ready")
        self.assertEqual(ready["engine"], "gigaam")

        request = {"id": "smoke-1", "audio": str(wav), "duration": 3.0, "prompt": ""}
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        result = self.read_until(process, "result")

        self.assertEqual(result["id"], "smoke-1")
        self.assertRegex(result["text"], re.compile(r"[А-Яа-яЁё]{3,}"))
        self.assertIn("встречу", result["text"].lower())
        self.assertLess(result["latency"], 2.0)
        self.assertIn(result["backend"], ("metal", "cpu"))


if __name__ == "__main__":
    unittest.main()
