# Движок GigaAM через transcribe.cpp — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Движок `gigaam` в Python-воркере VoiceSwitch считает через transcribe.cpp (GGUF Q8_0 на Metal) вместо torch: 0,03–0,15 с на фразу и ~400 МБ памяти вместо 0,5–1 с и 2 ГБ.

**Architecture:** Протокол JSON Lines между Swift и `worker/asr_worker.py` не меняется. Внутри воркера класс `TranscribeCppGigaAM` грузит `Runtime/native/libtranscribe.dylib` через биндинг `transcribe-cpp` и модель `Runtime/Models/gigaam/gigaam-v3-e2e-rnnt-Q8_0.gguf`, читает WAV напрямую и режет длинные записи чистой функцией `split_pcm`. Установщик `install_runtime.sh` качает dylib и GGUF с проверкой sha256. Swift проверяет наличие GGUF, чтобы предложить доустановку, и передаёт путь к библиотеке воркеру.

**Tech Stack:** Python 3.12 (venv рантайма, `uv`), numpy, `transcribe-cpp==0.2.3` + `transcribe-native-0.2.3-macos-arm64-metal`, Swift 5.10 / SwiftPM, zsh.

## Global Constraints

- Версия биндинга и dylib одна: `TRANSCRIBE_CPP_VERSION="0.2.3"` в установщике; биндинг сверяет контракт и откажется работать с другой.
- Модель: `gigaam-v3-e2e-rnnt-Q8_0.gguf`, 273 724 832 байт, sha256 `78d63b47723b7f8d78c6113a6ef983b5a86e2a86f6c273e1f5cb6967b1c4467a`.
- Архив dylib: `transcribe-native-0.2.3-macos-arm64-metal.tar.gz`, sha256 `1cc5e89d442f55c165a3f90e49090cec75cec349071d834c0aa656161afa9543`, внутри каталог `transcribe-native-macos-arm64-metal/` с пятью dylib и `contract.json`.
- Лимит модели 25 с на фразу: чанки ≤ 22 с, без резки при записи ≤ 24 с (как в прежнем `split_wav_for_gigaam`).
- `expectedRuntimeVersion` в Swift остаётся 4.
- Тексты для человека — по правилам копирайта MILKY не требуется (это личный инструмент), но сообщения воркера на русском, как остальные.
- Тесты воркера запускаются `python3 -m unittest discover -s worker/tests -v` (python из venv рантайма или любой с numpy + transcribe_cpp).

---

### Task 1: `split_pcm` — резка массива на чанки (TDD)

**Files:**
- Modify: `worker/asr_worker.py` (заменить `split_wav_for_gigaam`)
- Test: `worker/tests/test_split_pcm.py`, `worker/tests/__init__.py` (пустой)

**Interfaces:**
- Produces: `split_pcm(samples: np.ndarray, rate: int, *, maximum_seconds: float = 22.0, single_limit_seconds: float = 24.0, minimum_seconds: float = 8.0, search_seconds: float = 5.0, window_seconds: float = 0.12) -> list[np.ndarray]`

- [ ] **Step 1: Написать падающий тест**

```python
# worker/tests/test_split_pcm.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr_worker import split_pcm  # noqa: E402

RATE = 16000


def noisy(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, int(RATE * seconds)).astype(np.float32)


class SplitPcmTest(unittest.TestCase):
    def test_short_recording_is_not_split(self):
        samples = noisy(20.0)
        chunks = split_pcm(samples, RATE)
        self.assertEqual(len(chunks), 1)
        self.assertIs(chunks[0], samples)

    def test_long_recording_keeps_every_sample_in_order(self):
        samples = noisy(60.0)
        chunks = split_pcm(samples, RATE)
        self.assertGreater(len(chunks), 1)
        np.testing.assert_array_equal(np.concatenate(chunks), samples)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), int(RATE * 22.0))

    def test_boundary_prefers_silence(self):
        samples = noisy(40.0)
        quiet_start, quiet_end = int(RATE * 19.0), int(RATE * 19.5)
        samples[quiet_start:quiet_end] = 0.0
        chunks = split_pcm(samples, RATE)
        first_boundary = len(chunks[0])
        self.assertGreaterEqual(first_boundary, quiet_start)
        self.assertLessEqual(first_boundary, quiet_end)

    def test_first_boundary_not_before_minimum(self):
        samples = noisy(40.0)
        samples[: int(RATE * 5.0)] = 0.0  # тишина в начале не должна стать границей
        chunks = split_pcm(samples, RATE)
        self.assertGreaterEqual(len(chunks[0]), int(RATE * 8.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `cd ~/Developer/VoiceSwitch && "$HOME/Library/Application Support/VoiceSwitch/Runtime/venv/bin/python3" -m unittest discover -s worker/tests -v`
Expected: `ImportError: cannot import name 'split_pcm'`

- [ ] **Step 3: Реализовать `split_pcm`, удалить `split_wav_for_gigaam`**

```python
def split_pcm(
    samples: "np.ndarray",
    rate: int,
    *,
    maximum_seconds: float = 22.0,
    single_limit_seconds: float = 24.0,
    minimum_seconds: float = 8.0,
    search_seconds: float = 5.0,
    window_seconds: float = 0.12,
) -> list["np.ndarray"]:
    """Режет запись на куски ≤ maximum_seconds по самому тихому окну.

    Граница ищется в последних search_seconds перед жёстким пределом, но не
    раньше minimum_seconds от начала куска. Запись короче single_limit_seconds
    возвращается как есть — GigaAM принимает до 25 с.
    """
    import numpy as np

    total = len(samples)
    if total <= int(rate * single_limit_seconds):
        return [samples]

    maximum = int(rate * maximum_seconds)
    minimum = int(rate * minimum_seconds)
    search_span = int(rate * search_seconds)
    window = max(1, int(rate * window_seconds))

    boundaries = [0]
    start = 0
    while total - start > maximum:
        hard_end = min(total, start + maximum)
        search_start = max(start + minimum, hard_end - search_span)
        best_end = hard_end
        best_energy = float("inf")
        for candidate in range(search_start, hard_end, window):
            segment = samples[candidate : min(candidate + window, hard_end)]
            if segment.size == 0:
                continue
            energy = float(np.mean(np.abs(segment)))
            if energy < best_energy:
                best_energy = energy
                best_end = candidate + segment.size // 2
        boundaries.append(best_end)
        start = best_end
    boundaries.append(total)
    return [samples[left:right] for left, right in zip(boundaries, boundaries[1:])]
```

- [ ] **Step 4: Прогнать тесты — зелёные**

Run: та же команда. Expected: `OK` (4 теста).

- [ ] **Step 5: Коммит**

```bash
git add worker/asr_worker.py worker/tests/__init__.py worker/tests/test_split_pcm.py
git commit -m "Резка длинных записей GigaAM по массиву вместо временных файлов"
```

### Task 2: Движок `TranscribeCppGigaAM` в воркере + smoke-тест

**Files:**
- Modify: `worker/asr_worker.py` (класс `Recognizer`, ветка `gigaam`)
- Test: `worker/tests/test_worker_smoke.py`

**Interfaces:**
- Consumes: `split_pcm` из Task 1.
- Produces: `read_wav_pcm(path: Path) -> tuple[np.ndarray, int]`; `TranscribeCppGigaAM(cache_root: Path)` с `load()`, `transcribe(audio: Path) -> str`, атрибутом `backend: str`; `TranscribeCppGigaAM.library_path(cache_root) -> Path`, `.model_path() -> Path`; константы `GIGAAM_GGUF = "gigaam-v3-e2e-rnnt-Q8_0.gguf"`, `GIGAAM_MAX_SECONDS = 22.0`. Ответ `result` получает поле `backend`.

- [ ] **Step 1: Написать smoke-тест (пропускается без рантайма)**

```python
# worker/tests/test_worker_smoke.py
"""Поднимает настоящий воркер на настоящей модели и проверяет ответ.

Пропускается, если рантайм не установлен. Пути можно переопределить:
VOICESWITCH_RUNTIME (корень Runtime), VOICESWITCH_PYTHON (python с
transcribe_cpp), TRANSCRIBE_LIBRARY (dylib).
"""
import json
import os
import re
import subprocess
import sys
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
            subprocess.run(["say", "-v", "Milena", "-o", str(aiff), PHRASE], check=True, capture_output=True)
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
```

- [ ] **Step 2: Убедиться, что тест падает по существу**

Подготовить рантайм-макет из scratchpad (dylib, GGUF и venv с transcribe_cpp уже скачаны):

```bash
S=/private/tmp/claude-501/-Users-rudolf-milky/1a9ccb7f-4c19-4873-a831-57220d75e4cd/scratchpad
mkdir -p $S/runtime/Models/gigaam
ln -sfn $S/tnative/transcribe-native-macos-arm64-metal $S/runtime/native
ln -sf $S/models/gigaam-v3-e2e-rnnt-Q8_0.gguf $S/runtime/Models/gigaam/
ln -sfn $S/tvenv $S/runtime/venv
cd ~/Developer/VoiceSwitch && VOICESWITCH_RUNTIME=$S/runtime $S/tvenv/bin/python -m unittest worker.tests.test_worker_smoke -v
```
Expected: FAIL — воркер отвечает ошибкой (старый движок пытается импортировать gigaam/torch, а в tvenv их нет), либо `KeyError: 'backend'`.

- [ ] **Step 3: Реализовать движок**

В `worker/asr_worker.py`: убрать `tempfile`, добавить константы и функции, заменить ветку gigaam в `Recognizer`.

```python
GIGAAM_GGUF = "gigaam-v3-e2e-rnnt-Q8_0.gguf"
GIGAAM_MAX_SECONDS = 22.0
INSTALL_HINT = "Запустите установку моделей из меню VoiceSwitch."


def read_wav_pcm(path: Path) -> tuple["np.ndarray", int]:
    """Читает mono 16-bit WAV в float32 [-1, 1] без ffmpeg."""
    import numpy as np

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError("Ожидается mono PCM WAV 16-bit.")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


class TranscribeCppGigaAM:
    """GigaAM v3 e2e RNNT в GGUF через transcribe.cpp (Metal, откат на CPU)."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.model: Any = None
        self.session: Any = None
        self.backend = ""

    @staticmethod
    def library_path(cache_root: Path) -> Path:
        override = os.environ.get("TRANSCRIBE_LIBRARY")
        if override:
            return Path(override)
        return cache_root.parent / "native" / "libtranscribe.dylib"

    def model_path(self) -> Path:
        return self.cache_root / "gigaam" / GIGAAM_GGUF

    def load(self) -> None:
        import numpy as np

        library = self.library_path(self.cache_root)
        if not library.is_file():
            raise RuntimeError(f"Не найдена библиотека transcribe.cpp: {library}. {INSTALL_HINT}")
        model_file = self.model_path()
        if not model_file.is_file():
            raise RuntimeError(f"Не найдена модель GigaAM (GGUF): {model_file}. {INSTALL_HINT}")
        os.environ["TRANSCRIBE_LIBRARY"] = str(library)

        import transcribe_cpp

        devices = list(transcribe_cpp.backends())
        gpu = next((d for d in devices if d.device_type == "gpu"), None)
        cpu = next((d for d in devices if d.device_type == "cpu"), None)
        model = None
        if gpu is not None:
            try:
                model = transcribe_cpp.Model(str(model_file), device=gpu)
                self.backend = gpu.kind
            except transcribe_cpp.TranscribeError as error:
                print(f"GigaAM: GPU недоступен ({error}), перехожу на CPU.", file=sys.stderr)
        if model is None:
            if cpu is None:
                raise RuntimeError("transcribe.cpp не нашёл ни одного вычислительного устройства.")
            model = transcribe_cpp.Model(str(model_file), device=cpu)
            self.backend = "cpu"
        self.model = model
        self.session = model.session()
        # Прогрев: первый вызов на Metal компилирует пайплайны (~0,3 с).
        self.session.run(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: Path) -> str:
        samples, rate = read_wav_pcm(audio)
        if rate != 16000:
            raise ValueError(f"Ожидается WAV 16 кГц, получено {rate} Гц.")
        texts: list[str] = []
        for chunk in split_pcm(samples, rate, maximum_seconds=GIGAAM_MAX_SECONDS):
            text = self.session.run(chunk).text.strip()
            if text:
                texts.append(text)
        return " ".join(texts).strip()
```

В `Recognizer`:
```python
        elif self.engine == "gigaam":
            emit("loading", engine=self.engine, message="Загрузка GigaAM v3 E2E RNNT (transcribe.cpp)…")
            self.model = TranscribeCppGigaAM(self.cache_root)
            self.model.load()
```
`transcribe()` для gigaam: `return self.model.transcribe(audio), "ru"`; метод `_transcribe_gigaam` удалить. Добавить `Recognizer.backend` → `getattr(self.model, "backend", "")` и в `serve()` передавать `backend=recognizer.backend` в `emit("result", ...)`; в `download()` — `emit("ready", engine=engine, backend=recognizer.backend, message=...)`. Константу `GIGAAM_MODEL` удалить.

- [ ] **Step 4: Прогнать оба теста — зелёные**

Run: `VOICESWITCH_RUNTIME=$S/runtime $S/tvenv/bin/python -m unittest discover -s worker/tests -v`
Expected: `OK` (5 тестов), smoke-тест не пропущен.

- [ ] **Step 5: Коммит**

```bash
git add worker/asr_worker.py worker/tests/test_worker_smoke.py
git commit -m "GigaAM через transcribe.cpp: Metal вместо torch, WAV без ffmpeg"
```

### Task 3: Установщик — dylib, биндинг, GGUF с проверкой sha256

**Files:**
- Modify: `Resources/install_runtime.sh` (переменные в шапке, `write_install_marker`, блок `gigaam`)

**Interfaces:**
- Produces: `Runtime/native/{libtranscribe,libggml,libggml-base,libggml-cpu,libggml-metal}.dylib`, `Runtime/native/contract.json`, `Runtime/native/version.txt`; `Runtime/Models/gigaam/gigaam-v3-e2e-rnnt-Q8_0.gguf`; пакет `transcribe-cpp==0.2.3` в venv; маркер `components/gigaam.ready`.

- [ ] **Step 1: Переменные в шапке вместо `GIGAAM_COMMIT`/`GIGAAM_ARCHIVE`**

```zsh
TRANSCRIBE_CPP_VERSION="0.2.3"
TRANSCRIBE_NATIVE_URL="https://github.com/handy-computer/transcribe.cpp/releases/download/v${TRANSCRIBE_CPP_VERSION}/transcribe-native-${TRANSCRIBE_CPP_VERSION}-macos-arm64-metal.tar.gz"
TRANSCRIBE_NATIVE_SHA256="1cc5e89d442f55c165a3f90e49090cec75cec349071d834c0aa656161afa9543"
NATIVE_ROOT="${RUNTIME_ROOT}/native"
GIGAAM_GGUF_NAME="gigaam-v3-e2e-rnnt-Q8_0.gguf"
GIGAAM_GGUF_URL="https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/main/${GIGAAM_GGUF_NAME}"
GIGAAM_GGUF_SHA256="78d63b47723b7f8d78c6113a6ef983b5a86e2a86f6c273e1f5cb6967b1c4467a"
```
В `write_install_marker`: строку `gigaam_commit=` заменить на `print -r -- "transcribe_cpp=${TRANSCRIBE_CPP_VERSION}"`.

- [ ] **Step 2: Хелпер и новый блок gigaam**

```zsh
verify_sha256() {
  local file=$1
  local expected=$2
  local actual
  actual=$(shasum -a 256 "${file}" | cut -d ' ' -f 1)
  [[ "${actual}" == "${expected}" ]]
}
```

```zsh
if component_requested gigaam; then
  mkdir -p "${NATIVE_ROOT}" "${MODEL_ROOT}/gigaam"

  if [[ ! -f "${NATIVE_ROOT}/libtranscribe.dylib" ]] || \
     [[ "$(cat "${NATIVE_ROOT}/version.txt" 2>/dev/null)" != "${TRANSCRIBE_CPP_VERSION}" ]]; then
    NATIVE_ARCHIVE="${TOOLS_ROOT}/transcribe-native-${TRANSCRIBE_CPP_VERSION}.tar.gz"
    if [[ -f "${NATIVE_ARCHIVE}" ]] && ! verify_sha256 "${NATIVE_ARCHIVE}" "${TRANSCRIBE_NATIVE_SHA256}"; then
      rm -f "${NATIVE_ARCHIVE}"
    fi
    if [[ ! -f "${NATIVE_ARCHIVE}" ]]; then
      download_with_resume \
        "${TRANSCRIBE_NATIVE_URL}" \
        "${NATIVE_ARCHIVE}" \
        "Загружаю библиотеку transcribe.cpp…"
    fi
    verify_sha256 "${NATIVE_ARCHIVE}" "${TRANSCRIBE_NATIVE_SHA256}" || {
      rm -f "${NATIVE_ARCHIVE}"
      fail "Архив transcribe.cpp повреждён при загрузке. Нажмите «Продолжить установку»."
    }
    status "Распаковываю transcribe.cpp…"
    tar -xzf "${NATIVE_ARCHIVE}" -C "${NATIVE_ROOT}" --strip-components 1 || \
      fail_step "Распаковываю transcribe.cpp"
    print -r -- "${TRANSCRIBE_CPP_VERSION}" > "${NATIVE_ROOT}/version.txt"
  fi

  run_with_retries \
    "Устанавливаю биндинг transcribe.cpp…" \
    3 \
    "${UV_EXECUTABLE}" pip install \
      --python "${PYTHON}" \
      "transcribe-cpp==${TRANSCRIBE_CPP_VERSION}" || fail_step "Устанавливаю биндинг transcribe.cpp"

  GIGAAM_GGUF="${MODEL_ROOT}/gigaam/${GIGAAM_GGUF_NAME}"
  if [[ -f "${GIGAAM_GGUF}" ]] && ! verify_sha256 "${GIGAAM_GGUF}" "${GIGAAM_GGUF_SHA256}"; then
    rm -f "${GIGAAM_GGUF}"
  fi
  if [[ ! -f "${GIGAAM_GGUF}" ]]; then
    download_with_resume \
      "${GIGAAM_GGUF_URL}" \
      "${GIGAAM_GGUF}" \
      "Загружаю GigaAM v3 E2E RNNT (261 МБ)…"
    status "Проверяю контрольную сумму модели…"
    verify_sha256 "${GIGAAM_GGUF}" "${GIGAAM_GGUF_SHA256}" || {
      rm -f "${GIGAAM_GGUF}"
      fail "Модель GigaAM повреждена при загрузке. Нажмите «Продолжить установку»."
    }
  fi

  run_with_retries \
    "Проверяю движок GigaAM…" \
    2 \
    /usr/bin/env "TRANSCRIBE_LIBRARY=${NATIVE_ROOT}/libtranscribe.dylib" \
      "${PYTHON}" "${ASR_WORKER}" \
        --download \
        --engine gigaam \
        --cache "${MODEL_ROOT}" || fail_step "Проверяю движок GigaAM"
  mark_component gigaam
fi
```
Старый блок (torchaudio, `GIGAAM_ARCHIVE`, проверка `import torch`) удалить целиком.

- [ ] **Step 3: Синтаксис и прогон на этом Маке**

```bash
zsh -n Resources/install_runtime.sh
R="$HOME/Library/Application Support/VoiceSwitch/Runtime"
zsh Resources/install_runtime.sh "$R" "$PWD/worker/asr_worker.py" "$PWD/worker/text_worker.py" gigaam 2>&1 | tail -15
ls -la "$R/native" "$R/Models/gigaam"; cat "$R/install-complete.txt"
```
Expected: статусы до «Проверяю движок GigaAM…», `ready`, в `native/` пять dylib, в `Models/gigaam/` GGUF, маркер с `transcribe_cpp=0.2.3` и `component=gigaam`.

- [ ] **Step 4: Smoke-тест на настоящем рантайме**

Run: `"$R/venv/bin/python3" -m unittest discover -s worker/tests -v`
Expected: `OK`, 5 тестов, без пропусков.

- [ ] **Step 5: Коммит**

```bash
git add Resources/install_runtime.sh
git commit -m "Установщик: transcribe.cpp и GGUF GigaAM вместо torch"
```

### Task 4: Swift — проверка GGUF, путь к библиотеке, размер загрузки

**Files:**
- Modify: `Sources/VoiceSwitch/RuntimePaths.swift` (`isInstalled`, новые пути)
- Modify: `Sources/VoiceSwitch/ASRService.swift` (окружение воркера в `ensureWorker`)
- Modify: `Sources/VoiceSwitch/Models.swift` (`downloadSize` для `.gigaam`)

**Interfaces:**
- Produces: `RuntimePaths.transcribeLibrary: URL` (`Runtime/native/libtranscribe.dylib`), `RuntimePaths.gigaamModel: URL` (`Runtime/Models/gigaam/gigaam-v3-e2e-rnnt-Q8_0.gguf`).

- [ ] **Step 1: RuntimePaths**

```swift
    static let gigaamModelFileName = "gigaam-v3-e2e-rnnt-Q8_0.gguf"

    static var nativeRoot: URL {
        runtimeRoot.appendingPathComponent("native", isDirectory: true)
    }

    static var transcribeLibrary: URL {
        nativeRoot.appendingPathComponent("libtranscribe.dylib")
    }

    static var gigaamModel: URL {
        modelCache
            .appendingPathComponent("gigaam", isDirectory: true)
            .appendingPathComponent(gigaamModelFileName)
    }

    /// GigaAM считается установленным, только если на месте и библиотека
    /// transcribe.cpp, и GGUF: после обновления со старого torch-рантайма
    /// маркер есть, а файлов нет — приложение должно предложить доустановку.
    static var gigaamFilesPresent: Bool {
        FileManager.default.fileExists(atPath: transcribeLibrary.path)
            && FileManager.default.fileExists(atPath: gigaamModel.path)
    }

    static func isInstalled(_ component: RuntimeComponent) -> Bool {
        guard installedComponents.contains(component) else { return false }
        if component == .gigaam {
            return gigaamFilesPresent
        }
        return true
    }
```

- [ ] **Step 2: ASRService.ensureWorker — окружение**

После `environment["PYTHONPATH"] = ...`:
```swift
        environment["TRANSCRIBE_LIBRARY"] = RuntimePaths.transcribeLibrary.path
```

- [ ] **Step 3: Models.swift**

`case .gigaam: return "≈ 0,3 ГБ + базовое окружение"`.

- [ ] **Step 4: Сборка**

Run: `swift build -c release 2>&1 | tail -3`
Expected: `Compiling…`, `Build complete!` без ошибок.

- [ ] **Step 5: Коммит**

```bash
git add Sources/VoiceSwitch/RuntimePaths.swift Sources/VoiceSwitch/ASRService.swift Sources/VoiceSwitch/Models.swift
git commit -m "GigaAM установлен, только если есть библиотека transcribe.cpp и GGUF"
```

### Task 5: TextInjector — без паузы, когда окно уже впереди

**Files:**
- Modify: `Sources/VoiceSwitch/TextInjector.swift` (`paste`, участок с `activate` и `focusRestoreDelay`)

- [ ] **Step 1: Правка**

Заменить
```swift
        targetApplication?.activate(options: [])

        DispatchQueue.main.asyncAfter(deadline: .now() + focusRestoreDelay) {
```
на
```swift
        // Если целевое окно уже впереди (обычный случай: диктовка в активное
        // поле), активировать нечего и ждать восстановления фокуса не нужно.
        let needsActivation = currentPID == nil
        if needsActivation {
            targetApplication?.activate(options: [])
        }
        let delay = needsActivation ? focusRestoreDelay : 0

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
```

- [ ] **Step 2: Сборка**

Run: `swift build -c release 2>&1 | tail -3` → `Build complete!`

- [ ] **Step 3: Коммит**

```bash
git add Sources/VoiceSwitch/TextInjector.swift
git commit -m "Вставка без паузы 0,35 с, когда целевое окно уже впереди"
```

### Task 6: README, сборка приложения, выкатка, проверка

**Files:**
- Modify: `README.md` (строки про размер установки и лицензии)
- Modify: `docs/superpowers/specs/2026-09-08-transcribe-cpp-engine-design.md` — не требуется

- [ ] **Step 1: README**

Заменить «около 2,5 ГБ для рекомендуемой установки с GigaAM» на «около 0,6 ГБ для рекомендуемой установки с GigaAM», «**«Русская диктовка»**: только GigaAM, около 2,5 ГБ» на «около 0,6 ГБ». В список лицензий после GigaAM добавить:
```markdown
- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) и [ggml](https://github.com/ggml-org/ggml) — MIT; GigaAM выполняется через них в формате GGUF;
```
В таблицу сравнения после строки GigaAM — сноску: «С 08.09.2026 GigaAM выполняется через transcribe.cpp на Metal: ~0,03–0,15 с на фразу».

- [ ] **Step 2: Собрать и установить приложение**

```bash
VOICESWITCH_CODESIGN_IDENTITY=- zsh scripts/build_app.sh dist/staging 2>&1 | tail -3
osascript -e 'tell application "VoiceSwitch" to quit' 2>/dev/null; sleep 1
rm -rf /Applications/VoiceSwitch.app && cp -R dist/staging/VoiceSwitch.app /Applications/
tccutil reset Accessibility io.github.mitimaicode.VoiceSwitch
open /Applications/VoiceSwitch.app
```
Expected: приложение запускается, показывает запрос универсального доступа (выдаёт Рудольф), в меню GigaAM отмечен установленным, воркер запущен.

- [ ] **Step 3: Проверить воркер в бою**

```bash
sleep 20; pgrep -fl asr_worker.py; top -l 1 -stats pid,command,mem,cmprs -pid $(pgrep -f asr_worker.py)
```
Expected: процесс воркера ≈ 350–450 МБ. После первой диктовки Рудольфа в `comparison.jsonl` строка `transcription` с `latency_seconds` < 0,3.

- [ ] **Step 4: Коммит и пуш**

```bash
git add README.md && git commit -m "README: GigaAM через transcribe.cpp, размер установки"
git push origin no-clipboard-clobber
```
