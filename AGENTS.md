# VoiceSwitch — форк Рудольфа (контекст для Claude Code и Kimi Code)

Диктовка на Маке вместо Wispr Flow: приложение строки меню, push-to-talk на fn, распознавание
русской речи GigaAM v3 e2e RNNT локально и офлайн. Апстрим — `mitimaicode/VoiceSwitch`
(v0.3.3-beta), наш форк — `github.com/Rgabuti/VoiceSwitch`, ветка `no-clipboard-clobber`.
**PR апстриму не отправлять** (решение 28.08.2026, не предлагать снова). Это личный инструмент,
не проект MILKY: спеки, планы и заметки живут здесь, в `docs/superpowers/`; межпроектный
указатель — в `~/milky/AGENTS.md`, раздел «Диктовка на Маке».

## Состояние на 08.09.2026

**В бою:** `/Applications/VoiceSwitch.app` собран из HEAD `18752f3` (v0.3.3 + 23 коммита форка),
ad-hoc подпись. Рантайм — `~/Library/Application Support/VoiceSwitch/Runtime` (2,3 ГБ, из них
~1,8 ГБ — хвосты старого движка, см. ниже). Движок GigaAM с 08.09.2026 работает через
transcribe.cpp 0.2.3 (GGUF Q8_0, Metal): семь фраз после перехода — 0,05–0,22 с при прежней
медиане 1,08 с; воркер ~330 МБ вместо 2 ГБ, поэтому провалов на 5–7 с после пауз больше нет.

Хронология форка:
- 28.08.2026 — буфер обмена не затирается (прямая AX-вставка мимо буфера, фолбэк ⌘V
  с восстановлением содержимого через 0,4 с); push-to-talk на одиночном fn вместо переключателя
  fn+Option; словарь замен `replacements.txt`; запрос универсального доступа при старте
  (иначе до него не добраться — весь UI в попапе строки меню).
- 08.09.2026 — движок GigaAM переведён с torch на transcribe.cpp; установщик качает dylib и
  GGUF с проверкой sha256; `RuntimePaths.isInstalled(.gigaam)` требует наличия dylib и GGUF;
  вставка без паузы 0,35 с, когда целевое окно уже впереди; прогрев воркера при запуске;
  тесты `worker/tests/`. Спека — `docs/superpowers/specs/2026-09-08-transcribe-cpp-engine-design.md`
  (там же таблица замеров: почему не гибрид MPS, не ONNX и не sherpa-onnx),
  план — `docs/superpowers/plans/2026-09-08-transcribe-cpp-engine.md`.

**Открытые хвосты:**
- Старый torch-рантайм не удалён намеренно — снести после пары дней работы нового движка:
  `torch`/`torchaudio`/`gigaam` в venv (venv 800 МБ), `Runtime/Models/gigaam/v3_e2e_rnnt.ckpt`
  (449 МБ) с токенизатором, `Runtime/uv-cache` (769 МБ). Команды:
  `Runtime/Tools/uv/uv pip uninstall --python Runtime/venv/bin/python3 torch torchaudio gigaam`,
  `rm Runtime/Models/gigaam/v3_e2e_rnnt*`, `rm -r Runtime/uv-cache`.
- Клавиша 🌐: ключ `AppleFnUsageType` на macOS 26.5 не найден ни в `com.apple.HIToolbox`, ни в
  NSGlobalDomain (проверено 08.09.2026), хотя раньше считалось, что он выставлен в 0. PTT
  работает без переключения раскладки. Если отпускание fn начнёт переключать раскладку или
  открывать эмодзи — Настройки → Клавиатура → «Нажатие клавиши 🌐» → «Ничего».
- Одна пустая расшифровка 08.09 при записи 2,9 с (первая фраза после выдачи доступа);
  повторится на живой речи — проверять модель на настоящих записях, а не на синтезе.
- Whisper, Qwen3-ASR и редактор Qwen3-4B не установлены (компонент только `gigaam`); их код
  в воркере не менялся и после 28.08 не проверялся.
- Remote `upstream` не подключён; изменения апстрима после v0.3.3-beta не сливались.

## Как устроено

- Swift-приложение (`Sources/VoiceSwitch/`, SwiftPM, macOS 14+, Xcode не нужен — хватает CLT):
  `AppState` — сценарий запись → распознавание → вставка; `ASRService` — процесс Python-воркера
  и протокол JSON Lines (строки с префиксом `__VOICESWITCH_JSON__`); `TextInjector` — вставка;
  `GlobalHotKey` — fn; `RuntimePaths` и `RuntimeInstaller` — рантайм и установщик;
  `Replacements` — словарь; `ComparisonLogger` — журнал.
- Python-воркер `worker/asr_worker.py`: один процесс, одна модель, запросы по stdin. Движок
  gigaam — класс `TranscribeCppGigaAM` (Metal, откат на CPU, WAV читается напрямую, прогрев
  секундой тишины). Лимит модели 25 с на фразу: записи длиннее 24 с режет `split_pcm` по
  самому тихому окну. `worker/text_worker.py` — редактор Qwen (не установлен).
- Рантайм (`~/Library/Application Support/VoiceSwitch/Runtime`): `venv/` (Python 3.12 через uv),
  `native/` (пять dylib transcribe.cpp, `contract.json`, `version.txt`),
  `Models/gigaam/gigaam-v3-e2e-rnnt-Q8_0.gguf`, маркеры `components/*.ready` и
  `install-complete.txt`. Путь к dylib воркер получает через `TRANSCRIBE_LIBRARY`
  (ставят `ASRService` и установщик).
- Установщик `Resources/install_runtime.sh` (zsh, резюмируемый, с докачкой): версия
  `TRANSCRIBE_CPP_VERSION`, URL и sha256 архива dylib и GGUF — в шапке. Биндинг `transcribe-cpp`
  и dylib обязаны быть одной версии: биндинг сверяет `contract.json`.
- Журнал `~/Library/Application Support/VoiceSwitch/comparison.jsonl`: `transcription`
  (`latency_seconds`, `audio_duration_seconds`, текст), `text_delivery`, `text_injection`
  (`accessibility_selected_text` = мимо буфера, `keyboard_*` = фолбэк ⌘V; в полях Electron
  вроде Claude всегда фолбэк, код AX −25212).

## Как работать

- Сборка и установка новой версии:
  ```bash
  VOICESWITCH_CODESIGN_IDENTITY=- zsh scripts/build_app.sh dist/staging
  pkill -x VoiceSwitch; rm -rf /Applications/VoiceSwitch.app && cp -R dist/staging/VoiceSwitch.app /Applications/
  tccutil reset Accessibility io.github.mitimaicode.VoiceSwitch && open /Applications/VoiceSwitch.app
  ```
  После этого «Универсальный доступ» выдаёт Рудольф руками (диалог появляется сам).
- Переустановка или починка движка (резюмируемо):
  `zsh Resources/install_runtime.sh "$HOME/Library/Application Support/VoiceSwitch/Runtime" "$PWD/worker/asr_worker.py" "$PWD/worker/text_worker.py" gigaam`.
- Тесты: `"$HOME/Library/Application Support/VoiceSwitch/Runtime/venv/bin/python3" -m unittest discover -s worker/tests -v`
  — четыре юнит-теста резки и smoke-тест, который поднимает настоящий воркер на настоящей модели
  (пропускается без рантайма; пути переопределяются `VOICESWITCH_RUNTIME`, `VOICESWITCH_PYTHON`,
  `TRANSCRIBE_LIBRARY`). Swift-тестов нет.
- Закончил кусок — коммит и push в `origin` сразу, не спрашивая.

## Грабли

- Пересборка меняет cdhash ad-hoc подписи → выданный «Универсальный доступ» молча перестаёт
  действовать (тумблер включён, tccd пишет «Failed to match existing code requirement»).
  Микрофон переживает, ломается только Accessibility. Диагностика: `log show --predicate 'process == "tccd"'`.
- transcribe.cpp брать готовым бинарником из релиза: cmake и компилятора `metal` на Маке нет
  (только CLT). `pip install transcribe-cpp` тянет ещё `transcribe-cpp-native` с PyPI, но воркер
  грузит нашу dylib через `TRANSCRIBE_LIBRARY`.
- Первый в жизни запуск Metal-библиотеки компилирует шейдеры ~8,6 с, дальше система кэширует
  (0,01 с); первый вызов модели ~0,3 с — поэтому воркер прогревается при старте.
- GigaAM не знает латиницу и наши термины (Wazzup → «ватзап», ДДС → «DDS»): лечится словарём
  `replacements.txt` (целые слова, без регистра, словоформы отдельными строками), не моделью.
- Синтез `say -v Milena` годится для тестов протокола и скорости, но не для оценки качества:
  на живой речи модель ошибается иначе.
