# Движок GigaAM через transcribe.cpp — дизайн

Дата: 08.09.2026. Статус: утверждён Рудольфом («ок, поехали»).

## Зачем

Замер 08.09.2026 по журналу `comparison.jsonl` (190 фраз) и бенчам на M2 Pro
показал, что модель GigaAM v3 e2e RNNT не является узким местом диктовки:

- Python-воркер с torch занимает 1,9–2,0 ГБ и в паузах целиком уходит в сжатую
  память. Первая фраза после перерыва >10 мин ждёт 5–7 с: 23 % фраз, 59 % всего
  ожидания.
- torch на 6 потоках медленнее, чем на 2 (E-ядра тормозят OpenMP).
- `gigaam.load_audio` запускает ffmpeg сабпроцессом на каждую фразу.
- `TextInjector.paste` всегда ждёт 0,35 с перед вставкой, даже если целевое окно
  уже впереди.

Та же модель, перенесённая на ggml + Metal (transcribe.cpp, MIT, GGUF Q8_0):
0,03 с на фразу 3,7 с и 0,14 с на 22 с при 386 МБ памяти процесса. Тексты
совпали с torch-вариантом. Это снимает обе проблемы — скорость и холодные старты.

## Решение

Заменить реализацию движка `gigaam` внутри существующего Python-воркера:
вместо `torch` + пакета `gigaam` — биндинг `transcribe-cpp` 0.2.3 и готовые
dylib из релиза transcribe.cpp v0.2.3 (`transcribe-native-0.2.3-macos-arm64-metal`).
Протокол JSON Lines, `ASRService.swift`, движки Whisper/Qwen/Apple не меняются.

Отвергнутые варианты:
- **Нативный Swift через `TranscribeCpp.xcframework`** — биндинг помечен
  «в разработке», выигрыш против Python-пути почти нулевой (0,03 с). Возможен позже.
- **Гибрид MPS-энкодер + CPU-декодер в torch** — 0,04 с, но память остаётся 2 ГБ,
  холодные старты пришлось бы лечить «пингом». Рудольф выбрал transcribe.cpp.

## Компоненты

### 1. `worker/asr_worker.py`

- Движок `gigaam`: `transcribe_cpp.Model(gguf, device=<gpu из backends()>)`;
  если gpu-устройства нет или загрузка на нём упала — CPU с сообщением в stderr.
- Путь к библиотеке: переменная окружения `TRANSCRIBE_LIBRARY`, если не задана —
  `<cache>/../native/libtranscribe.dylib` (то есть `Runtime/native/`).
  Путь к модели: `<cache>/gigaam/gigaam-v3-e2e-rnnt-Q8_0.gguf`.
- Аудио читается напрямую (`wave` + numpy → float32 16 кГц mono); ffmpeg не
  вызывается. Если WAV не 16 кГц mono 16-bit — ошибка запроса.
- Резка длинных записей (лимит модели 25 с) сохраняется, но работает с массивом:
  чистая функция `split_pcm(samples, rate, maximum_seconds=22.0)` возвращает
  список массивов; границы — по минимуму энергии в окне 120 мс в последних 5 с
  чанка, не раньше 8 с от начала чанка. Логика та же, что у `split_wav_for_gigaam`.
- После загрузки — прогрев `session.run` на секунде тишины, затем `ready`.
- Ошибки старта — понятные сообщения: нет библиотеки / нет модели → «…запустите
  установку моделей из меню VoiceSwitch»; несовпадение версии биндинга и
  библиотеки — как есть, текст исключения.
- `--download --engine gigaam` делает то же, что раньше: загружает модель и
  отвечает `ready` (контрольная загрузка в установщике).
- Одна сессия на процесс, запросы обрабатываются последовательно (как и раньше).

### 2. `Resources/install_runtime.sh`, блок `gigaam`

- Переменные: `TRANSCRIBE_CPP_VERSION=0.2.3`, URL архива dylib (GitHub release),
  URL GGUF (`huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf`), sha256 GGUF.
- Шаги: скачать архив с докачкой → распаковать в `Runtime/native/` (плоско, пять
  dylib и `contract.json`); `uv pip install transcribe-cpp==<версия>`; скачать GGUF с
  докачкой в `Models/gigaam/`, сверить sha256 (при несовпадении удалить и упасть
  шагом); контрольная загрузка `asr_worker.py --download --engine gigaam`
  с `TRANSCRIBE_LIBRARY`; `mark_component gigaam`.
- torch, torchaudio, `GIGAAM_ARCHIVE` из блока удаляются. Базовое окружение
  (uv, Python 3.12, numpy, huggingface_hub, imageio-ffmpeg) остаётся: ffmpeg нужен
  Whisper. Версии биндинга и dylib берутся из одной переменной.

### 3. Swift

- `RuntimePaths.isInstalled(.gigaam)` дополнительно требует файл GGUF
  (`Models/gigaam/gigaam-v3-e2e-rnnt-Q8_0.gguf`). Так после обновления
  приложение само покажет GigaAM неустановленным и предложит доустановку, а не
  упадёт в воркере. `expectedRuntimeVersion` не поднимается.
- `RuntimeComponent.gigaam.downloadSize` → «≈ 0,3 ГБ + базовое окружение».
- `ASRService.ensureWorker` передаёт воркеру `TRANSCRIBE_LIBRARY`
  (`Runtime/native/libtranscribe.dylib`), чтобы путь задавался в одном месте
  для сервиса и установщика.
- `TextInjector.paste` (отдельный коммит): задержка 0,35 с только когда целевое
  приложение не было frontmost и его пришлось активировать; иначе вставка сразу.

### 4. Тесты — `worker/tests/`

- `test_split_pcm.py` (unittest): короткий массив не режется; 60 с режутся на
  чанки ≤ 22 с, суммарная длина сохраняется, граница попадает в тихую область;
  первая граница не раньше 8 с.
- `test_worker_smoke.py`: если есть `Runtime/native/libtranscribe.dylib` и GGUF
  (пути через `VOICESWITCH_RUNTIME`, по умолчанию Application Support), поднимает
  `asr_worker.py --serve --engine gigaam`, ждёт `ready`, шлёт запрос на
  `samples/ru.wav`-подобный файл (генерируется `say`) и проверяет непустой
  русский текст и `latency < 2`. Иначе `skipTest`.
- Запуск: `Runtime/venv/bin/python3 -m unittest discover worker/tests`.

### 5. Документация

- README: размер «Русской диктовки» и системные требования; в лицензиях —
  transcribe.cpp и ggml (MIT). Таблица сравнения движков — сноска, что GigaAM
  теперь через transcribe.cpp.
- `docs/superpowers/plans/` — план реализации.

## Выкатка на Маке Рудольфа

1. Прогнать блок gigaam установщика напрямую:
   `zsh Resources/install_runtime.sh "$RUNTIME" worker/asr_worker.py worker/text_worker.py gigaam`.
2. Прогнать тесты.
3. Собрать: `VOICESWITCH_CODESIGN_IDENTITY=- zsh scripts/build_app.sh dist/staging`,
   скопировать в `/Applications`, `tccutil reset Accessibility io.github.mitimaicode.VoiceSwitch`,
   перезапустить. «Универсальный доступ» выдаёт Рудольф руками.
4. torch/torchaudio/gigaam из venv пока не удалять; после пары дней работы —
   `uv pip uninstall`, освободит ~1,5 ГБ.

## Критерии готовности

- Тёплая фраза ≤ 0,2 с по `latency_seconds` в `comparison.jsonl`, память воркера
  ≤ 500 МБ (`top -stats mem,cmprs`).
- Первая фраза после паузы >10 мин без пятисекундного провала.
- Тексты на контрольных файлах совпадают с прежним движком.
