# Guarded LLM — Android research prototype

An English-language, offline toxicity research prototype for manually entered text and cautiously selected visible Instagram chat nodes. This is not a validated cyberbullying or grooming detector and is not a safety guarantee. Instagram can change or omit its accessibility tree, so monitoring may detect no messages at all. No parental notifications or remote service are included.

## Status

The repository provides the application and a reproducible model-export/evaluation pipeline. The ONNX checkpoint is not committed: it must be generated before building. This workspace contains the exported checkpoint and a synthetic evaluation in `docs/synthetic-evaluation.json`. The ONNX export and Kotlin tokenizer passed the supplied reference examples. Android build and JVM tests pass; physical-device tests and Instagram node compatibility remain unverified; see `docs/device-test.md`. The bundled synthetic sets are smoke-test material, not evidence of performance in production.

## Setup

Install Android Studio, its SDK platform 35 and build tools, and JDK 17 or 21. Configure `ANDROID_HOME` or Android Studio's local SDK path. The Gradle wrapper downloads Gradle 8.10.2. A phone with Android 8.0 or newer is required for device verification. For the model pipeline use Python 3.10 or 3.11, a network connection and sufficient disk and memory for PyTorch and the model.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt
python tools/prepare_model.py
python tools/evaluate.py
./gradlew testDebugUnitTest assembleDebug
```

The exporter downloads `unitary/toxic-bert`, records its immutable resolved revision and license in `model-info.json`, exports the original unquantized model and checks ONNX logits against PyTorch. For reproducibility on later machines, use `python tools/prepare_model.py --revision <recorded-revision>`. The generated assets are excluded from version control because of size; include them in the APK by generating before building. The export script checks model-level parity and generates reference vectors for a device-tokenizer unit test. Compare tokenizer outputs against the reference on a broader test set before relying on it. The current BERT tokenizer is intended for English and may differ from Hugging Face on Unicode edge cases.

Install `app/build/outputs/apk/debug/app-debug.apk` with Android Studio or `adb install -r ...`. Open the app, test manual analysis, then explicitly accept the accessibility disclosure and enable its service in Android settings. Grant notification permission if prompted. The app must be active in its own controls as well as enabled in Android settings. Stop or pause monitoring in the app; after process restart it is off. Android's accessibility settings can independently disable the service.

## Behavior and boundaries

The accessibility service receives events only from `com.instagram.android`, requires a recognizable chat composer, and reads only non-editable nodes named `message_text`, `direct_message_text`, or `message_content`. These are conservative hypotheses, **not verified Instagram identifiers**; version changes or absent IDs can make coverage zero. Direction, conversation identity, sarcasm and history are not reliably available. The service does not attempt OCR, screen capture, bypasses or keystroke collection. While processing, text stays in memory and is not persisted; a bounded temporary set of hash codes suppresses repeated text. Stopping or pausing clears this set and hides the last result. In-flight inference can finish in memory but cannot generate an automatic alert after stop or pause. The UI can retain manually entered text until cleared or the process is destroyed.

The app declares no internet permission and does not store or transmit conversations. Notifications contain generic text only. It uses a fixed experimental toxicity threshold of 0.5, not a validated probability or a grooming score. The current model was trained on toxicity data, not Instagram chat contexts; sarcasm and identity terms can lead to errors. The service may analyze sent and received messages alike. It is not suitable for unsupervised use as a protective system.

`tools/evaluate.py` measures precision, recall and confusion counts on separated synthetic calibration and test sets, saving a report under `reports/`. It does not tune the threshold or claim external validity. See `data/README.md` for provenance and `docs/device-test.md` for the device protocol. The build and model preparation require internet access, but the installed app does not.

Google Play restricts Accessibility API use; this research build has not been reviewed or approved for publication. See https://support.google.com/googleplay/android-developer/answer/10964491 and the model card https://huggingface.co/unitary/toxic-bert for upstream limits and license.

## Results in this workspace

The exported model revision is `4d6c22e74ba2fdd26bc4f7238f50766b045a0d94`; its ONNX file is 438,234,020 bytes. The maximum absolute difference from PyTorch logits on four examples was 0.00000465. Kotlin tokenizer output matched the four generated Hugging Face vectors using a standalone JVM check. At threshold 0.5 the held-out synthetic set had 5 true positives, 4 true negatives, 1 false positive and 0 false negatives (precision 0.83, recall 1.0). This tiny constructed dataset does not validate the model.

The Android SDK platform and build tools are installed in `.android-sdk/` in this workspace. `testDebugUnitTest` and `assembleDebug` succeeded; the debug APK is at `app/build/outputs/apk/debug/app-debug.apk`. No Android device was connected, so the APK and Instagram integration have not been tested on a phone. See `docs/research.md` for the AWARE review, datasets and an exploratory test on real YouTube comments.

The [research and development diary](docs/development-diary.md) records project decisions, experiments and open questions for the research report.

## Grooming research pilot

The separate PAN12 pilot compares TF-IDF, MobileBERT (frozen and fine-tuned) and a frozen ModernBERT probe on a small author-grouped sample, then tests ONNX export, quantization and host CPU inference. It does **not** alter the Android app or validate grooming detection. Model artifacts and sensitive PAN12 data remain locally in ignored directories. See `docs/research.md` and `docs/development-diary.md` for limitations, commands and results.
