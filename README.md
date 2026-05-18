# Introduction to GitHub

<img src="https://octodex.github.com/images/Professortocat_v2.png" align="right" height="200px" />

Hey Rajesh-2312!

Mona here. I'm done preparing your exercise. Hope you enjoy! 💚

Remember, it's self-paced so feel free to take a break! ☕️

[![](https://img.shields.io/badge/Go%20to%20Exercise-%E2%86%92-1f883d?style=for-the-badge&logo=github&labelColor=197935)](https://github.com/Rajesh-2312/skills-introduction-to-github/issues/1)

---

&copy; 2025 GitHub &bull; [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md) &bull; [MIT License](https://gh.io/mit)

---

## Real-time Object Detector (Python + OpenCV + YOLOv8)

A small, fully free desktop app that opens your webcam, detects objects in real
time using **YOLOv8 nano**, and shows **Wikipedia details** for any object on
demand.

### Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads `yolov8n.pt` (~6 MB) automatically.

**Linux only:** `pyttsx3` (TTS readout) talks to `espeak` under the hood, and
OCR uses the `tesseract` binary — install both once:

```bash
sudo apt install espeak espeak-data libespeak1 tesseract-ocr
```

macOS: `brew install tesseract` (speech is built-in). Windows: install
[Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) (speech
is built-in). If you don't want TTS or OCR, run with `--no-tts` / `--no-ocr`.

### Run

```bash
python -m app.main                       # default webcam (index 0)
python -m app.main --source 1            # second webcam
python -m app.main --conf 0.35 --width 1280 --height 720
python -m app.main --lang hi             # Hindi Wikipedia details
python -m app.main --lang te --no-tts    # Telugu, no speech

# Zero-shot mode: detect anything you name (slower, ~500 MB model on first run)
python -m app.main --prompts "wine bottle, guitar, red shoe, indoor plant"
```

### Use an Android phone as the camera (free, no extra code)

The Python app runs on your PC/laptop; the phone just streams its camera over
Wi-Fi. Both devices must be on the same network.

1. Install the free **IP Webcam** Android app
   ([Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam)).
2. Open the app → scroll down → **Start server**.
3. The phone screen shows a URL like `http://192.168.1.42:8080`.
4. On your PC, run:

   ```bash
   python -m app.main --source http://192.168.1.42:8080/video
   ```

   (Note the `/video` suffix — that's the MJPEG stream endpoint.)

The same `--source` flag works with any URL OpenCV can open, including:

- DroidCam: `http://PHONE_IP:4747/video`
- RTSP IP cameras: `rtsp://user:pass@CAMERA_IP:554/stream1`
- A recorded video file: `--source /path/to/clip.mp4`

> **Why this approach?** It's 100% free, requires no Android development, and
> gives you the phone's much-better camera while the heavy YOLO inference
> runs on your PC. If you'd rather run *everything on the phone itself* (no
> PC needed), see the roadmap below — that requires building a native
> Android app with TensorFlow Lite, which is a much bigger project.

### Controls

| Key | Action |
|-----|--------|
| `d` | Fetch Wikipedia details + speak them aloud |
| `r` | Re-speak the current details |
| `m` | Mute / unmute TTS |
| `n` / `p` | Cycle through detected objects |
| `c` | Clear the details panel |
| `s` | Save a snapshot to `snapshots/` |
| `q` | Quit |

Each detected object now also gets a **stable ID** (e.g. `#3 person 87%`) so
the same physical object keeps the same number across frames — pressing `d`
on an object that's already on-screen won't refetch its Wikipedia entry.

### Detect anything (zero-shot)

The default model knows 80 COCO classes. Pass `--prompts "..."` to swap in
**OWL-ViT v2**, which detects anything you describe in natural language:

```bash
python -m app.main --prompts "wine bottle, fender stratocaster, samsung phone"
```

First run downloads ~500 MB of weights (cached in `~/.cache/huggingface/`).
Inference on CPU is ~1–3 fps — for real-time use, stick with YOLOv8 and only
flip to zero-shot when you need to find something outside the 80 classes.

### Read labels on objects (OCR)

When you press `d`, the app also runs **Tesseract** on the selected object's
crop. If readable text is found (a book title, a bottle label, a sign), it
appears under the Wikipedia summary in the details panel and is included in
the speech readout. Disable with `--no-ocr`.

### How "details" works

When you press `d`, the app queries the Wikipedia REST API for the predicted
class label, picks the top opensearch result (to disambiguate words like
"mouse"), and caches the response in `cache/<label>.json`. After the first
fetch the same object loads instantly and works offline.

### Upgrade roadmap

See the plan file for the full roadmap. Highlights: object tracking IDs,
voice trigger via faster-whisper, text-to-speech readout, zero-shot detection
with CLIP/OWL-ViT, custom YOLOv8 fine-tuning, Flask web version, multi-language
details, OCR fallback, and a Groq-hosted Llama 3 follow-up Q&A.

#### Native Android (no PC needed)

To run the *entire* pipeline on the phone with no laptop:

1. Export YOLOv8n to **TensorFlow Lite**: `yolo export model=yolov8n.pt format=tflite`
2. Scaffold an Android app in **Android Studio** (Kotlin + CameraX).
3. Use the **TensorFlow Lite Android** library to run the `.tflite` model on
   CameraX frames.
4. Call the Wikipedia REST API from Android using OkHttp / Retrofit.
5. Optional: ship with a bundled offline JSON for the 80 COCO classes so
   details work without internet.

All of the above uses free tooling, but it's a separate ~few-hundred-line
Kotlin project rather than an extension of this Python codebase. Happy to
generate it as a sibling project on request.

