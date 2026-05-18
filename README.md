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

### Run

```bash
python -m app.main
```

Useful flags:

```bash
python -m app.main --camera 1 --conf 0.35 --width 1280 --height 720
```

### Controls

| Key | Action |
|-----|--------|
| `d` | Fetch Wikipedia details for the selected object |
| `n` / `p` | Cycle through detected objects |
| `c` | Clear the details panel |
| `s` | Save a snapshot to `snapshots/` |
| `q` | Quit |

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

