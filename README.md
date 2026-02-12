# Image & Video Tagger

A Python desktop application for automatically tagging images and videos using ONNX models.
Supports adding tags to **text files**, **image metadata (EXIF, IPTC, XMP)**, or **video metadata**. Built with **PyQt6**, **ONNX Runtime**, and **ExifTool**.

---

## Features

* Process **images** (`jpg`, `png`, `webp`, etc.) and **videos** (`mp4`, `mkv`, `mov`, `webm`)
* Extract frames from videos for tagging
* Supports multiple ONNX tagging models from Hugging Face
* Write tags to:

  * Text files
  * Image metadata (EXIF/IPTC/XMP)
  * Video metadata (MP4, WebM)
* Customizable thresholds for character/general tags
* Option to hide rating tags or prioritize character tags
* Extra tags and ignore lists
* Recursive folder processing
* Live preview of images/frames
* Stop button to cancel processing

---

## Requirements

* Python 3.10+
* [PyQt6](https://pypi.org/project/PyQt6/)
* [numpy](https://pypi.org/project/numpy/)
* [pandas](https://pypi.org/project/pandas/)
* [Pillow](https://pypi.org/project/Pillow/)
* [onnxruntime](https://pypi.org/project/onnxruntime/)
* [huggingface_hub](https://pypi.org/project/huggingface-hub/)
* [piexif](https://pypi.org/project/piexif/)
* [exiftool](https://exiftool.org/) installed and in your system PATH
* [ffmpeg](https://ffmpeg.org/) installed and in your system PATH

All Python dependencies are listed in `requirements.txt`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/image-video-tagger.git
cd image-video-tagger
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

### 3. Activate the virtual environment

* **Windows (cmd):**

```cmd
venv\Scripts\activate
```

* **Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

* **Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Ensure external tools are installed

* **ExifTool**: Must be available in PATH
* **FFmpeg**: Must be available in PATH

---

## Usage

```bash
python tagger.py
```

1. Select a folder containing images or videos
2. Configure thresholds, output type, extra tags, ignore tags, and model
3. Click **Start**
4. Progress will be displayed in the right panel, along with a live preview

### Output options

* **Text File**: Writes tags to a `.txt` file with the same name as the media
* **Metadata**: Writes tags to the image metadata (EXIF/IPTC/XMP)
* **Video Metadata**: Writes tags to video metadata (comment field)

---

## Models

Available tagging models (loaded from Hugging Face):

* `wd14-vit.v1`
* `wd14-vit.v2`
* `wd14-eva02.v3.large`
* `wd-v1-4-vit-tagger.v3`
* `wd14-vit.v3.large`

The model CSV (`selected_tags.csv`) contains tag names and categories (general, character, rating).

---

## Notes

* PNG, JPEG, and WebP images are fully supported for metadata writing.
* Video frame extraction uses FFmpeg; large videos may take time.
* If using **Overwrite Metadata**, previous tags are replaced. Otherwise, new tags are appended.
* Stop button can be used to abort processing at any time.

---

## Contact

For questions, feedback, or issues, you can contact me via:

* Email: `social@aimiko.moe`
* Matrix: `@aiiko:matrix.mochiart.moe`

---

## License

Apache 2.0 License

---

## Acknowledgements

* ONNX Runtime for fast inference
* Hugging Face for hosting tagging models
* ExifTool for robust metadata handling
* PyQt6 for GUI
