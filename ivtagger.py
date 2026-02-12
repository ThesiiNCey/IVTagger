import os
import sys
import csv
import shutil
import threading
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin
import piexif
import onnxruntime as rt
import huggingface_hub
from exiftool import ExifToolHelper

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFileDialog, QCheckBox, QSlider, QProgressBar,
    QComboBox, QTextEdit, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage


max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)


MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

type_map = {
    'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
    'gif': 'image/gif', 'bmp': 'image/bmp', 'webp': 'image/webp'
}
video_exts = [".mp4", ".mkv", ".mov", ".webm"]

interrogators = {
    'wd14-vit.v1': "SmilingWolf/wd-v1-4-vit-tagger",
    'wd14-vit.v2': "SmilingWolf/wd-v1-4-vit-tagger-v2",
    'wd14-eva02.v3.large': "SmilingWolf/wd-eva02-large-tagger-v3",
    'wd-v1-4-vit-tagger.v3': "SmilingWolf/wd-vit-tagger-v3",
    'wd14-vit.v3.large': "SmilingWolf/wd-vit-large-tagger-v3",
}


def download_model(model_repo):
    csv_path = huggingface_hub.hf_hub_download(model_repo, LABEL_FILENAME)
    model_path = huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME)
    return csv_path, model_path


class LabelData:
    def __init__(self, names, rating, general, character):
        self.names = names
        self.rating = rating
        self.general = general
        self.character = character


def load_model_and_tags(model_repo):
    csv_path, model_path = download_model(model_repo)
    df = pd.read_csv(csv_path)

    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0])
    )

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    model = rt.InferenceSession(model_path, providers=providers)
    target_size = model.get_inputs()[0].shape[2]

    return model, tag_data, target_size


def prepare_image(image, target_size):
    image = image.convert("RGB")
    max_dim = max(image.size)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, ((max_dim - image.size[0]) // 2,
                         (max_dim - image.size[1]) // 2))
    padded = padded.resize((target_size, target_size), Image.BICUBIC)
    arr = np.asarray(padded, dtype=np.float32)[..., [2, 1, 0]]
    return np.expand_dims(arr, axis=0)


def process_predictions(preds, tag_data, character_thresh,
                        general_thresh, hide_rating, char_first):
    scores = preds.flatten()
    character_tags = [tag_data.names[i] for i in tag_data.character if scores[i] >= character_thresh]
    general_tags = [tag_data.names[i] for i in tag_data.general if scores[i] >= general_thresh]
    rating_tags = [] if hide_rating else [tag_data.names[i] for i in tag_data.rating]

    final = character_tags + general_tags if char_first else general_tags + character_tags
    final += rating_tags
    return final


def extract_frames(video_path, frame_distance, stop_event):
    frame_folder = os.path.splitext(video_path)[0] + "_frames"
    os.makedirs(frame_folder, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select=not(mod(n\\,{frame_distance}))",
        "-vsync", "vfr",
        os.path.join(frame_folder, "frame_%04d.png")
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    while proc.poll() is None:
        if stop_event.is_set():
            proc.terminate()
            return [], frame_folder

    frames = sorted(Path(frame_folder).glob("*.png"))
    return frames, frame_folder


def write_text_file(base_path, tags):
    txt_path = os.path.splitext(base_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(tags))


def write_image_metadata_exiftool(base_path, tags, overwrite=False):
    """Schreibt Tags via ExifTool (EXIF, IPTC, XMP)"""
    try:
        with ExifToolHelper(encoding="utf-8") as et:
            existing = et.get_tags([base_path], ["IPTC:Keywords", "XMP:Subject"])[0]
            iptc_list = existing.get("IPTC:Keywords", []) or []
            xmp_list = existing.get("XMP:Subject", []) or []
            combined_tags = tags if overwrite else tags + iptc_list + xmp_list
            combined_tags = list(dict.fromkeys(combined_tags))
            et.set_tags([base_path], tags={"IPTC:Keywords": combined_tags,
                                           "XMP:Subject": combined_tags},
                        params=["-P", "-overwrite_original"])
    except Exception as e:
        print(f"ExifTool error for {base_path}: {str(e)}")


def write_image_metadata(base_path, tags, overwrite=False):
    """Schreibt Metadaten zuverlässig über ExifTool (JPEG, PNG, WebP usw.)"""
    try:
        tags = [t.strip() for t in tags if t.strip()]
        if not tags:
            return

        with ExifToolHelper() as et:
            existing = et.get_tags([base_path], ["IPTC:Keywords", "XMP:Subject"])[0]
            iptc_list = existing.get("IPTC:Keywords", []) or []
            xmp_list = existing.get("XMP:Subject", []) or []

            final_tags = tags if overwrite else list(dict.fromkeys(tags + iptc_list + xmp_list))

            et.set_tags([base_path], tags={
                "IPTC:Keywords": final_tags,
                "XMP:Subject": final_tags,
                "EXIF:ImageDescription": ", ".join(final_tags)
            }, params=["-P", "-overwrite_original"])
    except Exception as e:
        print(f"ExifTool konnte Metadaten für {base_path} nicht schreiben: {e}")


def write_video_metadata(video_path, tags, overwrite=False):
    """Metadaten nur ändern, Format bleibt gleich, WebM & MP4 kompatibel"""
    comment = ", ".join(tags) if overwrite else ""
    if not overwrite:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True
        )
        old_comment = result.stdout.strip()
        comment = old_comment + ", " + ", ".join(tags) if old_comment else ", ".join(tags)

    ext = os.path.splitext(video_path)[1].lower()
    tmp_path = video_path + "_tmp" + ext

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-metadata", f"comment={comment}",
        "-codec", "copy",
        tmp_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.move(tmp_path, video_path)


class TaggerGUI(QWidget):
    update_preview_signal = pyqtSignal(QPixmap)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image & Video Tagger")
        self.resize(1100, 700)

        self.stop_event = threading.Event()

        main_layout = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        main_layout.addLayout(left, 2)
        main_layout.addLayout(right, 3)

        left.addWidget(QLabel("Folder:"))
        self.folder_input = QLineEdit()
        left.addWidget(self.folder_input)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_folder)
        left.addWidget(browse)

        self.recursive_checkbox = QCheckBox("Recursive")
        left.addWidget(self.recursive_checkbox)

        left.addWidget(QLabel("General Threshold"))
        self.general_slider = QSlider(Qt.Orientation.Horizontal)
        self.general_slider.setRange(0, 100)
        self.general_slider.setValue(35)
        left.addWidget(self.general_slider)

        left.addWidget(QLabel("Character Threshold"))
        self.character_slider = QSlider(Qt.Orientation.Horizontal)
        self.character_slider.setRange(0, 100)
        self.character_slider.setValue(85)
        left.addWidget(self.character_slider)

        self.hide_rating_checkbox = QCheckBox("Hide rating")
        self.hide_rating_checkbox.setChecked(True)
        left.addWidget(self.hide_rating_checkbox)

        self.character_first_checkbox = QCheckBox("Character first")
        left.addWidget(self.character_first_checkbox)

        left.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(interrogators.keys())
        left.addWidget(self.model_combo)

        left.addWidget(QLabel("Output Type"))
        self.output_combo = QComboBox()
        self.output_combo.addItems(["Text File", "Metadata", "Video Metadata"])
        left.addWidget(self.output_combo)

        left.addWidget(QLabel("Overwrite Metadata"))
        self.overwrite_checkbox = QCheckBox("Overwrite Metadata")
        left.addWidget(self.overwrite_checkbox)

        left.addWidget(QLabel("Frame Distance"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 100)
        self.frame_spin.setValue(10)
        left.addWidget(self.frame_spin)

        left.addWidget(QLabel("Additional / Extra Tags (always first)"))
        self.extra_tags_input = QLineEdit()
        left.addWidget(self.extra_tags_input)

        left.addWidget(QLabel("Ignore Tags"))
        self.ignore_input = QLineEdit()
        left.addWidget(self.ignore_input)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_tagging)
        left.addWidget(self.start_btn)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.progress = QProgressBar()

        self.preview = QLabel()
        self.preview.setFixedSize(420, 420)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("border:1px solid black;")

        right.addWidget(self.status)
        right.addWidget(self.progress)
        right.addWidget(self.preview)

        self.update_preview_signal.connect(self.preview.setPixmap)
        self.log_signal.connect(self.status.append)
        self.progress_signal.connect(self.progress.setValue)
        self.finished_signal.connect(self.on_finished)

    def show_preview(self, pil_image):
        qt_image = QImage(
            pil_image.tobytes(),
            pil_image.width,
            pil_image.height,
            pil_image.width * 3,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.preview.width(),
            self.preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.update_preview_signal.emit(scaled)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            self.folder_input.setText(folder)

    def start_tagging(self):
        if self.start_btn.text() == "Start":
            self.stop_event.clear()
            self.start_btn.setText("Stop")
            threading.Thread(target=self.run_tagging, daemon=True).start()
        else:
            self.stop_event.set()
            self.log_signal.emit("⛔ Abgebrochen")

    def on_finished(self):
        self.start_btn.setText("Start")

    def normalize_tag_list(self, text):
        return [t.strip().lower().replace(" ", "_") for t in text.split(",") if t.strip()]

    def run_tagging(self):
        folder = self.folder_input.text()
        if not os.path.exists(folder):
            self.log_signal.emit("Folder not found.")
            self.finished_signal.emit()
            return

        ignore_list = self.normalize_tag_list(self.ignore_input.text())
        extra_tags = self.normalize_tag_list(self.extra_tags_input.text())
        output_type = self.output_combo.currentText()
        overwrite = self.overwrite_checkbox.isChecked()

        model_repo = interrogators[self.model_combo.currentText()]
        model, tag_data, target_size = load_model_and_tags(model_repo)

        files = []
        for root, _, fs in os.walk(folder):
            for f in fs:
                files.append(os.path.join(root, f))
            if not self.recursive_checkbox.isChecked():
                break

        total = len(files)

        for idx, file_path in enumerate(files, 1):
            if self.stop_event.is_set():
                break

            ext = os.path.splitext(file_path)[1].lower()

            try:
                if ext in video_exts:
                    frames, tmp_folder = extract_frames(file_path, self.frame_spin.value(), self.stop_event)

                    video_tags = []
                    for frame in frames:
                        if self.stop_event.is_set():
                            break

                        with Image.open(frame) as img:
                            img_rgb = img.convert("RGB")
                            self.show_preview(img_rgb)

                            preds = model.run(
                                None,
                                {model.get_inputs()[0].name: prepare_image(img_rgb, target_size)}
                            )[0]

                            tags = process_predictions(
                                preds,
                                tag_data,
                                self.character_slider.value()/100,
                                self.general_slider.value()/100,
                                self.hide_rating_checkbox.isChecked(),
                                self.character_first_checkbox.isChecked()
                            )
                            video_tags.extend(t for t in tags if t not in ignore_list)

                    shutil.rmtree(tmp_folder, ignore_errors=True)
                    final_tags = list(dict.fromkeys(extra_tags + video_tags))

                    if output_type == "Video Metadata":
                        write_video_metadata(file_path, final_tags, overwrite)
                    elif output_type == "Metadata":
                        write_image_metadata(file_path, final_tags, overwrite)
                    else:
                        write_text_file(file_path, final_tags)

                elif ext.lstrip('.') in type_map:
                    with Image.open(file_path) as img:
                        img_rgb = img.convert("RGB")
                        self.show_preview(img_rgb)

                        preds = model.run(
                            None,
                            {model.get_inputs()[0].name: prepare_image(img_rgb, target_size)}
                        )[0]

                        tags = process_predictions(
                            preds,
                            tag_data,
                            self.character_slider.value()/100,
                            self.general_slider.value()/100,
                            self.hide_rating_checkbox.isChecked(),
                            self.character_first_checkbox.isChecked()
                        )

                        final_tags = list(dict.fromkeys(extra_tags + [t for t in tags if t not in ignore_list]))

                        if output_type == "Metadata":
                            write_image_metadata(file_path, final_tags, overwrite)
                        else:
                            write_text_file(file_path, final_tags)

                else:
                    continue

                self.progress_signal.emit(int(idx / total * 100))
                self.log_signal.emit(os.path.basename(file_path))

            except Exception as e:
                self.log_signal.emit(f"Error: {e}")

        self.finished_signal.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TaggerGUI()
    gui.show()
    sys.exit(app.exec())
