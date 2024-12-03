from flask import Flask, render_template, request, redirect, url_for, send_file
from transformers import pipeline
import torch
import os
import time

app = Flask(__name__)

# Periksa GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Muat model sekali saja
model = pipeline("automatic-speech-recognition",
                 "openai/whisper-small",
                 chunk_length_s=30,
                 stride_length_s=5,
                 return_timestamps=True,
                 device=device)

# Folder untuk file sementara
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio_file' not in request.files:
        return redirect(url_for('index'))

    # Simpan file audio yang diunggah
    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return redirect(url_for('index'))

    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    # Proses transkripsi
    language = request.form.get("language")
    task = request.form.get("task")
    start_time = time.time()

    transcription = model(file_path, generate_kwargs={"language": language, "task": task})
    formatted_transcription = format_transcription(transcription)

    # Bersihkan file sementara
    os.remove(file_path)

    # Hitung waktu proses
    end_time = time.time()
    duration = round(end_time - start_time, 2)

    return render_template("result.html", transcription=formatted_transcription, task=task, duration=duration)

def format_transcription(transcription):
    """Format hasil transkripsi dengan timestamp."""
    formatted_text = ""
    for line in transcription['chunks']:
        text = line["text"]
        ts = line["timestamp"]
        formatted_text += f"[{ts[0]}:{ts[1]}] {text}\n"
    return formatted_text.strip()

if __name__ == "__main__":
    app.run(debug=True)
