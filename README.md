# PoreSight – Acne/Blackhead Detector (YOLOv8 + Flask)

Deteksi jerawat & komedo dari foto wajah menggunakan **Ultralytics YOLOv8**.  
Repo ini berisi **web app (Flask)** untuk upload gambar → deteksi → tampilkan hasil, serta skrip **training**.

## ✨ Fitur
- Upload gambar (PNG/JPG/JPEG) → inferensi YOLO → simpan & tampilkan hasil.
- Endpoint: `/` (UI), `/predict` (POST), `/results/<file>` (serve hasil).
- Folder `uploads/` & `results/` dibuat otomatis saat runtime.

## 🚀 Quickstart (Inference via Flask)
```bash
pip install -r requirements.txt
python app.py
# buka: http://127.0.0.1:5000
