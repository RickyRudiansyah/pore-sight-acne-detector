# bismillahirrahmanirrahim
from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_YAML  = ROOT / "yolo" / "data.yaml"

def main():
    # mulai dari model resmi yang kecil (bisa ganti: yolov8s.pt, m.pt, dll)
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=150,
        batch=24,
        imgsz=640,
        lr0=3e-4,                # bisa sedikit lebih besar daripada fine-tune best.pt
        optimizer="AdamW",
        patience=30,
        name="train_from_pretrained",
        augment=True,
        mosaic=1.0,
        mixup=0.3,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        device=0
    )

    metrics = model.val(data=str(DATA_YAML), batch=24, device=0)
    print(metrics)

if __name__ == "__main__":
    main()
