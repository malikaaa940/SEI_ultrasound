from ultralytics import YOLO


def train_segmentation_model():
    print("Initializing YOLOv8-Segmentation Model...")

    model = YOLO("yolov8n-seg.pt")

    print("Beginning training sequence")
    results = model.train(
        data="/Users/malikaeryomina/SEI_AI_ultrasound/dataset.yaml",
        name="train_seeded_v1",   # clearly tag the run trained on the new frozen split
        seed=42,
        deterministic=False,
        epochs=200,
        imgsz=640,
        device="mps",
        batch=8,
        cache=True,
        patience=50,
        copy_paste=0.4,
        mosaic=1.0, scale=0.5, fliplr=0.5, degrees=10, translate=0.1,
        hsv_h=0.0, hsv_s=0.0,     # inert on grayscale; hsv_v (brightness) kept at default
    )

    print("\nTraining complete")


if __name__ == "__main__":
    train_segmentation_model()