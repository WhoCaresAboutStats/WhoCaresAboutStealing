from ultralytics import YOLO

DATA = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\V2\data.yaml"
MODEL = "yolo11s.pt"
OUTPUT = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\OUT"

def main():
    model = YOLO(MODEL)

    results = model.train(
        data=DATA,
        epochs=80,
        imgsz=800,
        batch=16,
        device=0,
        lr0=0.002,
        optimizer="AdamW",
        patience=25,
        mosaic=0.5,
        project=OUTPUT,
        name="v4"
    )

    print("Training complete!")
    print("Best weights saved at:", results.save_dir)

if __name__ == "__main__":
    main()
