from ultralytics import YOLO


def take_final_exam():

    model = YOLO("/Users/malikaeryomina/runs/segment/train_seeded_v1-3/weights/best.pt")

    print("Starting official evaluation on the Blind Test Set...")

    # grades the AI against labels
    metrics = model.val(split='test', name = 'final_test_results' )

    print("\n--- FINAL RESULTS ---")
    print(f"Overall mAP50: {metrics.seg.map50:.3f}")
    print(f"Box recall:  {metrics.box.mr:.3f}")
    print(f"Box mAP50:   {metrics.box.map50:.3f}")
    print(f"Mask recall: {metrics.seg.mr:.3f}")


if __name__ == "__main__":
    take_final_exam()