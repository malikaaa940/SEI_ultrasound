from ultralytics import YOLO
import os


def test_ai_model_batch():
    # Load  newly trained custom brain
    model = YOLO("/Users/malikaeryomina/runs/segment/train-8/weights/best.pt")

    test_folder_path = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo/images/test"

    if not os.path.exists(test_folder_path):
        print(f"Error: Could not find the folder at {test_folder_path}")
        return

    print(f"Running AI prediction on ALL images in: {test_folder_path}")

    # 3. Run prediction on the whole folder
    # save=True means it will save all the images with drawings on them)
    results = model.predict(source=test_folder_path, show=False, save=True)

    print("\nBatch prediction complete")
    print(" check the 'runs/segment/predict' folder to see all the results!")


if __name__ == "__main__":
    test_ai_model_batch()