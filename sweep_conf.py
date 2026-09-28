from ultralytics import YOLO

model = YOLO("/Users/malikaeryomina/runs/segment/train-13/weights/best.pt")

print("conf   boxR    boxP")
for c in [0.001, 0.05, 0.10, 0.15, 0.25]:
    r = model.val(split='val', conf=c, iou=0.7, imgsz=1024, verbose=False)
    print(c, "  ", round(r.box.mr, 3), " ", round(r.box.mp, 3))