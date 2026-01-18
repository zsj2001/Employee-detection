import torch
from ultralytics import YOLO

def main():
    print("CUDA Available: " + str(torch.cuda.is_available()))
    model = YOLO('yolov8n.pt')
    results = model.train(data="config.yaml",
                          epochs=100,
                          batch=4,
                          workers=0,  # stops extra processes from eating VRAM
                          patience=10,
                          device=0,
                          imgsz=640) # After training, best.pt will appear. That's the best model. Pick that one.

if __name__ == '__main__':
    main()