from ultralytics import YOLO
import torch

print("--- ПЕРЕВІРКА СЕРЕДОВИЩА ---")
print(f"PyTorch версія: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA (GPU) доступна: {cuda_available}")

if cuda_available:
    print(f"Відеокарта: {torch.cuda.get_device_name(0)}")
else:
    print("УВАГА: CUDA не виявлена. Навчання буде на процесорі (довго).")

print("\n--- ПЕРЕВІРКА YOLO ---")
# Завантажуємо найменшу модель (ваги yolo11n.pt завантажаться автоматично)
model = YOLO("yolo11n.pt")
print("YOLO11n успішно завантажена!")