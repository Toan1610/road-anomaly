import cv2
import numpy as np
import openvino as ov
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL = ROOT / "models/road-segmentation-adas-0001/road-segmentation-adas-0001.xml"
IMAGE = ROOT / "test_frame.png"

image = cv2.imread(str(IMAGE))
if image is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {IMAGE}")

core = ov.Core()
model = core.read_model(str(MODEL))
compiled = core.compile_model(model, "CPU")

print("Input:", compiled.input(0).shape)
print("Output:", compiled.output(0).shape)

# Model yêu cầu ảnh BGR, kích thước 896x512
resized = cv2.resize(image, (896, 512))
tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None, ...])

result = compiled([tensor])[compiled.output(0)]
print("Kích thước tensor đầu ra:", result.shape)

# Phân lớp từng pixel
mask = np.argmax(result[0], axis=0).astype(np.uint8)

# Màu BGR cho các lớp BG, road, curb, mark
colors = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
], dtype=np.uint8)

mask_color = cv2.resize(
    colors[mask],
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

overlay = cv2.addWeighted(image, 0.65, mask_color, 0.35, 0)

cv2.imwrite(str(ROOT / "mask_openvino.png"), mask_color)
cv2.imwrite(str(ROOT / "outputs_openvino.png"), overlay)

print("Các lớp dự đoán:", np.unique(mask))
print("Đã lưu mask_openvino.png và outputs_openvino.png")
