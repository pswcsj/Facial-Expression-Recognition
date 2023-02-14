import cv2
import torch
from model.EfficientNet import EfficientNet

cap = cv2.VideoCapture(0)
print(111)
model = EfficientNet.from_pretrained('EfficientNet-b7', num_classes=7, in_channels=1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    cv2.imshow('Input', frame)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = torch.from_numpy(img)
    print(img.shape)

    print(12321)
    print(model(img).shape)
    # break
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()