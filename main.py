import cv2
import torch
from model.EfficientNet import EfficientNet
from data_loader import FERTrainDataLoader, FERTestDataLoader, FERTestDataSet
from PIL import Image

# cap = cv2.VideoCapture(0)
# print(111)


model = EfficientNet.from_pretrained('EfficientNet-b7', num_classes=7, in_channels=1)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

loader = transforms.Compose([transforms.Scale(48), transforms.Grayscale(1), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image #assumes that you're using GPU

image = image_loader('smile.jpeg')

model(image)
# Check if the webcam is opened correctly
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
#
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('Input', frame)
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     img = torch.from_numpy(img)
#     print(img.shape)
#
#     print(12321)
#     print(model(img).shape)
#     # break
#     c = cv2.waitKey(1)
#     if c == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()