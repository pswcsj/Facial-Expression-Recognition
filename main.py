# import cv2
import torch
from model.EfficientNet import EfficientNet
# from data_loader import FERTrainDataLoader, FERTestDataLoader, FERTestDataSet
from PIL import Image
from torchvision import transforms
from utils.util import make_model_from_pretrained
# from mtcnn import MTCNN
# cap = cv2.VideoCapture(0)
# print(111)

loader = transforms.Resize((224, 224))
convert_tensor = transforms.ToTensor()
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    # image = loader(image).float()
    return convert_tensor(image) #assumes that you're using GPU

img = image_loader('smile.jpeg')

# detector = MTCNN()
model = make_model_from_pretrained('0model.pt')
model.eval()
# faces = detector.detect_faces(img)
# for face in faces:
#     top_left_x, top_left_y, width, height = face['box']
#     cropped_img = img[:, top_left_y:top_left_y+height, top_left_x:top_left_x+width]
#     x = loader(cropped_img)
#     print(torch.argmax(model(x)))


print(model(image))
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