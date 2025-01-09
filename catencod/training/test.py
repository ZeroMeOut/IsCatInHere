import torch
import torch.nn as nn
from PIL import Image
from train import LeNet
from torchvision.io import read_image
import torchvision.transforms as transforms


transform = transforms.Compose([
transforms.ToTensor(),   # Convert to tensor
transforms.Resize(size=(32,32), antialias=True)
])

catload = Image.open('notcat2.jpg')
transformed_cat = transform(catload).unsqueeze(0)
print("Cat loaded")

model = LeNet()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('log/model.pt', weights_only=True))
model.eval()
print("Model loaded, predicting....")

with torch.no_grad():
  output = model(transformed_cat)
  print(output)

