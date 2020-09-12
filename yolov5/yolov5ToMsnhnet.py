from PytorchToMsnhnet import *
from models.experimental import attempt_load
import torch

weights     = "weights/yolov5s.pt"
msnhnetPath = "yolov5s.msnhnet"
msnhbinPath = "yolov5s.msnhbin"

model = attempt_load(weights, "cpu")
model.eval()

img = torch.rand(512*512*3).reshape(1,3,512,512)

trans(model,img,msnhnetPath,msnhbinPath)