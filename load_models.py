# Script to load models
from ultralytics import YOLO
from torch import hub, nn
from pymilvus import connections
from database_vec import create_new_collection

### LOADING THE MODELS ###
print('Loading the Models')

model_path = './models/yolov8m.pt'

# Load a REGULAR model
reg_model = YOLO(model_path, 'detect')  # pretrained YOLOv8n model
reg_model.to('cuda:0')
reg_model.name = 'yolov8'

# Load EMBEDDING model
emb_model = YOLO(model_path, 'detect')  # pretrained YOLOv8n model
emb_model.to('cuda:0')
emb_model.model.model = emb_model.model.model[:-1]
emb_model.name = "emb-yolov8"

# resnet models
resnet_model = hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet_model = nn.Sequential(*(list(resnet_model.children())[:-1]))
resnet_model.eval()
resnet_model.to('cuda:0')
resnet_model.name = 'resnet50'