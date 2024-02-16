# Script to load models
from ultralytics import YOLO
from torch import hub, nn, device, cuda
from facenet_pytorch import MTCNN, InceptionResnetV1

def load_all_modes(device_index = 0):

    print('Loading the Models')
    
    device_loc = device(f'cuda:{device_index}' if cuda.is_available() else 'cpu')

    model_path = './models/yolov8m.pt'

    # Load a REGULAR model
    reg_model = YOLO(model_path, 'detect')  # pretrained YOLOv8n model
    reg_model.to(device_loc)
    reg_model.name = 'yolov8'

    # Load EMBEDDING model
    emb_model = YOLO(model_path, 'detect')  # pretrained YOLOv8n model
    emb_model.to(device_loc)
    emb_model.model.model = emb_model.model.model[:-1]
    emb_model.name = "emb-yolov8"

    # resnet models
    # resnet_model = hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # resnet_model = nn.Sequential(*(list(resnet_model.children())[:-1]))
    # resnet_model.eval()
    # resnet_model.to(device_loc)
    # resnet_model.name = 'resnet50'

    # Facial detection and recognition
    # If required, create a face detection pipeline using MTCNN:
    face_mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device_loc, keep_all=True)

    # Create an inception resnet (in eval mode):
    face_resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device_loc)
    
    return reg_model, emb_model, face_mtcnn, face_resnet, device_loc