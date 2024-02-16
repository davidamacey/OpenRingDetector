# Script with functions for video/image processing to include detection and embedding

from cv2 import VideoCapture, CAP_PROP_FPS, imwrite, imread, resize, COLOR_BGR2RGB, cvtColor
from os import makedirs, path, listdir
import pandas as pd
from uuid import uuid4
from datetime import datetime
from torch import zeros, float32, tensor, stack, mean, norm, cat, Tensor
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import numpy as np
import pandas as pd
from datetime import datetime
from uuid import uuid4
from torch import no_grad, cuda, nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from utils import timer_func

def extract_frames(video_path, output_folder, start_time, stop_time):
    """
    Extract frames from a video within a specified time range and save them as image files.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder where frames will be saved.
        start_time (float): Start time in seconds.
        stop_time (float): Stop time in seconds.

    Returns:
        None
    """
    # Open the video file
    cap = VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = int(cap.get(CAP_PROP_FPS))

    # Calculate the start and stop frame numbers
    start_frame = int(start_time * fps)
    stop_frame = int(stop_time * fps)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Create the output folder if it doesn't exist
    if not path.exists(output_folder):
        makedirs(output_folder)

    frame_number = 0

    # Loop through the frames
    while True:
        ret, frame = cap.read()

        # Break the loop if the video ends or we've reached the stop frame
        if not ret or frame_number > stop_frame:
            break

        # If the current frame is within the specified range, save it
        if frame_number >= start_frame:
            filename = path.join(output_folder, f"frame_{frame_number:04d}.jpg")
            imwrite(filename, frame)

        frame_number += 1

    # Release the video capture object
    cap.release()

    print(f"Frames saved from {start_frame} to {stop_frame} as JPEG images in {output_folder}.")


def preprocess_image_single(input_image, im_size=640):
    """
    Preprocess a single image using OpenCV, ensuring that it matches the desired size (640x640) with padding.
    Accepts either a file path or an already opened OpenCV image.

    Args:
        input_image (str or numpy.ndarray): Either a file path to the image or an already opened OpenCV image.
        im_size (int): Desired image size (both width and height).

    Returns:
        Preprocessed image tensor.
    """
    if isinstance(input_image, str):
        # Load the image using OpenCV if input is a file path
        if not path.isfile(input_image):
            raise FileNotFoundError(f"Image file not found at: {input_image}")
        opencv_img = imread(input_image)
    elif isinstance(input_image, np.ndarray):
        # Use the provided OpenCV image if input is already opened
        opencv_img = input_image
        # imwrite("opencv_org_image.jpg", opencv_img)
    else:
        raise ValueError("Input must be a file path (str) or an already opened OpenCV image (numpy.ndarray).")

    # Calculate the new dimensions while preserving the original aspect ratio
    height, width, _ = opencv_img.shape
    aspect_ratio = width / height
    
    if aspect_ratio > 1:
        # Landscape orientation
        new_width = im_size
        new_height = int(new_width / aspect_ratio)
    else:
        # Portrait orientation
        new_height = im_size
        new_width = int(new_height * aspect_ratio)
    
    # Resize the image
    opencv_img = resize(opencv_img, (new_width, new_height))
    
    # Create a blank canvas of the desired size and paste the resized image onto it
    padded_img = zeros(3, im_size, im_size, dtype=float32)
    padded_img[:, :new_height, :new_width] = tensor(opencv_img, dtype=float32).permute(2, 0, 1) / 255.0
    
    # testing only
    # arrImg = (padded_img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)  # Conversion back to [0, 255] range
    # # cvImg = cvtColor(arrImg, COLOR_RGB2BGR)
    # imwrite("opencv_padded_image.jpg", arrImg)
    
    return padded_img    
    

def load_and_preprocess_images(input_paths, im_size=640, device='cuda:1'):
    """
    Load and preprocess images from a list of image paths or a directory path using parallel processing,
    and return a batch tensor in BCHW format with padding to the right and bottom.

    Args:
        input_paths (str or list): A single image path as a string, a list of image paths, or a directory path.
        im_size (int): Desired image size (both width and height).
        device (str): Device to move the tensors to (e.g., 'cuda:0' for GPU).
        num_processes (int): Number of parallel processes to use for image loading.

    Returns:
        Batch tensor in BCHW format containing the preprocessed images with padding.
    """
    if isinstance(input_paths, str):
        if path.isfile(input_paths):
            # Single image file
            image_paths = [input_paths]
        elif path.isdir(input_paths):
            # Directory containing image files
            image_paths = [path.join(input_paths, file) for file in listdir(input_paths) if file.endswith('.jpg')]
        else:
            raise ValueError("Input path must be a valid file or directory.")
    elif isinstance(input_paths, list):
        image_paths = input_paths
    elif isinstance(input_paths, np.ndarray):
        # Use the provided OpenCV image if input is already opened
        image_paths = [input_paths]
    else:
        raise ValueError("Input paths must be a string, a list of strings, or a directory path.")

    # Create a thread pool to parallelize image loading and preprocessing      
    with ThreadPoolExecutor() as executor:
        images_resized = executor.map(
            preprocess_image_single, image_paths, repeat(im_size)
        )

    tensor_images = list(images_resized)

    # Stack the individual tensors to create a batch tensor
    batch_tensor = stack(tensor_images, dim=0)
    
    # Move the batch tensor to the specified device
    batch_tensor = batch_tensor.to(device)

    return batch_tensor, image_paths

def model_embedding_norm(model, input_tensor, device_loc = 'cuda:0'):
    """
    Process an input tensor through a PyTorch model with the final layer (head) removed,
    then calculate the normalized mean of the output tensor's last dimension.

    Parameters:
        model (nn.Module): The PyTorch model with the head removed.
        input_tensor (torch.Tensor): A 4D input tensor (batch_size, channels, height, width).

    Returns:
        torch.Tensor: A normalized tensor of mean values along the last dimension
                      for each sample in the batch.

    Raises:
        ValueError: If the input tensor does not have 4 dimensions.
    """
    # Validate input tensor dimensions
    # if input_tensor.ndim != 4:
    #     raise ValueError("Input tensor should be 4D with shape (batch_size, channels, height, width).")
    
    if isinstance(input_tensor[0], np.ndarray):
        # Convert numpy arrays to PyTorch tensors        
        # if values greater than 1 than normal
        if any(np.max(arr) > 1 for arr in input_tensor):            
            input_tensor = [tensor(arr / 255, dtype=float32) for arr in input_tensor]
        else:
            input_tensor = [tensor(arr, dtype=float32) for arr in input_tensor]
        
    # stack the images, and permute to B,C,H,W
    batch_tensor = stack(input_tensor, dim=0).to(device_loc)
    batch_tensor = batch_tensor.permute(0, 3, 1, 2)
    
    # Process the tensor through the model
    results = model.model(batch_tensor)

    # Reshape the tensor to a 3D tensor where the last dimension is flattened
    batch_size, num_matrices, _, _ = results.shape
    reshaped_tensor = results.view(batch_size, num_matrices, -1)

    # Calculate the mean along the last dimension
    # for getting mean of the last dimension
    mean_values = reshaped_tensor.mean(dim=2)

    # Normalize each vector by dividing it by its L2 norm
    normed_mean_values = F.normalize(mean_values, p=2, dim=1)

    return normed_mean_values.tolist()

def face_embedding_norm(model, input_face_crops):
    """Run face_embedding model and normalize the output vectors

    Args:
        model (resnet model): Resnet face model
        input_tensor (tensor): Stacked tensors of cropped faces

    Returns:
        list: list of vectors representing faces
    """
    
    # using built in crops from facenet
    batch_tensor = stack(input_face_crops, dim=0)
    
    embeddings = model(batch_tensor)
    # convert to list and normalize vectors
    vector_list = F.normalize(embeddings, p=2, dim=1) 
    
    return vector_list.tolist()

## RESNET Items
class CustomDataset(Dataset):
    def __init__(self, image_paths, target_size=(640, 640)):
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(640),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data_item = self.image_paths[idx]

        if isinstance(data_item, str):
            img = Image.open(data_item)
        elif isinstance(data_item, np.ndarray):
            img = Image.fromarray(cvtColor(data_item, COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported data type. Data must be a file path or an OpenCV numpy array.")

        # Calculate padding if the image is smaller than the target size
        width, height = img.size
        padding_width = max(self.target_size[0] - width, 0)
        padding_height = max(self.target_size[1] - height, 0)

        # Resize the image to the target size before padding
        # img = transforms.Resize(self.target_size)(img)

        # Apply padding
        img = pad(img, (0, 0, padding_width, padding_height), fill=255)

        img = self.transform(img)

        # Move the image to the GPU if available
        if cuda.is_available():
            img = img.to('cuda')

        return img

def pad_image(img, max_height, max_width):
    """
    Pad an image to match the specified maximum height and width.

    Args:
        img (Tensor): The input image tensor.
        max_height (int): The maximum height for padding.
        max_width (int): The maximum width for padding.

    Returns:
        Tensor: The padded image tensor.
    """
    height_diff = max_height - img.shape[2]
    width_diff = max_width - img.shape[3]
    pad_bottom = height_diff
    pad_right = width_diff

    # Pad the image using F.pad
    padded_img = nn.functional.pad(img, (0, pad_right, 0, pad_bottom))
    return padded_img

def pad_images_to_match_largest_dimension(batch):
    """
    Pad a batch of images to match the largest width and height in the batch.

    Args:
        batch (list of Tensor): The batch of input images.

    Returns:
        Tensor: The padded batch of images.
    """
    # Find the maximum height and width in the batch
    max_height = max(tensor([img.shape[2] for img in batch]))
    max_width = max(tensor([img.shape[3] for img in batch]))

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Pad each image in parallel
        padded_images = list(executor.map(lambda img: pad_image(img, max_height, max_width), batch))

    # Convert the list of padded images back to a tensor
    padded_batch = stack(padded_images)
    return padded_batch

# Usage example:
# padded_batch = pad_images_to_match_largest_dimension(batch)



def process_tensor_RESNET(model, dataloader_input):
    """
    Process an input tensor through a model with the head removed, reshape the output,
    and calculate the mean along the last dimension of the reshaped tensor for each sample in the batch.

    Args:
        model (nn.Module): The PyTorch model with the head removed.
        input_tensor (Tensor): Input tensor to be processed.

    Returns:
        Tensor: A tensor containing the mean values along the last dimension for each sample in the batch.
    """
    
    device = next(model.parameters()).device  # Get the device of the model

    combined_results = Tensor().to(device)  # Initialize an empty tensor on the same device

    with no_grad():
        for batch in dataloader_input:
            # Run the input tensor through the model with the head removed
            results = model(batch)
            
            # Squeeze dimensions 2 and 3
            results = results.squeeze(2).squeeze(2)
            
            # Concatenate the tensor results along the desired dimension (0 for vertical stacking)
            combined_results = cat((combined_results, results), dim=0)

    # Normalize each vector by dividing it by its L2 norm (Euclidean norm)
    normed_mean_values = combined_results / norm(combined_results, p=2, dim=1, keepdim=True)

    return normed_mean_values

def get_embeddings(image_paths, emb_model, im_size=640, device='cuda:0', if_refence = False, batch_size = 32):
    """
    Load and preprocess a batch of images, then calculate feature embeddings for the batch using a model.

    Args:
        image_paths (list): A list of image paths.
        emb_model (nn.Module): The PyTorch model used for feature embedding.
        im_size (int): Desired image size (both width and height).
        device (str): Device to move the tensors to (e.g., 'cuda:0' for GPU).

    Returns:
        Tensor: Batch of feature embeddings.
    """
    # Load and preprocess the images
    batch_tensor, image_paths = load_and_preprocess_images(image_paths, im_size=im_size, device=device)
    
    if 'resnet' in emb_model.name:
        # setup dataloader for resnet
        dataset = CustomDataset(image_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        batch_feature_embeddings = process_tensor_RESNET(emb_model, dataloader)
    else:
        # Calculate feature embeddings for the batch
        batch_feature_embeddings = model_embedding_norm(emb_model, batch_tensor)
    
    # if reference image only
    if if_refence:
        ref_mean_vec = mean(batch_feature_embeddings, 0).tolist()
        return image_paths, batch_feature_embeddings.tolist(), ref_mean_vec
    
    # if images in is an np array then only return the embeddings
    if isinstance(image_paths[0], np.ndarray):
        return batch_feature_embeddings.tolist()
    
    # convert tensors to list on output
    return image_paths, batch_feature_embeddings.tolist(), None

def process_results_to_dataframe(results, model):
    """
    Process detection results and create dataframes for metadata and bounding boxes.

    Args:
        results (list): List of detection results, typically from a model.
        model: The detection model used for inference.

    Returns:
        pd.DataFrame: Dataframe containing metadata.
        pd.DataFrame: Dataframe containing bounding box information.
        list: List of Image crops from detections.
    """
    df_list = []
    df_metadata = []
    img_crops = []
    img_crop_paths = []
    index = 0

    for m_index, result in enumerate(results):
        file_uuid = uuid4().hex
        m_df = pd.DataFrame({'file_uuid': file_uuid, 'date': datetime.now(), 'path': result.path, 'height': result.orig_shape[0], 'width': result.orig_shape[1]}, index=[m_index])
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            conf = int(box.conf[0] * 100)
            bx = box.xywhn.tolist()[0]
            gen_uuid = uuid4().hex
            df = pd.DataFrame({'uuid': gen_uuid, 'file_uuid': file_uuid, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'xcenter': bx[0], 'ycenter': bx[1], 'width': bx[2], 'height': bx[3]}, index=[index])
            df_list.append(df)
            
            # crop img by the items found (car, truck)
            if cls in [2,7]:
                temp_box = [int(x) for x in box.xyxy.tolist()[0]]
                # Define xyxy coordinates
                x1, y1, x2, y2 = temp_box  # Replace with your desired coordinates

                # Ensure the coordinates are within the image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(result.orig_img.shape[1], x2)
                y2 = min(result.orig_img.shape[0], y2)

                # Crop the image using NumPy array slicing
                cropped_image = result.orig_img[y1:y2, x1:x2]

                img_crops.append(cropped_image)
                img_crop_paths.append(result.path)
                
            index += 1
        df_metadata.append(m_df)

    # Create dataframes
    df_meta = pd.concat(df_metadata)
    if len(df_list) > 0:
        df = pd.concat(df_list)
        
        # subest dataframe for cars, trucks
        sub_df_crops = df.loc[df['class_id'].isin([2,7])]
    else:
        df = pd.DataFrame()
        sub_df_crops = pd.DataFrame()
    
    return df_meta, df, img_crops, sub_df_crops, img_crop_paths

def results_to_dataframe_img_crops(results, image_paths, model):
    """
    Process detection results and create dataframes for metadata and bounding boxes.

    Args:
        results (list): List of detection results, typically from a model.
        model: The detection model used for inference.

    Returns:
        pd.DataFrame: Dataframe containing metadata.
        pd.DataFrame: Dataframe containing bounding box information.
        list: List of Image crops from detections.
    """
    df_list = []
    df_metadata = []
    img_crops = []
    index = 0

    for m_index, result in enumerate(results):
        file_uuid = uuid4().hex
        m_df = pd.DataFrame({'file_uuid': file_uuid, 'date': datetime.now(), 'height': result.orig_shape[0], 'width': result.orig_shape[1]}, index=[m_index])
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            conf = int(box.conf[0] * 100)
            bx = box.xywhn.tolist()[0]
            gen_uuid = uuid4().hex
            df_item = pd.DataFrame({'uuid': gen_uuid, 'file_uuid': file_uuid, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'xcenter': bx[0], 'ycenter': bx[1], 'width': bx[2], 'height': bx[3]}, index=[index])
            df_list.append(df_item)
            
            # crop img by the items found
            temp_box = [int(x) for x in box.xyxy.tolist()[0]]
            # Define xyxy coordinates
            x1, y1, x2, y2 = temp_box  # Replace with your desired coordinates

            # Ensure the coordinates are within the image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(result.orig_img.shape[1], x2)
            y2 = min(result.orig_img.shape[0], y2)

            # Crop the image using NumPy array slicing
            cropped_image = result.orig_img[y1:y2, x1:x2]

            img_crops.append(cropped_image)
                
            index += 1
        df_metadata.append(m_df)

    # Create dataframes
    df_meta = pd.concat(df_metadata)
    # add image paths
    df_meta['path'] = image_paths
    
    
    if len(df_list) > 0:
        df = pd.concat(df_list)
    else:
        df = pd.DataFrame()
    
    return df_meta, df, img_crops