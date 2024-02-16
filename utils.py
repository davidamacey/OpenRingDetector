from time import time
from database_sql import clear_and_delete_all_data
from pymilvus import utility
from os import path, walk
from cv2 import resize, INTER_AREA, imread, imwrite, cvtColor, COLOR_RGB2BGR, rectangle, COLOR_BGR2RGB
import rawpy

import pyheif
from PIL import Image
from numpy import array
from concurrent.futures import ThreadPoolExecutor
from pandas import DataFrame
from random import choice
from torch import from_numpy, stack, norm, cuda
from torch.nn.functional import pad as torch_pad
import torch.nn.functional as F
from gc import collect

import numpy as np
from uuid import uuid4
from itertools import repeat

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
#
def get_chunks(*lists, chunk_size):
    """ Helper function to get chunks for multiple lists

    Args:
        chunk_size (int): Chunks of lists to return

    Returns:
        _type_: _description_
    """
    return zip(*(chunk(lst, chunk_size) for lst in lists))

def timer_func(func):
    """Timer function using a property decorator

    Args:
        func (function): Input function we want to see the time of
    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(
            f' ^^^^^^^^ ----------- Function {func.__name__!r} executed in {(t2-t1):.4f}s --------- ^^^^^^^^^^^^^'
        )
        return result

    return wrap_func

def reset_databases():
    """Reset Postgres and Milvus Collections to empty states.
    """
    
    # clear sql database
    clear_and_delete_all_data()
    
    # clear milvus database
    for coll in utility.list_collections():
        utility.drop_collection(coll)
        
def split_image_files(image_files):
    """Spilt a list of images into  two lists

    Args:
        image_files (list): List of items

    Returns:
        tuple(list, list): Tuple of 1st and 2nd halves of the list
    """
    half_length = len(image_files) // 2
    first_half = image_files[:half_length]
    second_half = image_files[half_length:]
    
    return first_half, second_half
        
def get_files(directory, file_type='image'):
    
    image_extensions = ['.heic','.cr2', '.heic', '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    video_extensions = ['.mov', '.m4a', '.mpeg', '.mpg', '.m4v', '.mp4', '.mkv', '.flv', '.wmv', '.avi', '.mov', '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.ogg', '.webm']

    if file_type == "image":
        extensions = image_extensions
    elif file_type == "video":
        extensions = video_extensions
    elif file_type == "both":
        extensions = image_extensions + video_extensions
    else:
        extensions = image_extensions
    
    return [path.join(dp, f) for dp, dn, filenames in walk(directory) for f in filenames if path.splitext(f)[1].lower() in extensions and not f.startswith('._')]
    
def find_vectors_above_threshold(vectors, reference_vector, threshold = 0.85):
    """
    Find vectors in a list that are above a specified threshold using matrix multiplication.

    Parameters:
    vectors (list of numpy arrays): List of vectors to compare against the reference vector.
    reference_vector (numpy array): The reference vector for comparison.
    threshold (float): The threshold value above which vectors are considered.

    Returns:
    list of numpy arrays: List of vectors from 'vectors' that are above the threshold.
    """
    # Convert the input list of vectors to a NumPy array
    vectors_array = np.array(vectors)
    
    # Calculate the dot product of each vector with the reference vector
    dot_products = np.dot(vectors_array, reference_vector)
    
    print('Dot product outputs from the input images')
    print(dot_products)
    
    # Find the indices of vectors that are above the threshold
    above_threshold_indices = np.where(dot_products > threshold)[0]
    
    # Extract the vectors that meet the threshold condition
    above_threshold_vectors = vectors_array[above_threshold_indices]
    
    return above_threshold_vectors

def imread_safe(file_path):
    """Open various file formats with imread

    Args:
        file_path (str): path to the file

    Returns:
        np.ndarray: Image in opencv format np.ndarray
    """
    
    ext = path.splitext(file_path)[1].lower()
    if ext == '.heic':
        heif_file = pyheif.read(file_path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
            )
        image = array(image)
        # convert to BGR
        image = image[:, :, ::-1]
        
    elif ext == '.cr2':
        # print("Processing a RAW Image...")
        with rawpy.imread(file_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
            # convert to BGR
            image = cvtColor(rgb, COLOR_RGB2BGR)
    else:
        # read in as BGR
        image = imread(file_path)
        
    return image
    

def image_resize(image, height=None, width=None, inter=INTER_AREA):
    """Resize and image while maintaining aspect ratio of the image

    NOTE: only for input to the YOLO model

    Args:
        image (str): Path to the image
        height (int, optional): Height of image in pixels. Defaults to None.
        width (int, optional): Width of image in pixels. Defaults to None.
        inter (func, optional): Function for interpolation of image resize. Defaults to INTER_AREA.

    Returns:
        OpenCV-Image: Reduced size of the image in OpenCV format
    """

    # initialize the dimensions of the image to be resized and
    # grab the image size

    ## from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

    try:
        if isinstance(image, str):
            image = imread_safe(image)
        else:
            # read in as BGR
            image = imread(image)
        
        # For testing purposes
        # imwrite("opencv_image.jpg", image)
        
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = resize(image, dim)
    except Exception as e:
        print(f'Error loading: {image} with error {e}')
        return None

    # return the resized image
    return resized

def get_max_dimension(images: [np.ndarray]) -> int:
    """
    Get the maximum dimension (either width or height) from a list of images.

    Parameters:
    - images (list of np.ndarray): List of images represented as numpy arrays.

    Returns:
    - int: The maximum dimension found among the images.
    """
    
    # Extract dimensions of each image
    dimensions = [(img.shape[0], img.shape[1]) for img in images]

    # Get the maximum height and width
    max_height = max(dim[0] for dim in dimensions)
    max_width = max(dim[1] for dim in dimensions)

    # Determine the maximum size
    max_size = max(max_height, max_width)

    return max_size

def convert_coordinates_single(image_data):
    
    original_height, original_width = image_data.img_org_shape
    
    try:
        x1, y1, x2, y2 = image_data.img_face_box
    except:
        print(image_data.img_file_path)
        print(image_data.img_face_box)
        raise "Error with box"

    # Assuming the padding is on the right and bottom, 
    # we use the original dimensions to determine the effective area in the padded image
    effective_width = original_width * (image_data.img_resized_shape[1] / max(image_data.img_org_shape))
    effective_height = original_height * (image_data.img_resized_shape[0] / max(image_data.img_org_shape))

    # Convert the coordinates relative to the effective area of the padded image
    x1_rel = x1 / effective_width * original_width
    y1_rel = y1 / effective_height * original_height
    x2_rel = x2 / effective_width * original_width
    y2_rel = y2 / effective_height * original_height

    # Calculate center, width, and height in the normalized format
    w = x2_rel - x1_rel
    h = y2_rel - y1_rel
    xc = x1_rel + (w / 2)
    yc = y1_rel + (h / 2)

    # Normalize the coordinates to be between [0, 1]
    xc_norm = xc / original_width
    yc_norm = yc / original_height
    w_norm = w / original_width
    h_norm = h / original_height

    # Create the record for the pandas dataframe
    record = {
        "uuid": uuid4().hex,
        "path": image_data.img_file_path,
        "file_uuid": image_data.img_uuid_org,
        "class_name": "face",  # Assuming class name is "face"
        "class_id": None,  # Assuming class ID for face is 1
        "confidence": int(image_data.img_face_prob * 100),
        "xcenter": xc_norm if xc_norm > 0 else 0,
        "ycenter": yc_norm if yc_norm > 0 else 0,
        "width": w_norm if w_norm > 0 else 0,
        "height": h_norm if h_norm > 0 else 0
    }

    return record

def parallel_convert_coordinates(image_data_list):
    with ThreadPoolExecutor() as executor:
        records = list(executor.map(convert_coordinates_single, image_data_list))
        
    # Convert the records to a pandas dataframe
    results = DataFrame(records)

    return results

def draw_image_with_all_boxes(detections_df, metadata_df):
    # Randomly pick a unique file_uuid
    random_file_uuid = choice(detections_df['file_uuid'].unique())

    # Lookup the corresponding file path in metadata_df
    file_path = metadata_df[metadata_df['file_uuid'] == random_file_uuid]['path'].iloc[0]

    # Load the original image
    img = imread(file_path)

    # Get the original image shape
    original_height, original_width = img.shape[:2]

    # Get all rows with the selected file_uuid
    selected_rows = detections_df[detections_df['file_uuid'] == random_file_uuid]

    for _, row in selected_rows.iterrows():
        # Get the bounding box data from the row
        xc = row['xcenter']
        yc = row['ycenter']
        w = row['width']
        h = row['height']

        # Convert the normalized coordinates to actual pixel values
        x1 = int((xc - w/2) * original_width)
        y1 = int((yc - h/2) * original_height)
        x2 = int((xc + w/2) * original_width)
        y2 = int((yc + h/2) * original_height)

        # Draw the bounding box on the image
        rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for the box

    # Save the image with the drawn boxes
    save_path=f"boxed_image_{path.basename(file_path)}_{random_file_uuid}.jpg"
    imwrite(save_path, img)
    print(f"Saved image with bounding boxes to {save_path}")

def resize_imgs4face(resized_images):
    """Wrapper for preparing images for facial detection

    Args:
        resized_images (list): List of np.ndarray images

    Returns:
        list: List of images of np.ndarray images with padding for face detect
    """
    
    max_size = get_max_dimension(resized_images)
    resized_face_imgs = pad_batch_images(resized_images, im_size = max_size, convert_to_rgb=True)
        
    return resized_face_imgs

def _crop_and_resize(row):
    """
    Given a DataFrame row, crop and resize the original image for detected face.
    """
    # Extract details from the DataFrame row
    img_path = row.path

    # Load the image
    img = imread_safe(img_path)

    # Extracting image dimensions from the loaded image
    img_height, img_width = img.shape[:2]

    # Convert normalized xywh to pixel coordinates for cropping
    xcenter, ycenter, width_x, height_x = row.xcenter, row.ycenter, row.width, row.height
    x1 = max(0, int((xcenter - width_x/2) * img_width))
    x2 = max(0, int((xcenter + width_x/2) * img_width))
    y1 = max(0, int((ycenter - height_x/2) * img_height))
    y2 = max(0, int((ycenter + height_x/2) * img_height))

    # Crop the face from the original image
    try:
        cropped_face = img[y1:y2, x1:x2]
    except:
        print(f'error cropping: {img_path}')

    # Resize the face to 160x160
    try:
        resized_face = resize(cropped_face, (160, 160))
    except Exception as e:
        print(f'{e} in resize: {img_path} {x1, y1, x2, y2}')
        
    return resized_face

def extract_faces_from_data(df, device_loc, max_workers=24):
    """
    Process rows of a DataFrame to crop and resize all detected faces.
    """
    all_faces = []

    # Use ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for face in executor.map(_crop_and_resize, df.itertuples(index=False)):
            all_faces.append(face)
    
    # convert from numpy array
    tensors = [from_numpy(np.array(arr)).permute(2, 0, 1) for arr in all_faces]
    # must convert to float for torch batching
    aligned = stack(tensors).float().to(device_loc)
    
    return aligned

def resize_and_pad(img, size=640, convert_to_rgb = False):
    # Calculate the target aspect ratio
    aspect_ratio = size / max(img.shape[0], img.shape[1])
    
    # Resize the image to fit within the square dimensions
    resized_img = resize(img, (int(img.shape[1] * aspect_ratio), int(img.shape[0] * aspect_ratio)))
    
    # Create a blank square image of the given size
    square_img = np.zeros((size, size, 3), dtype=np.uint8) if len(img.shape) == 3 else np.zeros((size, size), dtype=np.uint8)
    
    # Place the resized image on top of the square image
    square_img[0:resized_img.shape[0], 0:resized_img.shape[1]] = resized_img
    
    # convert image to BGR
    if convert_to_rgb:
        square_img = cvtColor(square_img, COLOR_BGR2RGB)
    # imwrite(f'square_padding_check_{uuid4()}.jpg', square_img)
    
    return square_img

def pad_batch_images(image_list, im_size=640, convert_to_rgb = False):
    
    # catch if not a list of images
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        
    # Create a thread pool to parallelize image loading and preprocessing      
    with ThreadPoolExecutor() as executor:
        images_resized = executor.map(
            resize_and_pad, image_list, repeat(im_size), repeat(convert_to_rgb)
        )
        
    return list(images_resized)

def pad_images_to_square(batch, device_loc = 'cuda:0'):
    """
    Pad a batch of images (as numpy arrays) to have the largest dimension within the batch and make them square.
    The padding is applied to the bottom and right sides of the images.
    The batch is converted to a PyTorch tensor with padding.

    Args:
        batch (list of np.ndarray): The batch of input images.

    Returns:
        Tensor: The padded batch of images as a PyTorch tensor.
    """
    # Find the max height and width in the batch
    max_height = max(image.shape[0] for image in batch)
    max_width = max(image.shape[1] for image in batch)

    # Determine the max dimension for making the images square
    max_size = max(max_height, max_width)

    padded_images = []
    # Pad each image to the max size
    for img in batch:
        # Calculate padding for the right and bottom
        pad_right = max_size - img.shape[1]
        pad_bottom = max_size - img.shape[0]

        # The padding has the format (left, right, top, bottom)
        padding = (0, pad_right, 0, pad_bottom)
        
        # Pad the image on the right and bottom and convert to PyTorch tensor
        # Note that we need to move the channel dimension to the first dimension if it's not already
        if img.ndim == 3 and img.shape[-1] in {1, 3}:  # If it's a 3-channel or 1-channel image
            img = img.transpose((2, 0, 1))  # Move the channel to the first dimension
        padded_image = torch_pad(from_numpy(img).to(device_loc).float() / 255, padding, 'constant', 0)
        
        # Add to list of padded images
        padded_images.append(padded_image)

    # Stack the padded images into a new tensor
    padded_batch = stack(padded_images)

    return padded_batch

# Example usage:
# Assuming images is a list of numpy arrays resized to 640 in their larger dimension
# padded_batch = pad_images_to_square(images)


def normalize_to_L2(vector):
    """
    Normalize a vector to its L2 norm.

    Args:
    - vector (list or numpy array): The input vector to be normalized.

    Returns:
    - list: The normalized vector as a list.
    """
    
    l2_norm = np.linalg.norm(vector)
    if l2_norm == 0:
        raise ValueError("Can't normalize a zero vector.")
    
    normalized_vector = vector / l2_norm
    
    return normalized_vector.tolist()  # Convert to list before returning

def normalize_batch_to_L2(batch):
    """
    Normalize a batch of vectors to their L2 norms using PyTorch, suitable for GPU computation.

    Args:
    - batch (Tensor): The input batch of vectors to be normalized.

    Returns:
    - Tensor: The batch of normalized vectors.
    """
    # Check if the input is not empty and has the right shape
    if batch.ndimension() != 2:
        raise ValueError("Input must be a 2D tensor.")

    # Move the batch to the GPU if it's not already there
    if not batch.is_cuda:
        batch = batch.cuda()

    # Calculate the L2 norms for the entire batch (along the rows)
    l2_norms = norm(batch, p=2, dim=1, keepdim=True)

    # Avoid division by zero for zero vectors
    l2_norms = l2_norms.clamp(min=1e-8)

    # Normalize the batch
    normalized_batch = batch / l2_norms

    # No need to convert to a list of lists, as PyTorch tensors work well with GPUs
    return normalized_batch.tolist()

def clean_lists(image_batch, img_resized):
    
    # clean up lists of files that won't load
    # Getting indices of None values from list1
    none_indices = [i for i, x in enumerate(img_resized) if x is None]
    # Removing None values from list1
    img_resized = [x for x in img_resized if x is not None]
    # Removing values from list2 using the indices of None values from list1
    image_files_batch = [image_batch[i] for i in range(len(image_batch)) if i not in none_indices]
    
    return image_files_batch, img_resized

# Function to remove elements from lists by index
def remove_by_index(lists, indices):
    for idx in indices:
        for list in lists:
            list.pop(idx)

def clean_and_remove_nones(data):
    """
    Removes entries from zipped data where the first item (face crop) is None.

    Args:
    - data (iterator): An iterator of zipped tuples containing the data.

    Returns:
    - list: A list of tuples with None entries removed.
    """
    # Filters out entries where the first list (faces_crops) has a None
    return [tuple(entry) for entry in data if entry[1] is not None]

def clear_memory():
    
    # Clean Up Memory
    collect()
    cuda.empty_cache()
    
def open_prepare_images(image_files_list, resize_to = 640):
    """Open, resize, and pad images for processing in main batch function

    Args:
        image_files_list (str): List of image paths for processing.
        resize_to (int): Integer size of the height of the image to size to.

    Returns:
        tuple(lists): Three lists for output, image file list, resized images to height, padded resized images in bgr format (opencv)
    """
    
    print('Resizing and padding images')
    # Resize Images with opencv in BGR format for YOLOv8 detections
    with ThreadPoolExecutor(max_workers=20) as executer:
        resized_images = list(executer.map(image_resize, image_files_list, repeat(resize_to)))
    
    # clean up lists to ensure None is removed and fildered from filenames and resized images
    image_files_list, resized_images = clean_lists(image_files_list, resized_images)
    
    # Pad resized images, outputing tensors
    # return BGR for YOLOv8
    resized_images_pad_bgr = pad_batch_images(resized_images)
    
    return image_files_list, resized_images, resized_images_pad_bgr