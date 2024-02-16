from utils import get_files, image_resize, chunk, parallel_convert_coordinates, pad_images_to_square, clear_memory, split_image_files, get_chunks, clean_lists, pad_batch_images
from detect import results_to_dataframe_img_crops, model_embedding_norm, face_embedding_norm, load_and_preprocess_images
from database_sql import insert_metadata_bulk, insert_detections_bulk
from database_vec import initialize_milvus, open_milvus
from database_sql_model import ImageData
from load_models import load_all_modes

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from gc import collect
from torch import cuda, Tensor
from argparse import ArgumentParser
from tqdm import tqdm
from math import ceil
from random import sample
from numpy import ndarray, uint8
from cv2 import imwrite, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR

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
    with ThreadPoolExecutor() as executer:
        resized_images = list(executer.map(image_resize, image_files_list, repeat(resize_to)))
    
    # clean up lists to ensure None is removed and fildered from filenames and resized images
    image_files_list, resized_images = clean_lists(image_files_list, resized_images)
    
    # Pad resized images, outputing tensors
    # return BGR for YOLOv8
    resized_images_pad_bgr = pad_batch_images(resized_images)
    
    return image_files_list, resized_images, resized_images_pad_bgr
    

def process_batches(image_files_batch, resized_images, resized_images_pad_bgr, loaded_collections, loaded_models, batch_size):
    
    # print('Images: ')
    # print(image_files_batch)
    
    # Load models
    reg_model, emb_model, face_mtcnn, face_resnet, device_loc  = loaded_models
    
    # load collections
    detections_collection, fullimg_collection, facial_collection = loaded_collections
    
    if len(image_files_batch) == 0:
        print('*** No images to process in this batch. *******')
        return None
    
    print('Starting Detections')    
    # If Images start processing

    # Detect objects
    results = reg_model.predict(resized_images, imgsz = 640, device=device_loc, stream=False, verbose=False) 
    # Process results
    img_metadata, img_detects, img_crops = results_to_dataframe_img_crops(results, image_files_batch, reg_model)
    del results
    # Clean Up Memory
    clear_memory()
    
    print('Calculating embeddings.')
    # Embeddings
    # Crops
    if len(img_detects) > 0:
        
        # if len of image files is >150, must iterate through the images
        # embed and database in each loop
        
        # Compute the embedding batch size and the number of batches
        EMBEDDING_BATCH_SIZE = max(batch_size * 3, 150)
        number_batches = ceil(len(img_crops) / EMBEDDING_BATCH_SIZE) 
        
        # resize cropped items
        resized_crops_pad_bgr = pad_batch_images(img_crops)
        
        # Iterate through batches of data
        for crop_batch, uuids, file_uuids, class_names in tqdm(
            get_chunks(resized_crops_pad_bgr, img_detects['uuid'].tolist(), img_detects['file_uuid'].tolist(), img_detects['class_name'].tolist(), chunk_size=EMBEDDING_BATCH_SIZE),
            desc='Processing Crop Batches', total=number_batches
        ):                
            # Get embeddings for the current batch
            crop_embeddings_batch = model_embedding_norm(emb_model, crop_batch, device_loc)
            # Write the crop embeddings to the detections collection
            detections_collection.insert([uuids, file_uuids, crop_embeddings_batch, class_names, list(repeat('none', len(crop_embeddings_batch)))])
            
        del resized_crops_pad_bgr, crop_embeddings_batch
        # Clean Up Memory
        clear_memory()
    
    print('Calculating the full embeddings')
    # Full Image
    image_full_embeddings_yolov8 = model_embedding_norm(emb_model, resized_images_pad_bgr, device_loc)
    # write the full embeddings to milvus
    fullimg_collection.insert([img_metadata['file_uuid'].tolist(), image_files_batch, image_full_embeddings_yolov8, list(repeat('none', len(img_metadata['file_uuid']))), list(repeat('none', len(img_metadata['file_uuid'])))])
    del image_full_embeddings_yolov8
    # Clean Up Memory
    clear_memory()

    # Postgres Detections and metadata
    print('Database to SQL the detections and metadata')
    # database the dataframes
    insert_metadata_bulk(img_metadata)
    insert_detections_bulk(img_detects)        
    
    ###################### Facial Detection and Embedding  ###########################
    print('Detecting faces in images.')     
    
    # list of uuids from the image metadata to match with images with faces
    uuid_list = img_metadata['file_uuid'].tolist()  
    FACE_BATCH_SIZE = 25
    
    faces_crops = []
    image_data_list = []
    
    # convert padded images to RGB for facial detection and recognition
    resized_images_pad_rgb = [cvtColor(image, COLOR_BGR2RGB) for image in resized_images_pad_bgr]
    
    ##### WOW THIS WAS REALLY HARD, DO NOT CHANGE WITHOUT MAJOR CONSIERATION ######
    number_batches = ceil(len(resized_images_pad_rgb) / FACE_BATCH_SIZE)

    # Initial batch size
    batch_size = FACE_BATCH_SIZE

    while batch_size > 0:
        try:
            for face_imgs, uuid_item, file_item, org_img in tqdm(
                zip(chunk(resized_images_pad_rgb, batch_size), 
                    chunk(uuid_list, batch_size), 
                    chunk(image_files_batch, batch_size), 
                    chunk(resized_images, batch_size)), 
                desc='Face batches:', 
                total=number_batches
            ):
                try:
                    
                    # testing only, numpy in rgb format to be processed in the mtcnn model
                    imwrite("opencv_padded_image.jpg", cvtColor(face_imgs[0], COLOR_RGB2BGR))
                    
                    # run face detect model on face images
                    # for test of face crops add: save_path = 'face.jpg'                    
                    faces_crops_i, faces_bboxes_i, probs_i = face_mtcnn(face_imgs, return_prob=True)
                    
                    for index, (face_i, box_i, prob_i) in enumerate(zip(faces_crops_i, faces_bboxes_i, probs_i)):
                        if isinstance(face_i, Tensor) and isinstance(box_i, ndarray):
                            for crop_i2, box_i2, prob_i2 in zip(face_i, box_i, prob_i):              
                                faces_crops.append(crop_i2)
                                image_data_list.append(
                                    ImageData(
                                        img_file_path=file_item[index],
                                        img_face_box=box_i2,
                                        img_uuid_org=uuid_item[index],
                                        img_face_prob=prob_i2,
                                        img_org_shape=org_img[index].shape[:2],
                                        img_resized_shape=face_imgs[index].shape[:2]
                                    )
                                )

                except cuda.OutOfMemoryError: #type: ignore
                    # Free up CUDA memory and garbage collect
                    cuda.empty_cache()
                    collect()
                    
                    # Adjust the batch size down
                    batch_size = batch_size // 2
                    print(f"Reducing batch size to {batch_size} due to CUDA out of memory error.")

                    # Restart the loop with the new batch size
                    break

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

        # If no errors occurred in the while loop, break the while loop
        else:
            del faces_crops_i, faces_bboxes_i, probs_i
            break
    
    ############### DO NOT CHANGE ABOVE CODE  ################
    
    # if face crops, then process to database and get embeddings
    if len(faces_crops) == 0:
        print("No faces in this batch.")
        del faces_crops, resized_images_pad_rgb
        clear_memory()
        print('Memory cleared')
        
        return None     
                    
    # Databasing face detections
    print('Databasing face detections')
    # Generate the records using parallel processing
    df_faces = parallel_convert_coordinates(image_data_list)

    # Drop path column to database in postgres, path is in metadata table
    db_df_faces = df_faces.drop(columns=['path'])
    insert_detections_bulk(db_df_faces)
    
    # Prepare for face embeddings
    print('Processing face embeddings')  
    
    # using bounding boxes on original images, not the padded crops, as resolution can vary with padding, 
    # ensures good vector, must use same extracting for creaating or comparing new vectors    
    vector_list = face_embedding_norm(face_resnet, faces_crops)

    # Database the face embeddings
    print('Databasing face embeddings')
    # write the crop embeddings to milvus
    facial_collection.insert([df_faces['uuid'].tolist(), df_faces['path'].tolist(), vector_list, list(repeat('face', len(df_faces))), list(repeat('none', len(df_faces)))])
    
    del df_faces, db_df_faces, faces_crops, vector_list
    clear_memory()
    print('Memory cleared')


def main(input_directory, batch_size = 50, reset_milvus = False):
    """
    Main function to process images and interact with Milvus.

    Args:
    - input_directory (str): Directory containing images to be processed.
    - batch_size (int, optional): The size of each batch of images to process. Default is 50.
    - reset_milvus (bool, optional): Flag to reset Milvus collections. Default is False.
    """
    print('Starting main processing function')
    
    if reset_milvus:
        loaded_collections = initialize_milvus()
        print('Resetting Milvus collections')
    else:  
        loaded_collections = open_milvus()
        print('Opened milvus collections') 

    print('Getting the files list')
    image_files = get_files(input_directory)
    
    # image_files = ['/home/superdave/Pictures/david_cindy.jpg']
    
    # image_files = ['/mnt/nas/Pictures/Random Stuff/so you dance.jpg', '/mnt/nas/Pictures/Random Stuff/F14_water.jpg', '/mnt/nas/Pictures/Random Stuff/nighthawk001.jpg', '/mnt/nas/Pictures/Random Stuff/09-12-06_1244.jpg', '/mnt/nas/Pictures/Random Stuff/drinking stick figure copy.jpg', '/mnt/nas/Pictures/Random Stuff/David Wheelie in action.jpg', '/mnt/nas/Pictures/Random Stuff/PS3 stock photo.jpg', '/mnt/nas/Pictures/Random Stuff/math1.jpg', '/mnt/nas/Pictures/Random Stuff/The Unit - icon - yellah helan.jpg', '/mnt/nas/Pictures/Random Stuff/Math Bar Tour 2008 - 01 Front.jpg', '/mnt/nas/Pictures/Random Stuff/Mil ID 002.jpg', '/mnt/nas/Pictures/Random Stuff/RA Class Group Picture 1.jpg', '/mnt/nas/Pictures/Random Stuff/The Unit - icon - newbie.jpg', '/mnt/nas/Pictures/Random Stuff/The Unit - icon - hero shot.jpg', '/mnt/nas/Pictures/Random Stuff/JoePa shrine.jpg', '/mnt/nas/Pictures/Random Stuff/david marine corps ball001.jpg', '/mnt/nas/Pictures/Random Stuff/IMG_2303.JPG', '/mnt/nas/Pictures/Random Stuff/Screen Shot 2013-10-05 at 12.46.29 PM.png', '/mnt/nas/Pictures/Random Stuff/aviator wings.jpg', '/mnt/nas/Pictures/Random Stuff/chicken man 1.jpg', '/mnt/nas/Pictures/Random Stuff/Chesapeake Bay Bridges 2.jpg', '/mnt/nas/Pictures/Random Stuff/psuJoePa_edited-1.jpg', '/mnt/nas/Pictures/Random Stuff/Ranger in Snow002.jpg', '/mnt/nas/Pictures/Random Stuff/old_main2_bw.sized.jpg', '/mnt/nas/Pictures/Random Stuff/08-08-06_0803.jpg', '/mnt/nas/Pictures/Random Stuff/PS3 console.jpg', '/mnt/nas/Pictures/Random Stuff/Tail of Dragon with Corners.jpg', '/mnt/nas/Pictures/Random Stuff/CBR600F4 on a track.jpg', '/mnt/nas/Pictures/Random Stuff/09-30-06_1313.jpg', '/mnt/nas/Pictures/Random Stuff/The Unit - icon - the cave.jpg', '/mnt/nas/Pictures/Random Stuff/halp us jon carry.jpg', '/mnt/nas/Pictures/Random Stuff/FINAL - 1st Floor Locker Room Design.jpg', '/mnt/nas/Pictures/Random Stuff/Ranger in Snow003.jpg', '/mnt/nas/Pictures/Random Stuff/12-05-06_1206.jpg', '/mnt/nas/Pictures/Random Stuff/Chesapeake Bay Bridges 3.jpg', '/mnt/nas/Pictures/Random Stuff/David.jpg', '/mnt/nas/Pictures/Random Stuff/IMG_2304.JPG', '/mnt/nas/Pictures/Random Stuff/08-25-06_1441.jpg', '/mnt/nas/Pictures/Random Stuff/Ranger in Snow004.jpg', '/mnt/nas/Pictures/Random Stuff/The Unit - icon - spec ops1.jpg']
    print("Processing # of images:", len(image_files))
    
    # get first half of image_files
    # first_half, second_half = split_image_files(image_files)
    
    # sampling for testing purposes
    print('Sampling the images for testing purposes')
    image_files = sample(image_files, 300)
    
    # Initialized the models  
    loaded_models = load_all_modes(1)
    
    number_batches = ceil(len(image_files) / batch_size)  
    
    print('Starting the batch processing')
    # if len of image files is >100, must iterate through the images
    for batch in tqdm(chunk(image_files, batch_size), desc='Processing Embeddings', total = number_batches):
        
        # proprocess images
        image_files_batch, resized_images, resized_images_pad_bgr = open_prepare_images(batch, resize_to = 640)
        
        # run detections and facial recognition
        process_batches(image_files_batch, resized_images, resized_images_pad_bgr, loaded_collections, loaded_models, batch_size)
        
    print('***** Completed Processing all images *******')
    
if __name__ == '__main__':  
    
    # Parse command line arguments
    parser = ArgumentParser(description='Process images and save results to database.')
    parser.add_argument('dir', type=str, help='Directory containing images to be processed.')
    parser.add_argument('--batch_size', type=int, default=50, help='The size of each batch of images to be processed.')
    parser.add_argument('--reset_milvus', action='store_false', help='Whether to reset Milvus collections before processing.')

    args = parser.parse_args()
    
    # Start the main process
    main(args.dir, args.batch_size, args.reset_milvus)