from utils import chunk, parallel_convert_coordinates, clear_memory, get_chunks, pad_batch_images
from detect import results_to_dataframe_img_crops, model_embedding_norm, face_embedding_norm
from database_sql import insert_metadata_bulk, insert_detections_bulk
from database_sql_model import ImageData

from itertools import repeat
from gc import collect
from torch import cuda, Tensor
from tqdm import tqdm
from math import ceil
from numpy import ndarray
from cv2 import imwrite, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR


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
            # detections_collection.insert([uuids, file_uuids, crop_embeddings_batch, class_names, list(repeat('none', len(crop_embeddings_batch)))])
            
        del resized_crops_pad_bgr, crop_embeddings_batch
        # Clean Up Memory
        clear_memory()
    
    print('Calculating the full embeddings')
    # Full Image
    image_full_embeddings_yolov8 = model_embedding_norm(emb_model, resized_images_pad_bgr, device_loc)
    # write the full embeddings to milvus
    # fullimg_collection.insert([img_metadata['file_uuid'].tolist(), image_files_batch, image_full_embeddings_yolov8, list(repeat('none', len(img_metadata['file_uuid']))), list(repeat('none', len(img_metadata['file_uuid'])))])
    del image_full_embeddings_yolov8
    # Clean Up Memory
    clear_memory()

    # Postgres Detections and metadata
    print('Database to SQL the detections and metadata')
    # database the dataframes
    # insert_metadata_bulk(img_metadata)
    # insert_detections_bulk(img_detects)        
    
    ###################### Facial Detection and Embedding  ###########################
    print('Detecting faces in images.')     
    
    # list of uuids from the image metadata to match with images with faces
    uuid_list = img_metadata['file_uuid'].tolist()  
    FACE_BATCH_SIZE = batch_size * 2
    
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
                    # imwrite("opencv_padded_image.jpg", cvtColor(face_imgs[0], COLOR_RGB2BGR))
                    
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

                except (cuda.OutOfMemoryError, RuntimeError): #type: ignore
                    # Free up CUDA memory and garbage collect
                    cuda.empty_cache()
                    collect()
                    
                    # Adjust the batch size down
                    batch_size = batch_size // 2
                    print(f"Reducing batch size to {batch_size} due to CUDA out of memory error or Runtime Error.")

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
        
        return None     
                    
    # Databasing face detections
    print('Databasing face detections')
    # Generate the records using parallel processing
    df_faces = parallel_convert_coordinates(image_data_list)

    # Drop path column to database in postgres, path is in metadata table
    db_df_faces = df_faces.drop(columns=['path'])
    # insert_detections_bulk(db_df_faces)
    
    # Prepare for face embeddings
    print('Processing face embeddings')  
    
    # using bounding boxes on original images, not the padded crops, as resolution can vary with padding, 
    # ensures good vector, must use same extracting for creaating or comparing new vectors    
    vector_list = face_embedding_norm(face_resnet, faces_crops)

    # Database the face embeddings
    print('Databasing face embeddings')
    # write the crop embeddings to milvus
    # facial_collection.insert([df_faces['uuid'].tolist(), df_faces['path'].tolist(), vector_list, list(repeat('face', len(df_faces))), list(repeat('none', len(df_faces)))])
    
    del df_faces, db_df_faces, faces_crops, vector_list
    clear_memory()