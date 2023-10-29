# Script to process images or video media to database (embeddings and detections)
# TODO: Accept video files for processing (detect, embedding) using batch
from ultralytics import YOLO
from pymilvus import connections, Collection, utility
from torch import hub, nn
from cv2 import imwrite
from time import time
from statistics import mean

import ring
import detect
import ntfy
import database_vec
import database_sql


if __name__ == '__main__': 

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

    # Ref images path
    ref_input_path = './training_vehicle'

    # resnet models
    resnet_model = hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    resnet_model = nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model.eval()
    resnet_model.to('cuda:0')
    resnet_model.name = 'resnet50'

    print('Connecting to Milvus')
    ### Connect to Milvus ###
    mv_conn = connections.connect(
    alias="default",
    host='172.21.0.5',
    port='19530'
    )
    
    # connect to ring api
    # ring_login = ring.ring_login_cached()

    print('Creating new detections collection in Milvus')
    # create reference Collection
    new_collection = "detections_yolov8"
    utility.drop_collection(new_collection)
    database_vec.create_milvus_collection(new_collection, 576)
    detections_collection_yolov8 = Collection(new_collection)

    # setup indexing for the collection
    database_vec.create_milvus_index(detections_collection_yolov8)
    
    # create reference Collection
    new_collection = "detections_resnet"
    utility.drop_collection(new_collection)
    database_vec.create_milvus_collection(new_collection, 2048)
    detections_collection_resnet = Collection(new_collection)

    # setup indexing for the collection
    database_vec.create_milvus_index(detections_collection_resnet)
    
    ### Processing Input Images and Database the results ###
    
    print('Computing image detections and embeddings with YOLOv8')
    
    test_img = './cleaner_video-20230918/frame_0137.jpg'
    # test_img = './cleaner_video-20230918/frame_0088.jpg'
    # test_img = './img_0531.jpg'
    # test_img = './img_0558.jpg'
    
    test_img = [ele for ele in [test_img] for i in range(50)]
    
    # get latest camera snapshot
    # test_img = ring.get_latest_snapshot(ring_login, camera_type = 'stickup_cams', camera_name = 'Side')
    
    # detections
    results = reg_model.predict(test_img, imgsz = 640,  classes = list(range(9)), stream=False) 
    
    # process the results to dataframes and image crops
    df_meta, df_detection, img_crops, df_crops_subset, image_crop_paths = detect.process_results_to_dataframe(results, reg_model)
    
    print('Database to SQL the detections and metadata')
    # database the dataframes
    # database_sql.insert_metadata_bulk(df_meta)
    # database_sql.insert_detections_bulk(df_detection)
    
    # using image crops do the embeddings
    if len(img_crops) > 0:
        
        # image_crop_embeddings_yolov8 = detect.get_embeddings(img_crops, emb_model)
        
        # test for speed of the yolov8 emebedding, preprocess and embeddings output
        times_yolov8 = []
        for _ in range(100):
            t1 = time()
            image_crop_embeddings_yolov8 = detect.get_embeddings(img_crops, emb_model)
            t2 = time()
            times_yolov8.append(t2-t1)
        ave = mean(times_yolov8)
        print(f' ^^^ YOLOv8 time for 100 iterations (mean): {ave:.4f}')
        
        ##  ^^^ YOLOv8 time for 100 iterations (mean): 1.7467 ## with 50 images in each batch at size 640
            
        # save crops to file for easy comparison check
        # for index, crop in enumerate(img_crops):
        #     imwrite(f'crop_{index}.jpg', crop)

        # image_crop_embeddings_resnet = detect.get_embeddings(img_crops, resnet_model)
        
        # test for speed of the yolov8 emebedding, preprocess and embeddings output
        times_resnet = []
        for _ in range(100):
            t1 = time()
            image_crop_embeddings_resnet = detect.get_embeddings(img_crops, resnet_model)
            t2 = time()
            times_resnet.append(t2-t1)
        ave = mean(times_resnet)
        print(f' ^^^ ResNet time for 100 iterations (mean): {ave:.4f}')
        
        ##  ^^^ ResNet time for 100 iterations (mean): 2.6212 ## with 50 images in each batch at size 640
        
        print('Inserting crop data to Milvus')
        # write the embeddings to milvus
        detections_collection_yolov8.insert([df_crops_subset['uuid'].tolist(), image_crop_paths, image_crop_embeddings_yolov8])
        
        detections_collection_resnet.insert([df_crops_subset['uuid'].tolist(), image_crop_paths, image_crop_embeddings_resnet])
        
        # Check in the crop image embeddings for similarity to reference image
        
        print('Getting reference vector and comparing to crops for similarity')
        # get reference image vector for comparison
        ref_data = database_sql.get_ref_vector_by_name(text_name='cleaners_car_yolov8')
        ref_embedding_yolov8 = eval(ref_data)
        matching_vectors = database_sql.find_vectors_above_threshold(image_crop_embeddings_yolov8, ref_embedding_yolov8)
        
        if len(matching_vectors) > 0:
            print('Found the cleaners car with YOLOv8!!')
            ntfy.cleaner_ntfy('Cleaners have arrived!')
        else:
            print('Cleaners are not there!')
        
        ref_data_resent = database_sql.get_ref_vector_by_name(text_name='cleaners_car_resnet')
        ref_embedding_resnet = eval(ref_data_resent)
        
        x = 1
        
        matching_vectors = database_sql.find_vectors_above_threshold(image_crop_embeddings_resnet, ref_embedding_resnet)
        
        if len(matching_vectors) > 0:
            print('Found the cleaners car with ResNet!!')
            ntfy.cleaner_ntfy('Cleaners have arrived!')
        else:
            print('Cleaners are not there!')
            
    else:
        print('No image crops that are cars or trucks in media.')