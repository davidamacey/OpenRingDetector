# Script to process images or video media to database (embeddings and detections)
# TODO: Accept video files for processing (detect, embedding) using batch

from detect import process_results_to_dataframe, get_embeddings
from ntfy import cleaner_ntfy
from database_sql import insert_metadata_bulk, insert_detections_bulk, get_ref_vector_by_name
from utils import find_vectors_above_threshold

from load_models import *

from database_vec import detections_collection

if __name__ == '__main__': 

    ### Processing Input Images and Database the results ###
    
    print('Computing image detections and embeddings with YOLOv8')
    
    test_img = './cleaner_video-20230918/frame_0137.jpg'
    # test_img = './cleaner_video-20230918/frame_0088.jpg'
    # test_img = './img_0531.jpg'
    # test_img = './img_0558.jpg'
    
    test_img = [ele for ele in [test_img] for i in range(50)]
    
    # detections
    results = reg_model.predict(test_img, imgsz = 640,  classes = list(range(9)), stream=False) 
    
    # process the results to dataframes and image crops
    df_meta, df_detection, img_crops, df_crops_subset, image_crop_paths = process_results_to_dataframe(results, reg_model)
    
    print('Database to SQL the detections and metadata')
    # database the dataframes
    insert_metadata_bulk(df_meta)
    insert_detections_bulk(df_detection)
    
    # using image crops do the embeddings
    if len(img_crops) > 0:
        
        image_crop_embeddings_yolov8 = get_embeddings(img_crops, emb_model)
            
        # save crops to file for easy comparison check
        # for index, crop in enumerate(img_crops):
        #     imwrite(f'crop_{index}.jpg', crop)
        
        print('Inserting crop data to Milvus')
        # write the embeddings to milvus
        detections_collection.insert([df_crops_subset['uuid'].tolist(), image_crop_paths, image_crop_embeddings_yolov8])
        
        # Check in the crop image embeddings for similarity to reference image
        
        print('Getting reference vector and comparing to crops for similarity')
        # get reference image vector for comparison
        ref_data_vector = eval(get_ref_vector_by_name(text_name='cleaners_car'))
        matching_vectors = find_vectors_above_threshold(image_crop_embeddings_yolov8, ref_data_vector)
        
        if len(matching_vectors) > 0:
            print('Found the cleaners car with YOLOv8!!')
            cleaner_ntfy('Cleaners have arrived!')
        else:
            print('Cleaners are not there!')
            
    else:
        print('No image crops that are cars or trucks in media.')