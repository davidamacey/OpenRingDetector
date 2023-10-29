# Script to take folder of images and database (sql and vector)

from uuid import uuid4
from detect import get_embeddings
from ntfy import new_ref_ntfy
from database_vec import create_new_collection, milvus_similarity_search
from database_sql import insert_reference

from load_models import emb_model

## iniitalize the models on start ###

if __name__ == '__main__':     
       
    print('Creating new reference collection in Milvus fr YOLOv8')
    # create reference Collection
    new_collection = "cleaners_car"
    reference_collection = create_new_collection(new_collection)

    ### Processing Training Images and Database the results ###

    print('Computing image embeddings with YOLOv8')
    # Get Image embeddings of the reference images
    # Ref images path
    ref_input_path = './training_vehicle'

    image_paths, ref_img_embeddings, ref_vector = get_embeddings(ref_input_path, emb_model, if_refence=True)

    # create list of UUIDs for database
    ref_uuid_list = [uuid4().hex for _ in range(len(image_paths))]

    print('Database reference vectors to Milvus')
    # database the vectors to Milvus
    reference_collection.insert([ref_uuid_list, image_paths, ref_img_embeddings])

    # send notification of new reference entry
    new_ref_ntfy(f'Added new vehicle!')

    print('Database reference information to Postgres')
    # database reference to postgres
    insert_reference(new_collection,'cleaners_car', str(ref_vector))

    # query the collection for the closest reference image to the ref_vector
    reference_collection.load()

    print('Querying for most similar sample image...')
    search_results = milvus_similarity_search(reference_collection, ref_vector, 1)

    # image path extracted from results
    img_sample_path = search_results[0][0].entity.get('img_path')
    print(f'Training vehicle mean average photo: {img_sample_path}')