from utils import get_files, chunk, open_prepare_images
from database_vec import initialize_milvus, open_milvus
from load_models import load_all_modes
from pub_sub import ImagePublisher, ImageProcessor
from database_sql import remove_existing_image_paths

from argparse import ArgumentParser
from tqdm import tqdm
from math import ceil
from random import sample, seed
from image_processor import process_batches
from queue import Queue
from threading import Event


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
        print('Resetting Milvus collections')
        loaded_collections = initialize_milvus()
    else:  
        print('Opened milvus collections') 
        loaded_collections = open_milvus()
        

    print('Getting the files list')
    image_files = get_files(input_directory)
    
    image_files = image_files[:100]
    
    # filter based on what is alread in the database
    # image_files = remove_existing_image_paths(image_files)
    
    if image_files is None or len(image_files) == 0:
        print('No files to process')
        
        return None
    
    # image_files = ['/home/superdave/Pictures/david_cindy.jpg']
    
    print("Processing # of images:", len(image_files))
    
    # get first half of image_files
    # first_half, second_half = split_image_files(image_files)
    
    # sampling for testing purposes
    # print('Sampling the images for testing purposes')
    # seed(5439)
    # image_files = sample(image_files, 1000)
    
    number_batches = ceil(len(image_files) / batch_size)
    
    # Initialized the models  
    loaded_models = load_all_modes(0)
    
    ##### Queueing PubSub Method #######
    # real    4m19.057s
    # user    23m47.956s
    # sys     3m17.372s
    # batch of 100
    # real    4m48.345s
    # user    26m54.946s
    # sys     3m57.696s
    # shutdown_event = Event()
    
    # # Set up the queue - size can be adjusted depending on memory constraints
    # queue = Queue(maxsize=5)  # Allows up to 5 prepared batches in the buffer

    # # Start the publisher thread with preloading
    # preload_batches = 3  # Adjust this to change the number of preloaded batches
    # publisher = ImagePublisher(image_files, batch_size, queue, preload_batches, shutdown_event)
    # publisher.start()

    # # Start the processor thread
    # processor = ImageProcessor(queue, loaded_collections, loaded_models, number_batches, batch_size, shutdown_event)
    # processor.start()

    # # Wait for both threads to finish
    # publisher.join()
    # processor.join()
    
    
    ###### Original method #######
    # real    4m50.601s
    # user    20m31.453s
    # sys     2m45.092s
    # batch of 100
    # real    3m52.316s
    # user    20m40.906s
    # sys     3m3.518s
    
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
    parser.add_argument('--batch-size', type=int, default=50, help='The size of each batch of images to be processed.')
    parser.add_argument('--reset-milvus', action='store_true', help='Whether to reset Milvus collections before processing.')

    args = parser.parse_args()
    
    # Start the main process
    main(args.dir, args.batch_size, args.reset_milvus)