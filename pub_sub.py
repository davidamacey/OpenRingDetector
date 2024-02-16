from threading import Thread
from queue import Queue
from math import ceil
import time
from tqdm import tqdm
from  utils import chunk, open_prepare_images
from image_processor import process_batches

class ImagePublisher(Thread):
    def __init__(self, image_files, batch_size, queue, preload_batches, shutdown_event):
        Thread.__init__(self)
        self.image_files = image_files
        self.batch_size = batch_size
        self.queue = queue
        self.preload_batches = preload_batches
        self.daemon = True
        self.shutdown_event = shutdown_event

    def preload_queue(self):
        # Preload the queue with the initial set of prepared images
        for _ in range(min(self.preload_batches, len(self.image_files) // self.batch_size)):
            batch = next(self.image_file_generator)
            image_files_batch, resized_images, resized_images_pad_bgr = open_prepare_images(batch, resize_to=640)
            self.queue.put((image_files_batch, resized_images, resized_images_pad_bgr))

    def run(self):
        # Preloading the queue
        self.preload_queue()

        # Regular publishing loop
        for batch in self.image_file_generator:
            if self.shutdown_event.is_set():
                break
            image_files_batch, resized_images, resized_images_pad_bgr = open_prepare_images(batch, resize_to=640)
            self.queue.put((image_files_batch, resized_images, resized_images_pad_bgr))
        self.queue.put(None)  # Signal that publisher is done

    @property
    def image_file_generator(self):
        # Generator to create batches of image files
        for i in range(0, len(self.image_files), self.batch_size):
            yield self.image_files[i:i+self.batch_size]

class ImageProcessor(Thread):
    def __init__(self, queue, loaded_collections, loaded_models, expected_batches, batch_size, shutdown_event):
        Thread.__init__(self)
        self.queue = queue
        self.loaded_collections = loaded_collections
        self.loaded_models = loaded_models
        self.expected_batches = expected_batches
        self.batch_size = batch_size
        self.daemon = True
        self.shutdown_event = shutdown_event

    def run(self):
        progress_bar = tqdm(total=self.expected_batches, desc='Processing Embeddings')
        while True:
            batch_data = self.queue.get()
            if batch_data is None:  # Check for the sentinel value to stop the thread
                self.queue.task_done()
                break
            image_files_batch, resized_images, resized_images_pad_bgr = batch_data
            process_batches(image_files_batch, resized_images, resized_images_pad_bgr, self.loaded_collections, self.loaded_models, self.batch_size)
            self.queue.task_done()
            progress_bar.update(1)
        progress_bar.close()
