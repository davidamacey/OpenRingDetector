from time import time
from database_sql import clear_and_delete_all_data
from pymilvus import utility

import numpy as np

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
    
    # clear sql database
    clear_and_delete_all_data()
    
    # clear milvus database
    for coll in utility.list_collections():
        utility.drop_collection(coll)
    
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