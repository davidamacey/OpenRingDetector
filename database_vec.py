# script to manage milvus database connection, insert, and query

from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility
from uuid import uuid4
from utils import reset_databases

### Setup Milvus ###
print('Connecting to Milvus')
### Connect to Milvus ###
mv_conn = connections.connect(
alias="default",
host='172.18.0.2',
port='19530'
)
print('Milvus connection established')

# print('Creating new detections collection in Milvus')
# create reference Collection
# new_collection = "detections"
# detections_collection = create_new_collection(new_collection)

def create_milvus_collection(collection_name, vector_size = 576):
    """
    Create a Milvus collection with the specified collection name.

    Args:
        collection_name (str): The name of the collection to create.
        vector_size (int): The size of the collection vector length

    Returns:
        None
    """
    # Define the UUID and vector field schemas
    vector_uuid = FieldSchema(
        name="uuid",
        dtype=DataType.VARCHAR,
        max_length=100,
        default_value=uuid4(),
        is_primary=True,
    )
    vector_path = FieldSchema(
        name="img_path",
        dtype=DataType.VARCHAR,
        max_length=1064,
    )
    vector_feature = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=vector_size
    )
    vector_label = FieldSchema(
        name="label",
        dtype=DataType.VARCHAR,
        default_value='None',
        max_length=512,
    )
    vector_person_uuid = FieldSchema(
        name="person_uuid",
        dtype=DataType.VARCHAR,
        max_length=256,
    )
    

    # Define the collection schema
    schema = CollectionSchema(
        fields=[vector_uuid, vector_path, vector_feature, vector_label, vector_person_uuid],
        description="Vector and UUID for query",
        enable_dynamic_field=True
    )

    # Create the collection
    Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2
    )
    
def create_new_collection(collection_name, vector_length = 576, reset_name = False):
    """Create new milvus collection using provided name

    Args:
        collection_name (str): Name of new collection in database.
        reset_name (bool, optional): Reset the name in the database, if exists. Defaults to False.

    Returns:
        milvus collection: Created collection in milvus
    """
    
    if reset_name:
        utility.drop_collection(collection_name)
        
    create_milvus_collection(collection_name, vector_length)
    reference_collection = Collection(collection_name)

    # setup indexing for the collection
    create_milvus_index(reference_collection)
    
    return reference_collection

def create_milvus_index(collection, field_name = "embedding"):
    """
    Create an index on the specified field in a Milvus collection.

    Args:
        collection (pymilvus.Collection): The Milvus collection on which to create the index.
        field_name (str): The name of the field for which to create the index.

    Returns:
        None
    """
    # Define index parameters
    index_params = {
        "metric_type": 'L2',    # Choose the appropriate metric type (e.g., L2 or IP)
        "index_type": 'IVF_FLAT',  # Choose the appropriate index type
        "params": {"nlist": 128}   # Specify index parameters as needed
    }

    # Create the index
    collection.create_index(field_name=field_name, index_params=index_params)

def milvus_similarity_search(collection, data, limit):
    """
    Perform a similarity search in a Milvus collection with the provided data and limit (top-k).

    Args:
        collection (pymilvus.Collection): The Milvus collection on which to perform the search.
        data (list): The query data for the similarity search.
        limit (int): The maximum number of results to retrieve (top-k).

    Returns:
        list: A list of query results containing matching vectors.
    """
    # Define search parameters
    search_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
    }

    # Perform the similarity search
    results = collection.search(
        data=[data],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        expr=None,
        output_fields=['uuid', 'img_path'],
        consistency_level="Strong"
    )

    return results

def milvus_radius_search(input_collection, data_embedding, radius_search = 0.05, limit = 25):
    """Search vectors based on radius distance from reference vector

    Args:
        input_collection (collection): Input collection loaded
        data_embedding (list): List of vector to search
        radius_search (float, optional): Radius value, smaller is closer to reference. Defaults to 0.05.
        limit (int, optional): Limit of results. Defaults to 25.
    """
    
    search_params = {
    # use `L2` as the metric to calculate the distance
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {
        # search for vectors with a distance smaller than 10.0
        "radius": radius_search,
        # # filter out vectors with a distance smaller than or equal to 5.0
        # "range_filter" : 5.0
    }
}

    res = input_collection.search(data=[data_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        expr=None,
        output_fields=['uuid', 'img_path'],
        consistency_level="Strong")
    
    return res

def initialize_milvus():
    
    # # clear databases for these examples
    reset_databases()
    # print('Cleared Postgres and Milvus database')
    
    # Create Collections Required
    # Detections
    new_collection = "detections"
    detections_collection = create_new_collection(new_collection)
    
    # create Full Image Collection
    new_collection = "full_image"
    fullimg_collection = create_new_collection(new_collection)
    
    # create Full Image Collection
    new_collection = "facial"
    facial_collection = create_new_collection(new_collection, 512)
    print('Created Detections, Image, face collections.')
    
    
    return detections_collection, fullimg_collection, facial_collection

def open_milvus():

    # open Collections Required
    detections_collection = Collection('full_image')
    fullimg_collection = Collection('detections')
    facial_collection = Collection('facial')
    print('Opened Detections, Image, face collections.')
    
    return detections_collection, fullimg_collection, facial_collection