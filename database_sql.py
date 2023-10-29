from os import getenv
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from database_sql_model import Metadata, Detection, Reference, Base
from uuid import uuid4

DB_NAME = getenv("DB_NAME")
DB_USER = getenv("DB_USER")
DB_PASSWORD = getenv("DB_PASSWORD")

# Create the SQLAlchemy engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@172.21.0.4/{DB_NAME}')

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()
    
def create_tables():
    """
    Create the 'metadata', 'detections', and 'reference' tables in the database.
    """
    Base.metadata.create_all(engine)

# Single insert function for metadata
def insert_metadata(file_uuid, path, date, file_name, height, width):
    """
    Insert metadata into the 'metadata' table.

    Args:
        file_uuid (str): Unique identifier for the file.
        date (datetime.date): Date of the file.
        file_name (str): Name of the file.
        height (int): Height of the file.
        width (int): Width of the file.
    """
    new_metadata = Metadata(
        file_uuid=file_uuid,
        path=path,
        date=date,
        file_name=file_name,
        height=height,
        width=width
    )
    session.add(new_metadata)
    session.commit()

# Single insert function for detections
def insert_detection(file_uuid, class_name, class_id, confidence, xcenter, ycenter, width, height):
    """
    Insert detection data into the 'detections' table.

    Args:
        file_uuid (str): Unique identifier for the file associated with the detection.
        path (str): Path to the detection.
        class_name (str): Name of the detection class.
        class_id (int): ID of the detection class.
        confidence (float): Confidence score of the detection.
        xcenter (float): X-coordinate of the center of the detection.
        ycenter (float): Y-coordinate of the center of the detection.
        width (float): Width of the detection.
        height (float): Height of the detection.
    """
    new_detection = Detection(
        file_uuid=file_uuid,
        class_name=class_name,
        class_id=class_id,
        confidence=confidence,
        xcenter=xcenter,
        ycenter=ycenter,
        width=width,
        height=height
    )
    session.add(new_detection)
    session.commit()

# Bulk insert function for metadata
def insert_metadata_bulk(data):
    """
    Bulk insert metadata into the 'metadata' table from a DataFrame.

    Args:
        data (pandas.DataFrame): DataFrame containing metadata to be inserted.
    """
    # Convert DataFrame to a list of dictionaries
    data_to_insert = data.to_dict(orient='records')
    
    # Bulk insert the data
    stmt = Metadata.__table__.insert().values(data_to_insert)
    session.execute(stmt)
    session.commit()

# Bulk insert function for detections
def insert_detections_bulk(data):
    """
    Bulk insert detection data into the 'detections' table from a DataFrame.

    Args:
        data (pandas.DataFrame): DataFrame containing detection data to be inserted.
    """
    # Convert DataFrame to a list of dictionaries
    data_to_insert = data.to_dict(orient='records')
    
    # Bulk insert the data
    stmt = Detection.__table__.insert().values(data_to_insert)
    session.execute(stmt)
    session.commit()

def update_metadata(file_uuid, date, file_name, height, width):
    """
    Update metadata in the 'metadata' table.

    Args:
        file_uuid (str): Unique identifier for the file.
        date (datetime.date): Date of the file.
        file_name (str): Name of the file.
        height (int): Height of the file.
        width (int): Width of the file.
    """
    file_metadata = session.query(Metadata).filter_by(file_uuid=file_uuid).first()
    if file_metadata:
        file_metadata.date = date
        file_metadata.file_name = file_name
        file_metadata.height = height
        file_metadata.width = width
        session.commit()

def delete_metadata(file_uuid):
    """
    Delete metadata and associated detections from the 'metadata' and 'detections' tables (cascade delete).

    Args:
        file_uuid (str): Unique identifier for the file to delete.
    """
    metadata = session.query(Metadata).filter_by(file_uuid=file_uuid).first()
    if metadata:
        session.delete(metadata)
        session.commit()

def close_session():
    """
    Close the SQLAlchemy session.
    """
    session.close()
    
def clear_and_delete_all_data(session=session):
    """
    Clear and delete all data from all tables in the PostgreSQL database using the provided session.

    Args:
        session (sqlalchemy.orm.Session): The SQLAlchemy session connected to the database.
    """
    # Get the SQLAlchemy engine from the session
    engine = session.get_bind()

    # Get a list of all table names in the database
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_names = metadata.tables.keys()

    # Clear and delete data from each table
    for table_name in table_names:
        table = Table(table_name, metadata, autoload=True, autoload_with=engine)
        
        # Clear data from the table (truncate)
        session.execute(table.delete())

        # Optional: Delete the table itself (uncomment the line below)
        # session.execute(table.delete().where(table.c.id.isnot(None)))  # Remove the 'where' clause to delete the entire table

    # Commit the changes to the database
    session.commit()

# Function to insert a reference record
def insert_reference(collection_name, display_name, ref_vector):
    """
    Insert a new record into the 'reference' table.

    Args:
        collection_name (str): The collection name to associate with the reference.
        display_name (str): The display name for the reference.
        ref_vector (float): The reference vector to store.

    Returns:
        None
    """
    new_reference = Reference(
        uuid=uuid4().hex,
        collection_name=collection_name,
        display_name=display_name,
        ref_vector=ref_vector
    )
    session.add(new_reference)
    session.commit()
    
# Function to update a reference record
def update_reference(uuid, collection_name, display_name, ref_vector):
    """
    Update a record in the 'reference' table with the provided UUID.

    Args:
        uuid (str): The UUID of the record to update.
        collection_name (str): The new collection name to set.
        display_name (str): The new display name to set.
        ref_vector (float): The new reference vector to set.

    Returns:
        None
    """
    reference = session.query(Reference).filter_by(uuid=uuid).first()
    if reference:
        reference.collection_name = collection_name
        reference.display_name = display_name
        reference.ref_vector = ref_vector
        session.commit()

# Function to delete a reference record
def delete_reference(uuid):
    """
    Delete a record in the 'reference' table with the provided UUID.

    Args:
        uuid (str): The UUID of the record to delete.

    Returns:
        None
    """
    reference = session.query(Reference).filter_by(uuid=uuid).first()
    if reference:
        session.delete(reference)
        session.commit()


# Function to get the ref_vector by UUID
def get_ref_vector_by_uuid(uuid):
    """
    Get the 'ref_vector' from the 'reference' table based on the provided UUID.

    Args:
        uuid (str): The UUID of the record to retrieve.

    Returns:
        float: The 'ref_vector' associated with the provided UUID, or None if not found.
    """
    reference = session.query(Reference).filter_by(uuid=uuid).first()
    if reference:
        return reference.ref_vector
    return None

# Function to get the ref_vector by UUID
def get_ref_vector_by_name(text_name):
    """
    Get the 'ref_vector' from the 'reference' table based on the provided UUID.

    Args:
        uuid (str): The UUID of the record to retrieve.

    Returns:
        float: The 'ref_vector' associated with the provided UUID, or None if not found.
    """
    reference = session.query(Reference).filter_by(display_name=text_name).first()
    if reference:
        return reference.ref_vector
    return None