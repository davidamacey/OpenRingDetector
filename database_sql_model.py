from sqlalchemy import Column, Integer, Float, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from uuid import uuid4
from dataclasses import dataclass
from numpy import ndarray

Base = declarative_base()

# Define the Metadata and Detections tables
class Metadata(Base):
    __tablename__ = 'metadata'

    file_uuid = Column(String, primary_key=True)
    date = Column(Date)
    path = Column(String)
    file_name = Column(String)
    height = Column(Integer)
    width = Column(Integer)

    # Relationship to Detections table
    detections = relationship('Detection', back_populates='file_metadata')

class Detection(Base):
    __tablename__ = 'detections'

    uuid = Column(String, primary_key=True, default=str(uuid4()))
    file_uuid = Column(String, ForeignKey('metadata.file_uuid', ondelete='CASCADE'))
    class_name = Column(String)
    class_id = Column(Integer)
    confidence = Column(Float)
    xcenter = Column(Float)
    ycenter = Column(Float)
    width = Column(Float)
    height = Column(Float)

    # Relationship to Metadata table
    file_metadata = relationship('Metadata', back_populates='detections')

# Define the new Reference table
class Reference(Base):
    __tablename__ = 'reference'

    uuid = Column(String, primary_key=True, default=str(uuid4()))
    collection_name = Column(String)
    display_name = Column(String)
    ref_vector = Column(String)
    
# Data Class
@dataclass
class ImageData:
    img_file_path: str
    img_face_box: ndarray
    img_uuid_org: str
    img_face_prob: float
    img_org_shape: tuple
    img_resized_shape: tuple