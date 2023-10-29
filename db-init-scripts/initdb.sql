-- Create the metadata table
CREATE TABLE metadata (
    file_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    path VARCHAR(255),
    date DATE,
    file_name VARCHAR(255),
    height INT,
    width INT
);

-- Create the detections table with a foreign key to metadata
CREATE TABLE detections (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_uuid UUID REFERENCES metadata (file_uuid) ON DELETE CASCADE,
    class_name VARCHAR(255),
    class_id INT,
    confidence FLOAT,
    xcenter FLOAT,
    ycenter FLOAT,
    width FLOAT,
    height FLOAT
);

CREATE TABLE IF NOT EXISTS reference (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_name TEXT,
    display_name TEXT,
    ref_vector TEXT
);
