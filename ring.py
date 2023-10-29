# Script with functions to make connections and get information from Ring API

from json import loads, dumps
from pathlib import Path
from ring_doorbell import Ring, Auth
from ntfy import file_dl_ntfy
from os import path
from ring_doorbell.const import SNAPSHOT_ENDPOINT, SNAPSHOT_TIMESTAMP_ENDPOINT
from datetime import datetime

def ring_login_cached(project_name = 'Trinitrophenylmethylnitraminehtgydj', cache_path = "./tokens/token.cache"):
    """Function to login into Ring API with cached AWS token.

    Args:
        project_name (str, optional): String, random name for project login. Defaults to 'Trinitrophenylmethylnitramine'.
        cache_path (str, optional): String path to cached token. Defaults to "test_token.cache".

    Returns:
        ring Class or None: Returns authenticated Ring API login or none if failed.
    """
    
    cache_file = Path(cache_path)
    
    def token_updated(token):
        cache_file.write_text(dumps(token))
    
    if cache_file.is_file():
        auth = Auth(project_name, loads(cache_file.read_text()), token_updated)
        
        ring = Ring(auth)
        ring.update_data()
        
        return ring
    else:
        print('No cached token, you must create a new 2FA session.')
        return None
    
def get_latest_ring_video(connection, camera_type, camera_name, output_path = './'):
    """Query and download the latest ring video from a specific session, camera type, and name

    Args:
        connection (ringClass): Ring connection class
        camera_type (str): String name of the camera type to query
        camera_name (str): String name of the camera name within the camera_type
        
    Return
        str, None: Return the name of the file downloaded or None if not able to download
    """
    
    # get all devices on account
    cam_devices = connection.devices()
    
    # get index of camera name from the camera types query
    cam_type_idx = [i for i, x in enumerate(cam_devices[camera_type]) if x.name == camera_name][0]
    
    # camera of interest
    cam = cam_devices[camera_type][cam_type_idx]
    
    # last recording data pull and formating
    latest_metadata = cam.history(limit=1)[0]
    vid_id = latest_metadata.get('id')
    vid_date_str = latest_metadata.get('created_at').strftime('%Y-%m-%d_%H-%M-%S')
    video_filename = f'{vid_id}_-_{vid_date_str}.mp4'
    
    output_path_file = path.join(output_path, video_filename)
    
    # start file download
    try:
        cam.recording_download(vid_id, output_path_file)
        
        # if successful download, then notify channel of processing
        display_text = f'Motion video at {vid_date_str}, processing detections...'
        file_dl_ntfy(display_text)
        
        return video_filename
    except Exception as e:
        print(e)
        print(f'Video id: {vid_id} unable to download')
        return None
    
def format_timestamp_to_filename(timestamp_ms):
    """
    Converts a timestamp in milliseconds to a human-readable file-friendly time format.

    Args:
        timestamp_ms (int): The timestamp in milliseconds since the Unix epoch.

    Returns:
        str: A human-readable time string in the format 'YYYY-MM-DD_HH-MM-SS'.
    """
    timestamp_sec = timestamp_ms / 1000  # Convert milliseconds to seconds

    # Create a datetime object from the timestamp
    dt_object = datetime.fromtimestamp(timestamp_sec)

    # Format the datetime object as a file-friendly string
    file_friendly_time = dt_object.strftime('%Y-%m-%d_%H-%M-%S')

    return file_friendly_time
    

def get_latest_snapshot(ring_login, camera_type, camera_name, output_dir = None):
    """Get the latest screen capture from the camera

    Args:
        ring_conn (ring Class): Ring connection class
        camera_type (str): Ring Camera Type
        camera_name (str): Camera name in Ring app
        output_filename (str): String of file name, can include folder path.
    """
    
    # get all devices on account
    cam_devices = ring_login.devices()

    # get index of camera name from the camera types query
    cam_type_idx = [i for i, x in enumerate(cam_devices[camera_type]) if x.name == camera_name][0]

    # camera of interest
    cam = cam_devices[camera_type][cam_type_idx]
    
    snapshot = ring_login.query(
        SNAPSHOT_ENDPOINT.format(cam._attrs.get("id"))
    ).content
    
    ## keep looping until time info is received
    while True:
        payload = {"doorbot_ids": [cam._attrs.get("id")]}
        last_snap_info = ring_login.query(SNAPSHOT_TIMESTAMP_ENDPOINT, method="POST", json=payload).json()
        
        if len(last_snap_info) == 1:
            break
    
    last_snap_time = last_snap_info.get('timestamps')[0].get('timestamp')
    cor_snap_time = format_timestamp_to_filename(last_snap_time)
    
    jpg_filename = f'{camera_name}_-_{cor_snap_time}.jpg'
    
    if output_dir:
        jpg_filename = path.join(output_dir, jpg_filename)
    
    with open(jpg_filename, "wb") as jpg:
        jpg.write(snapshot)
        
    return jpg_filename