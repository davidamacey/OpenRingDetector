# Script to run functions for ntfy

from requests import post

def file_dl_ntfy(text_input, ntfy_url = 'http://ntfy.superdave.us/ring_cam'):
    """Send notification via ntfy of new motion video at camera.

    Args:
        text_input (str): String text to display in notification
        ntfy_url (str, optional): String URL to ntfy server. Defaults to 'http://ntfy.superdave.us/ring_cam'.
    """
    
    post(url=ntfy_url, data = text_input, headers = {'title': 'Motion at Cabin', 'Tags': 'camera, car'})
    
def cleaner_ntfy(text_input, ntfy_url = 'http://ntfy.superdave.us/ring_cam'):
    """Send notification via ntfy that cleaner car is confirmed to be at the cabin.

    Args:
        text_input (str): String text to display in notification
        ntfy_url (str, optional): String URL to ntfy server. Defaults to 'http://ntfy.superdave.us/ring_cam'.
    """
    
    post(url=ntfy_url, data = text_input, headers = {'title': 'Cleaner Arrived', 'Tags': 'broom, car'})
    
def new_ref_ntfy(text_input, ntfy_url = 'http://ntfy.superdave.us/ring_cam'):
    """Send notification via ntfy that cleaner car is confirmed to be at the cabin.

    Args:
        text_input (str): String text to display in notification
        ntfy_url (str, optional): String URL to ntfy server. Defaults to 'http://ntfy.superdave.us/ring_cam'.
    """
    
    post(url=ntfy_url, data = text_input, headers = {'title': 'Reference Data Added', 'Tags': 'car'})