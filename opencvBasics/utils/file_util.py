import os
import cv2
from django.core.files.storage import FileSystemStorage
from django.conf import settings

def save_uploaded_image(image):
    fs = FileSystemStorage()
    filename = fs.save(image.name, image)
    uploaded_image_url = fs.url(filename)
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    return uploaded_image_url, image_path

def delete_all_media():
    media_root = settings.MEDIA_ROOT
    for filename in os.listdir(media_root):
        file_path = os.path.join(media_root, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Archivo eliminado: {file_path}")
        except Exception as e:
            print(f"Error al eliminar {file_path}: {e}")