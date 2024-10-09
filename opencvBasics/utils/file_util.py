import os
import cv2
import numpy as np
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
            
def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_gray = cv2.medianBlur(img_gray, 7) 
 
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize) 
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV) 
 
    if sketch_mode: 
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
 
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
    
    for i in range(num_repetitions): 
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space) 
 
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR) 
 
    dst = np.zeros(img_gray.shape) 
    dst = cv2.bitwise_and(img_output, img_output, mask=mask) 
    return dst 