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

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('opencvBasics/utils/xml/haarcascade_frontalface_alt.xml')
    face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3) 
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3) 
    return img

face_mask = cv2.imread('opencvBasics/utils/images/batman-mask.png')
face_cascade = cv2.CascadeClassifier('opencvBasics/utils/xml/haarcascade_frontalface_alt.xml')

def batman_mask(img):
    # Detectar rostros en la imagen
    face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)
    
    for (x, y, w, h) in face_rects:
        h, w = int(1.6 * h), int(1.3 * w)
        y -= int(0.35 * h)
        x -= int(0.1 * w)

        if y < 0: y = 0
        if x < 0: x = 0

        img_roi = img[y:y+h, x:x+w]

        face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)

        gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)

        mask_inv = cv2.bitwise_not(mask)

        masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
        masked_img = cv2.bitwise_and(img_roi, img_roi, mask=mask_inv)

        img[y:y+h, x:x+w] = cv2.add(masked_face, masked_img)

    return img


# mouth_cascade = cv2.CascadeClassifier('opencvBasics/utils/xml/haarcascade_mcs_mouth.xml') 
# moustache_mask = cv2.imread('opencvBasics/utils/images/moustache.png')
# def moustache_filter(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
#     mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5) 
#     if len(mouth_rects) > 0: 
#         (x,y,w,h) = mouth_rects[0] 
#         h, w = int(0.6*h), int(1.2*w) 
#         x -= int(0.05*w)
#         y -= int(0.55*h)
#         img_roi = img[y:y+h, x:x+w] 
#         moustache_mask_small = cv2.resize(moustache_mask, (w, h), interpolation=cv2.INTER_AREA) 
 
#         gray_mask = cv2.cvtColor(moustache_mask_small, cv2.COLOR_BGR2GRAY) 
#         ret, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY_INV) 
#         mask_inv = cv2.bitwise_not(mask) 
#         masked_mouth = cv2.bitwise_and(moustache_mask_small, moustache_mask_small, mask=mask) 
#         masked_img = cv2.bitwise_and(img_roi, img_roi, mask=mask_inv) 
#         img[y:y+h, x:x+w] = cv2.add(masked_mouth, masked_img) 
        
#     return img

mouth_cascade = cv2.CascadeClassifier('opencvBasics/utils/xml/haarcascade_mcs_mouth.xml')
moustache_mask = cv2.imread('opencvBasics/utils/images/moustache.png')  # Cargar con transparencia si existe

def moustache_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar bocas en la imagen
    mouth_rects = mouth_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5) 
    if len(mouth_rects) > 0: 
        (x,y,w,h) = mouth_rects[0] 
        h, w = int(0.6*h), int(1.2*w) 
        x -= int(0.05*w)
        y -= int(0.55*h)
        img_roi = img[y:y+h, x:x+w] 
        moustache_mask_small = cv2.resize(moustache_mask, (w, h), interpolation=cv2.INTER_AREA) 
 
        gray_mask = cv2.cvtColor(moustache_mask_small, cv2.COLOR_BGR2GRAY) 
        ret, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY_INV) 
        mask_inv = cv2.bitwise_not(mask) 
        masked_mouth = cv2.bitwise_and(moustache_mask_small, moustache_mask_small, mask=mask) 
        masked_img = cv2.bitwise_and(img_roi, img_roi, mask=mask_inv) 
        img[y:y+h, x:x+w] = cv2.add(masked_mouth, masked_img) 
        
    return img

eye_cascade = cv2.CascadeClassifier('opencvBasics/utils/xml/haarcascade_eye.xml') 
sunglasses_img = cv2.imread('opencvBasics/utils/images/sunglasses2.png')
def sunglasses_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    centers = []

    for (x,y,w,h) in faces: 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray) 
        for (x_eye,y_eye,w_eye,h_eye) in eyes: 
            centers.append((x + int(x_eye + 0.5*w_eye), y + int(y_eye + 0.5*h_eye))) 
    
    if len(centers) > 1: # if detects both eyes
        h, w = sunglasses_img.shape[:2]
        eye_distance = abs(centers[1][0] - centers[0][0])
        sunglasses_width = 2.12 * eye_distance
        scaling_factor = sunglasses_width / w
        print(scaling_factor, eye_distance)
        overlay_sunglasses = cv2.resize(sunglasses_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0] 
     
        x -= int(0.26*overlay_sunglasses.shape[1])
        y += int(0.23*overlay_sunglasses.shape[0])
        
        h, w = overlay_sunglasses.shape[:2]
        h, w = int(h), int(w)
        img_roi = img[y:y+h, x:x+w]
        gray_overlay_sunglassess = cv2.cvtColor(overlay_sunglasses, cv2.COLOR_BGR2GRAY) 
        ret, mask = cv2.threshold(gray_overlay_sunglassess, 180, 255, cv2.THRESH_BINARY_INV) 

        mask_inv = cv2.bitwise_not(mask) 

        try:
            masked_face = cv2.bitwise_and(overlay_sunglasses, overlay_sunglasses, mask=mask) 
            masked_img = cv2.bitwise_and(img_roi, img_roi, mask=mask_inv) 
        except cv2.error as e:
            print('Ignoring arithmentic exceptions: '+ str(e))

        img[y:y+h, x:x+w] = cv2.add(masked_face, masked_img)
    else:
        print('Eyes not detected')
        
    return img

nose_cascade = cv2.CascadeClassifier('opencvBasics/utils/xml/haarcascade_mcs_nose.xml')
circle_mask = cv2.imread('opencvBasics/utils/images/nariz.png', cv2.IMREAD_UNCHANGED)
def nose_circle_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nose_rects = nose_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(nose_rects) == 0:
        print("No se detectó ninguna nariz.")
        return img

    for (x, y, w, h) in nose_rects:
        mask_width = int(2 * w)  
        mask_height = int(1.5 * h)
        resized_circle_mask = cv2.resize(circle_mask, (mask_width, mask_height), interpolation=cv2.INTER_AREA)

        if resized_circle_mask.shape[2] == 4:  
            bgr_mask = resized_circle_mask[:, :, :3]  
            alpha_mask = resized_circle_mask[:, :, 3]  
        else:
            bgr_mask = resized_circle_mask
            alpha_mask = cv2.cvtColor(bgr_mask, cv2.COLOR_BGR2GRAY)  

        _, mask = cv2.threshold(alpha_mask, 50, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Asegurar que las coordenadas no salgan de los límites de la imagen
        x_start = max(x-30, 0)
        y_start = max(y-10, 0)
        x_end = min(x_start + mask_width, img.shape[1])
        y_end = min(y_start + mask_height, img.shape[0])

        roi_color = img[y_start:y_end, x_start:x_end]

        roi_h, roi_w = roi_color.shape[:2]
        resized_circle_mask = cv2.resize(bgr_mask, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(mask_inv, (roi_w, roi_h), interpolation=cv2.INTER_AREA)

        masked_nose = cv2.bitwise_and(resized_circle_mask, resized_circle_mask, mask=mask)
        masked_img = cv2.bitwise_and(roi_color, roi_color, mask=mask_inv)

        img[y_start:y_end, x_start:x_end] = cv2.add(masked_nose, masked_img)

    return img
    # nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5) 
    # for (x,y,w,h) in nose_rects: 
    #     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3) 
    #     break 

    # return img