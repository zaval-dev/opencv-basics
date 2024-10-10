import base64
import cv2
import numpy as np
import os
import json
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import opencvBasics.utils.file_util as utils

def index(request):
    return render(request, 'cartooning.html')

def open_webcam(request):
    if request.method == 'POST':
        image_data = request.POST.get('imageData')

        if image_data:
            # Remover el prefijo 'data:image/png;base64,' de la cadena
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            _, buffer = cv2.imencode('.png', img)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

    return render(request, 'webcam.html')

def rectangle(request):
    if request.method == 'POST':
        image_data = request.POST.get('imageData')
        rect_coords = json.loads(request.POST.get('rectCoords'))

        if image_data and rect_coords:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            x0, y0, x1, y1 = rect_coords['x0'], rect_coords['y0'], rect_coords['x1'], rect_coords['y1']

            # Aqui se invierten los colores
            img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]

            _, buffer = cv2.imencode('.png', img)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Devolvemos como json la imagen
            return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

    return render(request, 'interactive.html')

def median_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f"{base_filename}_median.jpg"
        
        output_image = cv2.medianBlur(img, ksize=7)
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'median.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'median.html')
    
def bilateral_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f"{base_filename}_bilateral.jpg"
        
        rows, cols = img.shape[:2]
        factor = min(cols, rows) / 2000
        diameter = int(9*factor)
        desv_color = int(75*factor)
        desv_space = int(75*factor)
        
        output_image = cv2.bilateralFilter(img, diameter, desv_color, desv_space)
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'bilateral.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'bilateral.html')
    
def cartoonig_image(request):
    if request.method == 'POST':
        image_data = request.POST.get('imageData')

        if image_data:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            output_img = utils.cartoonize_image(img, ksize=5, sketch_mode=False)
            # converted_image_path = os.path.join(settings.MEDIA_ROOT, 'test.jpg') 
            # print(converted_image_path)           
            # cv2.imwrite(converted_image_path, output_img)
            _, buffer = cv2.imencode('.png', output_img)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

    return render(request, 'cartoon.html')