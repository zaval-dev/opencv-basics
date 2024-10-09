import cv2
import os
import math
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import HttpResponse
import opencvBasics.utils.file_util as utils

# Create your views here.
def index(request):
    return render(request, 'transformation.html')

def first_app(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        format_selected = request.POST.get('format')
        
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        
        base_filename = os.path.splitext(uploaded_file.name)[0]

        if format_selected.lower() == 'png':
            new_filename = f"{base_filename}_convert.png"
            third_parameter = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            new_filename = f"{base_filename}_convert.jpeg"
            third_parameter = [cv2.IMWRITE_JPEG_QUALITY, 100]
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, img, third_parameter)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'convert.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'convert.html')
    
def second_app(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        color_space = request.POST.get('color-space')
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)
        
        base_filename = os.path.splitext(uploaded_file.name)[0]

        if color_space.lower() == 'yuv':
            new_filename = f"{base_filename}_convert2yuv.jpg"
            output_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        else:
            new_filename = f"{base_filename}_convert2hsv.jpg"
            output_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'YUV&HSV.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'YUV&HSV.html')
    
def third_app(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        grade = request.POST.get('grade')
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)
        
        base_filename = os.path.splitext(uploaded_file.name)[0]
        
        new_filename = f"{base_filename}_rotate{grade}_grades.jpg"
        rows, cols = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((cols//2, rows//2), -int(grade), 0.7)
        output_image = cv2.warpAffine(img, matrix, (cols, rows))
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'rotation.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'rotation.html')
    
def quarter_app(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        factor = request.POST.get('factor')
        try:
            factor = float(factor)
        except:
            print('Error al convertir el factor')
            return render(request, 'resize.html', {
                'error' : 'Ingrese un factor de conversión válido.'
            })
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)
        
        base_filename = os.path.splitext(uploaded_file.name)[0]
        
        new_filename = f"{base_filename}_resize{factor}.jpg"
        
        if factor < 1:
            my_interpolation = cv2.INTER_AREA
        else:
            my_interpolation = cv2.INTER_CUBIC
        
        output_image = cv2.resize(img, None, fx=factor, fy=factor,interpolation=my_interpolation)
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'resize.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'resize.html')    
    
def fifth_app(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        olas = request.POST.get('olas')
        
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)
        
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f"{base_filename}_deform_{olas}.jpg"
        
        rows, cols = img.shape[:2]
        
        output_image = np.zeros(img.shape, dtype=img.dtype)
        
        if olas == 'horizontal':
            for i in range(rows):
                for j in range(cols):
                    offset_y = int(16.0 * math.sin(2*3.14*j/150))
                    if i + offset_y < rows:
                        output_image[i,j] = img[(i+offset_y)%rows,j]
                    else:
                        output_image[i,j] = 0
        elif olas == 'vertical':
            for i in range(rows):
                for j in range(cols):
                    offset_x = int(25.0 * math.sin(2*3.14*i/180))
                    if j + offset_x < rows:
                        output_image[i,j] = img[i, (j+offset_x)%cols]
                    else:
                        output_image[i,j] = 0
        else:
            for i in range(rows):
                for j in range(cols):
                    offset_x = int(20.0 * math.sin(2*3.14*i/150))
                    offset_y = int(20.0 * math.sin(2*3.14*j/150))
                    if i+offset_y < rows and j + offset_x < cols:
                        output_image[i,j] = img[(i+offset_y)%rows, (j+offset_x)%cols]
                    else:
                        output_image[i,j] = 0
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'affine.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'affine.html') 