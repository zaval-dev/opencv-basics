import cv2
import os
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import opencvBasics.utils.file_util as utils
from django.shortcuts import render

def index(request):
    return render(request, 'filters.html')

def blur_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        size = int(request.POST.get('level'))
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f"{base_filename}_blur.png"
        
        kernel_motion_blur = np.zeros((size, size)) 
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size) 
        kernel_motion_blur = kernel_motion_blur / size 

        # applying the kernel to the input image 
        output_image = cv2.filter2D(img, -1, kernel_motion_blur) 
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'blur.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'blur.html')
    
def sharp_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        format = request.POST.get('format')
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f"{base_filename}_sharp.jpg"
        print(format)
        if format == 'soft':
            kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) 
        elif format == 'hard':
            kernel_sharpen = np.array([[1,1,1], [1,-7,1], [1,1,1]]) 
        else:
            kernel_sharpen = np.array([[-1,-1,-1,-1,-1], 
                                        [-1,2,2,2,-1], 
                                        [-1,2,8,2,-1], 
                                        [-1,2,2,2,-1], 
                                        [-1,-1,-1,-1,-1]]) / 8.0 

        output_image = cv2.filter2D(img, -1, kernel_sharpen) 
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'sharp.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'sharp.html')
    
def vignette_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f"{base_filename}_vignette.jpg"
        
        rows, cols = img.shape[:2] 
        
        desv_x = cols/6
        desv_y = rows/6
        
        kernel_x = cv2.getGaussianKernel(cols,desv_x) 
        kernel_y = cv2.getGaussianKernel(rows,desv_y) 
        kernel = kernel_y * kernel_x.T 
        mask = 255 * kernel / np.linalg.norm(kernel) 
        output_image = np.copy(img) 
        
        for i in range(3): 
            output_image[:,:,i] = output_image[:,:,i] * mask 
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'vignette.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'vignette.html')
    
def erosion_dilation_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        action = request.POST.get('format')
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        
        kernel = np.ones((5,5), np.uint8)
        
        if action == 'erosion':
            new_filename = f"{base_filename}_erode.jpg"
            output_image = cv2.erode(img, kernel, iterations=1)    
        else:
            new_filename = f"{base_filename}_dilate.jpg"
            output_image = cv2.dilate(img, kernel, iterations=1)    
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'erosion_dilation.html', {   
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'erosion_dilation.html') 
        
def enhacing_contrast_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        
        img = cv2.imread(image_path)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f'{base_filename}_contrast.jpg'
        
        kernel = np.ones((5,5), np.uint8)
        
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 
 
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 
        
        output_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, output_image)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'contrast.html', {   
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'contrast.html')     