import cv2
import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import opencvBasics.utils.file_util as utils

# Create your views here.
def index(request):
    return render(request, 'features.html')

def detect_corners(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f'{base_filename}_corners.jpg' 
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        dst = cv2.cornerHarris(gray, blockSize=4, ksize=5, k=0.04)
        dst = cv2.dilate(dst, None)
        
        img[dst>0.01*dst.max()] = [0,255,0]
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, img)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'corners.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'corners.html')
    
def detect_main_corners(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f'{base_filename}_main_corners.jpg' 
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.05, minDistance=20)
        corners = np.float32(corners)
        
        for corner in corners:
            x, y = corner[0]
            cv2.circle(img, (int(x), int(y)), 5, (0,0,255), -1)
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, img)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'main_corners.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'main_corners.html')
    
def sift_algorithm(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f'{base_filename}_sift.jpg' 
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = sift.detect(gray, None)
        
        cv2.drawKeypoints(img, keypoints, img, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, img)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'sift.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'sift.html')
    
def fast_algorithm(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f'{base_filename}_fast.jpg' 
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
        img_keypoint_with_nonmax = img.copy()
        cv2.drawKeypoints(img, keypoints, img_keypoint_with_nonmax, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, img_keypoint_with_nonmax)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'fast.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'fast.html')
    
def orb_algorithm(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        new_filename = f'{base_filename}_main_corners.jpg' 
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create()
        keypoints = orb.detect(gray, None)
        keypoints, descriptors = orb.compute(gray, keypoints)
        cv2.drawKeypoints(img, keypoints, img, color=(0,255,0))
        
        converted_image_path = os.path.join(settings.MEDIA_ROOT, new_filename)            
        cv2.imwrite(converted_image_path, img)
        fs = FileSystemStorage()
        converted_image_url = fs.url(new_filename)
        
        return render(request, 'orb.html', {
            'original_image': uploaded_image_url,
            'converted_image': converted_image_url,
            'download_link': converted_image_url
        })
    else:
        return render(request, 'orb.html')