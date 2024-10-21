import cv2
import os
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import opencvBasics.utils.file_util as utils
from django.shortcuts import render
import joblib

# Create your views here.
def index(request):
    return render(request, 'dectect-objetcs-index.html')

def dense_images(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']  # Imagen del usuario
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        result = dense_keypoint_detector(image_path)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        matched_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_dense.jpg")
        cv2.imwrite(matched_image_path, result)

        fs = FileSystemStorage()
        matched_image_url = fs.url(f"{base_filename}_dense.jpg")

        return render(request, 'dense.html', {
            'original_image': uploaded_image_url,
            'converted_image': matched_image_url,
            'download_link': matched_image_url
        })

    return render(request, 'dense.html')

def sift_images(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']  # Imagen del usuario
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        result = sift_keypoint_detector(image_path)
        base_filename = os.path.splitext(uploaded_file.name)[0]
        matched_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_sifted.jpg")
        cv2.imwrite(matched_image_path, result)

        fs = FileSystemStorage()
        matched_image_url = fs.url(f"{base_filename}_sifted.jpg")

        return render(request, 'd-sift.html', {
            'original_image': uploaded_image_url,
            'converted_image': matched_image_url,
            'download_link': matched_image_url
        })

    return render(request, 'd-sift.html')

def dogs(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']  # Imagen del usuario
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)
        resize_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  
        result = dog_breed_result(resize_img)
        return render(request, 'breeds.html', {
            'original_image': uploaded_image_url,
            'resultant_class': result
        })

    return render(request, 'breeds.html')

def animals(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']  # Imagen del usuario
        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)
        resize_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  
        result = dog_or_cat(resize_img)
        print(result)
        if result == 'dogs':
            result = 'perro'
        else:
            result = 'gato'
        return render(request, 'dog-cats.html', {
            'original_image': uploaded_image_url,
            'resultant_class': result
        })

    return render(request, 'dog-cats.html')

def dense_keypoint_detector(path):
    input_image = cv2.imread(path)
    # input_image = cv2.resize(input_image, (350, 350))
    step = 20
    feature_scale = 20
    image_margin = 20
    keypoints = []
    rows, cols = input_image.shape[:2]
    for y in range(image_margin, rows - image_margin, step):
        for x in range(image_margin, cols - image_margin, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), feature_scale))
    dense_image = cv2.drawKeypoints(input_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return dense_image

def sift_keypoint_detector(path):
    input_image = cv2.imread(path)
    # input_image = cv2.resize(input_image, (350, 350))
    sift_detector = cv2.SIFT_create()
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    keypoints = sift_detector.detect(gray_image, None)
    sift_image = cv2.drawKeypoints(input_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_image

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def predict_image_class(image, model, kmeans):
    descriptors = extract_sift_features(image)
    if descriptors is None:
        return "No se pudieron extraer características de la imagen."

    histogram = np.zeros(kmeans.n_clusters)
    cluster_indices = kmeans.predict(descriptors)
    for index in cluster_indices:
        histogram[index] += 1

    predicted_class = model.predict([histogram])
    return predicted_class[0]

def dog_breed_result(image):
    model = joblib.load('detectingObject/models/razas_perros.pkl')
    kmeans = joblib.load('detectingObject/models/kmeans.pkl') 

    predicted_class = predict_image_class(image, model, kmeans)
    return predicted_class

def dog_or_cat(image):
    model = joblib.load('detectingObject/models/perros_gatos.pkl')
    kmeans = joblib.load('detectingObject/models/kmeans_dc.pkl') 

    predicted_class = predict_image_class(image, model, kmeans)
    return predicted_class