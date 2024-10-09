from django.shortcuts import render
import base64
import cv2
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from opencvBasics.utils.file_util import detect_face, batman_mask, moustache_filter, sunglasses_filter, nose_circle_filter

def index(request):
    return render(request, 'detectors.html')

@csrf_exempt
def face_detector(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = json.loads(data).get('imageData')
        if image_data:
            try:
                # Remover el prefijo 'data:image/png;base64,' de la cadena
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)

                # Convertir los bytes en un array de numpy para OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Procesar el fotograma (por ejemplo, detectar rostro y superponer imagen)
                processed_img = detect_face(img)

                # Codificar la imagen procesada de vuelta a base64
                _, buffer = cv2.imencode('.png', processed_img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Devolver la imagen procesada como base64
                return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

            except Exception as e:
                return JsonResponse({'exception error': str(e)})
    else:
        return render(request, 'face.html')

@csrf_exempt
def face_mask(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = json.loads(data).get('imageData')
        if image_data:
            try:
                # Remover el prefijo 'data:image/png;base64,' de la cadena
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)

                # Convertir los bytes en un array de numpy para OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Procesar el fotograma (por ejemplo, detectar rostro y superponer imagen)
                processed_img = batman_mask(img)

                # Codificar la imagen procesada de vuelta a base64
                _, buffer = cv2.imencode('.png', processed_img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Devolver la imagen procesada como base64
                return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

            except Exception as e:
                return JsonResponse({'exception error': str(e)})
    else:
        return render(request, 'mask.html')
    
def moustache_mask(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = json.loads(data).get('imageData')
        if image_data:
            try:
                # Remover el prefijo 'data:image/png;base64,' de la cadena
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)

                # Convertir los bytes en un array de numpy para OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Procesar el fotograma (por ejemplo, detectar rostro y superponer imagen)
                processed_img = moustache_filter(img)

                # Codificar la imagen procesada de vuelta a base64
                _, buffer = cv2.imencode('.png', processed_img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Devolver la imagen procesada como base64
                return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

            except Exception as e:
                return JsonResponse({'exception error': str(e)})
    else:
        return render(request, 'moustache.html')
    
def sunglasses_mask(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = json.loads(data).get('imageData')
        if image_data:
            try:
                # Remover el prefijo 'data:image/png;base64,' de la cadena
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)

                # Convertir los bytes en un array de numpy para OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Procesar el fotograma (por ejemplo, detectar rostro y superponer imagen)
                processed_img = sunglasses_filter(img)

                # Codificar la imagen procesada de vuelta a base64
                _, buffer = cv2.imencode('.png', processed_img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Devolver la imagen procesada como base64
                return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

            except Exception as e:
                return JsonResponse({'exception error': str(e)})
    else:
        return render(request, 'sunglasses.html')
    
def nose_mask(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = json.loads(data).get('imageData')
        if image_data:
            try:
                # Remover el prefijo 'data:image/png;base64,' de la cadena
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)

                # Convertir los bytes en un array de numpy para OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Procesar el fotograma (por ejemplo, detectar rostro y superponer imagen)
                processed_img = nose_circle_filter(img)

                # Codificar la imagen procesada de vuelta a base64
                _, buffer = cv2.imencode('.png', processed_img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Devolver la imagen procesada como base64
                return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

            except Exception as e:
                return JsonResponse({'exception error': str(e)})
    else:
        return render(request, 'nose.html')
    
