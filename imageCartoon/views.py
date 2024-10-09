import base64
import cv2
import numpy as np
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

def index(request):
    return render(request, 'cartooning.html')

def rectangle(request):
    if request.method == 'POST':
        # Obtener la imagen en formato base64 desde el formulario
        image_data = request.POST.get('imageData')
        rect_coords = json.loads(request.POST.get('rectCoords'))

        if image_data and rect_coords:
            # Remover el prefijo 'data:image/png;base64,' de la cadena
            image_data = image_data.split(',')[1]

            # Decodificar la imagen de base64 a bytes
            image_bytes = base64.b64decode(image_data)

            # Convertir los bytes en un array de numpy para OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decodificar la imagen en formato PNG a una imagen OpenCV
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Obtener las coordenadas del rectángulo
            x0, y0, x1, y1 = rect_coords['x0'], rect_coords['y0'], rect_coords['x1'], rect_coords['y1']

            # Invertir los colores dentro del área del rectángulo
            img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]

            # Codificar la imagen de vuelta a PNG
            _, buffer = cv2.imencode('.png', img)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Devolver la imagen procesada como base64 al cliente
            return JsonResponse({'processed_image': 'data:image/png;base64,' + processed_image_base64})

    return render(request, 'interactive.html')