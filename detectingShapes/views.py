import cv2
import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import opencvBasics.utils.file_util as utils  # Asumiendo que tienes un utilitario para guardar imágenes

def index(request):
    return render(request, 'shapes-index.html')

def contour_matching_with_triangle(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']  # Imagen del usuario

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img2 = cv2.imread(image_path)  # Leer la imagen subida por el usuario

        img1 = cv2.imread('./detectingShapes/static/triangle_reference.png')

        ref_contour = get_ref_contour(img1)

        input_contours = get_all_contours(img2)

        closest_contour = None
        min_dist = None
        contour_img = img2.copy()

        cv2.drawContours(contour_img, input_contours, -1, color=(0, 255, 0), thickness=3)

        MATCH_THRESHOLD = 0.02
        matching_contours = []

        for i, contour in enumerate(input_contours):
            ret = cv2.matchShapes(ref_contour, contour, 3, 0.0)
            print(f"Contour {i} matches with distance {ret}")

            if ret < MATCH_THRESHOLD:
                matching_contours.append(contour)

        cv2.drawContours(img2, matching_contours, -1, color=(0, 255, 0), thickness=3)


        base_filename = os.path.splitext(uploaded_file.name)[0]
        matched_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_matched.jpg")
        cv2.imwrite(matched_image_path, img2)
        # cv2.imwrite(matched_image_path, contour_img)

        fs = FileSystemStorage()
        matched_image_url = fs.url(f"{base_filename}_matched.jpg")

        return render(request, 'contour_matching.html', {
            'original_image': uploaded_image_url,
            'converted_image': matched_image_url,
            'download_link': matched_image_url
        })

    return render(request, 'contour_matching.html')

def contours(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img2 = cv2.imread(image_path)

        img1 = cv2.imread('./detectingShapes/static/triangle_reference.png')

        ref_contour = get_ref_contour(img1)

        input_contours = get_all_contours(img2)

        contour_img = img2.copy()

        cv2.drawContours(contour_img, input_contours, -1, color=(0, 255, 0), thickness=3)


        base_filename = os.path.splitext(uploaded_file.name)[0]
        matched_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_contours.jpg")
        # cv2.imwrite(matched_image_path, img2)
        cv2.imwrite(matched_image_path, contour_img)

        fs = FileSystemStorage()
        matched_image_url = fs.url(f"{base_filename}_contours.jpg")

        return render(request, 'contours.html', {
            'original_image': uploaded_image_url,
            'converted_image': matched_image_url,
            'download_link': matched_image_url
        })

    return render(request, 'contours.html')

def contour_matching_with_circle(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']  # Imagen del usuario

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img2 = cv2.imread(image_path)  # Leer la imagen subida por el usuario

        img1 = cv2.imread('./detectingShapes/static/circle_reference.jpg')

        ref_contour = get_ref_contour(img1)

        input_contours = get_all_contours(img2)

        closest_contour = None
        min_dist = None
        contour_img = img2.copy()

        cv2.drawContours(contour_img, input_contours, -1, color=(0, 255, 0), thickness=3)

        MATCH_THRESHOLD = 0.02
        matching_contours = []

        for i, contour in enumerate(input_contours):
            ret = cv2.matchShapes(ref_contour, contour, 3, 0.0)
            print(f"Contour {i} matches with distance {ret}")

            if ret < MATCH_THRESHOLD:
                matching_contours.append(contour)

        cv2.drawContours(img2, matching_contours, -1, color=(0, 255, 0), thickness=3)


        base_filename = os.path.splitext(uploaded_file.name)[0]
        matched_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_matched.jpg")
        cv2.imwrite(matched_image_path, img2)
        # cv2.imwrite(matched_image_path, contour_img)

        fs = FileSystemStorage()
        matched_image_url = fs.url(f"{base_filename}_matched.jpg")

        return render(request, 'circle_matching.html', {
            'original_image': uploaded_image_url,
            'converted_image': matched_image_url,
            'download_link': matched_image_url
        })

    return render(request, 'circle_matching.html')

def convexity(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']  # Imagen del usuario

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)

        factor = 0.01

        for contour in get_all_contours(img):
            orig_contour = contour
            epsilon = factor * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)

            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            if defects is None:
                continue

            for i in range(defects.shape[0]):
                start_defect, end_defect, far_defect, _ = defects[i, 0]
                start = tuple(contour[start_defect][0])
                end = tuple(contour[end_defect][0])
                far = tuple(contour[far_defect][0])

                cv2.circle(img, far, 7, [255, 0, 0], -1)
                cv2.drawContours(img, [orig_contour], -1, color=(0, 0, 0), thickness=2)

        base_filename = os.path.splitext(uploaded_file.name)[0]
        matched_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_convex.jpg")
        cv2.imwrite(matched_image_path, img)
        # cv2.imwrite(matched_image_path, contour_img)

        fs = FileSystemStorage()
        matched_image_url = fs.url(f"{base_filename}_convex.jpg")

        return render(request, 'convex.html', {
            'original_image': uploaded_image_url,
            'converted_image': matched_image_url,
            'download_link': matched_image_url
        })

    return render(request, 'convex.html')

def watershed(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)

        markers = markers+1

        markers[unknown==255] = 0
        markers = cv2.watershed(img, markers)
        img[markers==-1] = [0, 255, 0]

        base_filename = os.path.splitext(uploaded_file.name)[0]
        matched_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_watershed.jpg")
        cv2.imwrite(matched_image_path, img)

        fs = FileSystemStorage()
        matched_image_url = fs.url(f"{base_filename}_watershed.jpg")

        return render(request, 'watershed.html', {
            'original_image': uploaded_image_url,
            'converted_image': matched_image_url,
            'download_link': matched_image_url
        })

    return render(request, 'watershed.html')


def get_all_contours(img): 
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0) 
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    return contours

def get_ref_contour(img):
    contours = get_all_contours(img)
    
    for contour in contours: 
        area = cv2.contourArea(contour) 
        img_area = img.shape[0] * img.shape[1] 
        if 0.05 < area/float(img_area) < 0.8: 
            return contour