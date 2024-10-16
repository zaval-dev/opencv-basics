import cv2
import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import opencvBasics.utils.file_util as utils

def index(request):
    return render(request, 'seam-index.html')
 
def reduce_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        num_seams = int(request.POST.get('num-seams', 50))  # Número de costuras a eliminar

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        uploaded_img = cv2.imread(image_path)
    
        img = np.copy(uploaded_img)
        img_overlay_seam = np.copy(uploaded_img)
        energy = compute_energy_matrix(img)
        print('Procesando...')
        for _ in range(num_seams):
            seam = find_vertical_seam(img, energy)
            img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
            img = remove_vertical_seam(img, seam) 
            energy = compute_energy_matrix(img) 
            img = remove_vertical_seam(img, seam)

        base_filename = os.path.splitext(uploaded_file.name)[0]
        reduced_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_reduced.jpg")
        cv2.imwrite(reduced_image_path, img)
        
        fs = FileSystemStorage()
        reduced_image_url = fs.url(f"{base_filename}_reduced.jpg")

        return render(request, 'reduce.html', {
            'original_image': uploaded_image_url,
            'converted_image': reduced_image_url,
            'download_link': reduced_image_url
        })
    return render(request, 'reduce.html')

def expand_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        num_seams = int(request.POST.get('num-seams', 50))

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        uploaded_img = cv2.imread(image_path)

        img = np.copy(uploaded_img)
        img_output = np.copy(uploaded_img)
        img_overlay_seam = np.copy(uploaded_img)
        energy = compute_energy_matrix(img)
 
        for i in range(num_seams): 
            seam = find_vertical_seam(img, energy) 
            img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
            img = remove_vertical_seam(img, seam)
            img_output = add_vertical_seam(img_output, seam, i) 
            energy = compute_energy_matrix(img) 
            print('Number of seams added =', i+1)
        
        # img_expanded = np.copy(img)
        # for _ in range(num_seams):
        #     energy = compute_energy_matrix(img)
        #     seam = find_vertical_seam(energy)
        #     img = remove_vertical_seam(img, seam)
        #     img_expanded = add_vertical_seam(img_expanded, seam, _)

        base_filename = os.path.splitext(uploaded_file.name)[0]
        expanded_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_expanded.jpg")
        cv2.imwrite(expanded_image_path, img_output)

        fs = FileSystemStorage()
        expanded_image_url = fs.url(f"{base_filename}_expanded.jpg")

        return render(request, 'expand.html', {
            'original_image': uploaded_image_url,
            'converted_image': expanded_image_url,
            'download_link': expanded_image_url
        })
    return render(request, 'expand.html')

def vertical_waves(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        uploaded_img = cv2.imread(image_path)

        img = np.copy(uploaded_img)
        img_overlay_seam = np.copy(uploaded_img)
        energy = compute_energy_matrix(img)
        
        for i in range(10):
            seam = find_vertical_seam(img, energy) 
            img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)

        base_filename = os.path.splitext(uploaded_file.name)[0]
        wave_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_vwaves.jpg")
        cv2.imwrite(wave_image_path, img_overlay_seam)

        fs = FileSystemStorage()
        wave_image_url = fs.url(f"{base_filename}_vwaves.jpg")

        return render(request, 'vertical-waves.html', {
            'original_image': uploaded_image_url,
            'converted_image': wave_image_url,
            'download_link': wave_image_url
        })
    return render(request, 'vertical-waves.html')

def horizontal_waves(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)

        img_r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        img_rotated = np.copy(img_r)
        img_overlay_seam = np.copy(img_rotated)
        energy = compute_energy_matrix(img_rotated)
        
        for i in range(10):
            seam = find_vertical_seam(img_rotated, energy) 
            img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)

        # Rotar la imagen de vuelta
        img_final = cv2.rotate(img_overlay_seam, cv2.ROTATE_90_COUNTERCLOCKWISE)

        base_filename = os.path.splitext(uploaded_file.name)[0]
        wave_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_hwaves.jpg")
        cv2.imwrite(wave_image_path, img_final)

        fs = FileSystemStorage()
        wave_image_url = fs.url(f"{base_filename}_hwaves.jpg")

        return render(request, 'horizontal-waves.html', {
            'original_image': uploaded_image_url,
            'converted_image': wave_image_url,
            'download_link': wave_image_url
        })
    return render(request, 'horizontal-waves.html')

def full_waves(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']

        uploaded_image_url, image_path = utils.save_uploaded_image(uploaded_file)
        img = cv2.imread(image_path)

        img_r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        img_rotated = np.copy(img_r)
        img_overlay_seam = np.copy(img_rotated)
        energy = compute_energy_matrix(img_rotated)
        
        for i in range(10):
            seam = find_vertical_seam(img_rotated, energy) 
            img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)

        # Rotar la imagen de vuelta
        img_final = cv2.rotate(img_overlay_seam, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        img = np.copy(img_final)
        img_overlay_seam = np.copy(img_final)
        energy = compute_energy_matrix(img_final)
        
        for i in range(10):
            seam = find_vertical_seam(img, energy) 
            img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)

        base_filename = os.path.splitext(uploaded_file.name)[0]
        wave_image_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}_fwaves.jpg")
        cv2.imwrite(wave_image_path, img_overlay_seam)

        fs = FileSystemStorage()
        wave_image_url = fs.url(f"{base_filename}_fwaves.jpg")

        return render(request, 'full-waves.html', {
            'original_image': uploaded_image_url,
            'converted_image': wave_image_url,
            'download_link': wave_image_url
        })
    return render(request, 'full-waves.html')


#Funtions

def overlay_vertical_seam(img, seam): 
    img_seam_overlay = np.copy(img)
 
    # Extract the list of points from the seam 
    x_coords, y_coords = np.transpose([(i,int(j)) for i,j in enumerate(seam)]) 
 
    # Draw a green line on the image using the list of points 
    img_seam_overlay[x_coords, y_coords] = (0,0,255) 
    return img_seam_overlay
 
# Compute the energy matrix from the input image 
def compute_energy_matrix(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Compute X derivative of the image 
    sobel_x = cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=3) 
 
    # Compute Y derivative of the image 
    sobel_y = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=3) 
 
    abs_sobel_x = cv2.convertScaleAbs(sobel_x) 
    abs_sobel_y = cv2.convertScaleAbs(sobel_y) 
 
    # Return weighted summation of the two images i.e. 0.5*X + 0.5*Y 
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0) 
 
# Find vertical seam in the input image 
def find_vertical_seam(img, energy): 
    rows, cols = img.shape[:2] 
 
    # Initialize the seam vector with 0 for each element 
    seam = np.zeros(img.shape[0]) 
 
    # Initialize distance and edge matrices 
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0,:] = np.zeros(img.shape[1]) 
    edge_to = np.zeros(img.shape[:2]) 
 
    # Dynamic programming; iterate using double loop and compute the paths efficiently 
    for row in range(rows-1): 
        for col in range(cols): 
            if col != 0 and dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]: 
                dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1]
                edge_to[row+1, col-1] = 1 
 
            if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]: 
                dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col] 
                edge_to[row+1, col] = 0 

            if col != cols-1 and dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]: 
                    dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1] 
                    edge_to[row+1, col+1] = -1 
 
    # Retracing the path 
    # Returns the indices of the minimum values along X axis.
    seam[rows-1] = np.argmin(dist_to[rows-1, :]) 
    for i in (x for x in reversed(range(rows)) if x > 0): 
        seam[i-1] = seam[i] + edge_to[i, int(seam[i])] 
 
    return seam 
 
# Remove the input vertical seam from the image 
def remove_vertical_seam(img, seam): 
    rows, cols = img.shape[:2] 
 
    # To delete a point, move every point after it one step towards the left 
    for row in range(rows): 
        for col in range(int(seam[row]), cols-1): 
            img[row, col] = img[row, col+1] 
 
    # Discard the last column to create the final output image 
    img = img[:, 0:cols-1] 
    return img 

def add_vertical_seam(img, seam, num_iter): 
    seam = seam + num_iter 
    rows, cols = img.shape[:2] 
    zero_col_mat = np.zeros((rows,1,3), dtype=np.uint8) 
    img_extended = np.hstack((img, zero_col_mat)) 
 
    for row in range(rows): 
        for col in range(cols, int(seam[row]), -1): 
            img_extended[row, col] = img[row, col-1] 
 
        # To insert a value between two columns, take the average 
        # value of the neighbors. It looks smooth this way and we 
        # can avoid unwanted artifacts. 
        for i in range(3): 
            v1 = img_extended[row, int(seam[row])-1, i] 
            v2 = img_extended[row, int(seam[row])+1, i] 
            img_extended[row, int(seam[row]), i] = (int(v1)+int(v2))/2 
 
    return img_extended