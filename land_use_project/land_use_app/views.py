from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import numpy as np
from PIL import Image
import io

def analyze_images(request):
    context = {}
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            image1 = Image.open(request.FILES['image1'])
            image2 = Image.open(request.FILES['image2'])

            # Convert PIL to OpenCV
            img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

            # Resize to same size for comparison
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Basic difference analysis
            diff = cv2.absdiff(img1, img2)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

            # Save change map to display
            _, buffer = cv2.imencode('.png', thresh_diff)
            change_map = buffer.tobytes()
            context['change_map'] = change_map

    else:
        form = ImageUploadForm()

    context['form'] = form
    return render(request, 'land_use_app/input_images.html', context)
