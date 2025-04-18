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

            # Difference analysis
            diff = cv2.absdiff(img1, img2)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

            # Draw contours around changed regions
            contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Highlight changes in red (semi-transparent overlay)
            mask = cv2.cvtColor(thresh_diff, cv2.COLOR_GRAY2BGR)
            mask[:, :, 1] = 0  # Zero green
            mask[:, :, 0] = 0  # Zero blue
            highlighted = cv2.addWeighted(img2, 0.7, mask, 0.3, 0)

            # Calculate percentage change
            changed_pixels = np.sum(thresh_diff > 0)
            total_pixels = thresh_diff.size
            percent_change = (changed_pixels / total_pixels) * 100
            context['change_percent'] = round(percent_change, 2)

            # Convert to PNG for display
            _, buffer = cv2.imencode('.png', highlighted)
            change_map = buffer.tobytes()
            context['change_map'] = change_map

    else:
        form = ImageUploadForm()

    context['form'] = form
    return render(request, 'land_use_app/input_images.html', context)
