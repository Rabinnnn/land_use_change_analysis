from django.shortcuts import render
from django.apps import apps 
from .forms import ImageUploadForm
import cv2
import numpy as np
from PIL import Image
import torch
import base64
import traceback
import logging 
# Removed pipeline import since we are using the model directly from AppConfig
# from transformers import pipeline

logger = logging.getLogger(__name__)

# --- Category Mappings ---
SIMPLIFIED_IDS = {
    "Other": 0, "Buildings": 1, "Vegetation": 2, "Water": 3,
}
SIMPLIFIED_COLORS = {
    SIMPLIFIED_IDS["Other"]: (100, 100, 100),
    SIMPLIFIED_IDS["Buildings"]: (255, 255, 0),
    SIMPLIFIED_IDS["Vegetation"]: (0, 255, 0),
    SIMPLIFIED_IDS["Water"]: (0, 0, 255),
}

# --- Helpers ---
def center_crop(img, target_size):
    width, height = img.size
    new_w, new_h = target_size
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

def segment_image(image: Image.Image):
    app_config = apps.get_app_config('land_use_app') 
    if app_config.model_load_error:
        logger.error("Model loading failed during app startup.")
        return None

    feature_extractor = app_config.feature_extractor
    model = app_config.model

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # inputs = feature_extractor(image=image, return_tensors="pt").to(device)
        # Extract the tensor (e.g., 'pixel_values') from the returned dictionary
        inputs = feature_extractor(image=image, return_tensors="pt")

        # Move the tensor to the device (e.g., 'pixel_values' is the key with tensor)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values)
            # outputs = model(**inputs)

        logits = outputs
        logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode='bilinear',
            align_corners=False
        )
        segmentation = logits.argmax(dim=1)[0].cpu().numpy()

        return segmentation.astype(np.uint8)
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        traceback.print_exc()
        return None

# --- Basic Analysis ---
def _perform_basic_analysis(image1_cropped, image2_cropped):
    results = {}

    try:
        img1 = cv2.cvtColor(np.array(image1_cropped), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2_cropped), cv2.COLOR_RGB2BGR)

        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, thresh_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img2_with_contours = img2.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img2_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)

        red_mask = np.zeros_like(img2)
        red_mask[:, :, 2] = thresh_diff
        highlighted_image = cv2.addWeighted(img2_with_contours, 0.7, red_mask, 0.3, 0)

        changed_pixels = np.sum(thresh_diff > 0)
        total_pixels = thresh_diff.size
        percent_change = (changed_pixels / total_pixels) * 100

        results['basic_message'] = "Basic pixel difference analysis results:"
        results['basic_change_percent'] = round(percent_change, 2)

        _, buffer = cv2.imencode('.png', highlighted_image)
        results['basic_change_map'] = buffer.tobytes()

        return results
    except Exception as e:
        logger.error(f"Error during Basic Analysis: {e}")
        traceback.print_exc()
        return {'error_message': f"An error occurred during Basic Analysis: {e}"}

# --- Advanced Analysis ---
def _perform_advanced_analysis(image1: Image.Image, image2: Image.Image):
    try:
        seg1 = segment_image(image1)
        seg2 = segment_image(image2)

        if seg1 is None or seg2 is None:
            return {'error_message': 'Segmentation failed for one or both images.'}

        if seg1.shape != seg2.shape:
            seg2 = cv2.resize(seg2, (seg1.shape[1], seg1.shape[0]), interpolation=cv2.INTER_NEAREST)

        change_map = (seg1 != seg2).astype(np.uint8) * 255
        change_color = np.zeros((seg1.shape[0], seg1.shape[1], 3), dtype=np.uint8)
        change_color[change_map == 255] = [255, 0, 0]
        change_pil = Image.fromarray(change_color)

        buffer = cv2.imencode('.png', cv2.cvtColor(np.array(change_pil), cv2.COLOR_RGB2BGR))[1]
        change_base64 = base64.b64encode(buffer).decode('utf-8')

        unique_labels = set(np.unique(seg1)) | set(np.unique(seg2))
        category_changes = {}
        for label in unique_labels:
            mask1 = (seg1 == label)
            mask2 = (seg2 == label)
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            change_pct = 100 - (intersection / union * 100) if union != 0 else 0
            category_changes[f"Class {label}"] = round(change_pct, 2)

        return {
            'change_map_base64': change_base64,
            'category_changes': category_changes
        }

    except Exception as e:
        logger.error(f"Error during advanced analysis: {e}")
        traceback.print_exc()
        return {'error_message': 'Failed to perform advanced analysis.'}

# --- Main View ---
def analyze_images(request):
    context = {}

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        analysis_type = request.POST.get('analysis_type')
        logger.info(f"Received POST request. Analysis type: {analysis_type}")

        if form.is_valid():
            try:
                image1 = Image.open(request.FILES['image1']).convert('RGB')
                image2 = Image.open(request.FILES['image2']).convert('RGB')

                common_width = min(image1.width, image2.width)
                common_height = min(image1.height, image2.height)

                image1_cropped = center_crop(image1, (common_width, common_height))
                image2_cropped = center_crop(image2, (common_width, common_height))

                analysis_results = {}

                if analysis_type == 'basic':
                    analysis_results = _perform_basic_analysis(image1_cropped, image2_cropped)
                    context['analysis_type_performed'] = 'basic'

                elif analysis_type == 'advanced':
                    app_config = apps.get_app_config('land_use_app')
                    if app_config.model_load_error:
                        context['error_message'] = "Advanced Analysis model failed to load on server."
                        context['analysis_type_performed'] = 'advanced'
                        context['form'] = form 
                        return render(request, 'land_use_app/index.html', context)

                    analysis_results = _perform_advanced_analysis(image1_cropped, image2_cropped)
                    context['analysis_type_performed'] = 'advanced'

                else:
                    context['error_message'] = "Invalid analysis type specified."
                    context['form'] = form
                    return render(request, 'land_use_app/index.html', context)

                context.update(analysis_results)

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                traceback.print_exc()
                context['error_message'] = f"Server error occurred: {e}"

        context['form'] = form

    else:
        form = ImageUploadForm()
        context['form'] = form

    return render(request, 'land_use_app/index.html', context)