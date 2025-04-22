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

logger = logging.getLogger(__name__)

# --- Category Mappings ---
# --- Simplified Category Mappings ---
SIMPLIFIED_IDS = {
    "Other": 0,
    "Buildings": 1,
    "Vegetation": 2,
    "Roads": 3,
    "Agricultural": 4,
    "Swimming Pool": 5,
    "Bare Soil": 6,
}

SIMPLIFIED_COLORS = {
    SIMPLIFIED_IDS["Other"]: (100, 100, 100),          # Grey
    SIMPLIFIED_IDS["Buildings"]: (255, 255, 0),        # Yellow
    SIMPLIFIED_IDS["Vegetation"]: (34, 139, 34),       # Forest Green
    SIMPLIFIED_IDS["Roads"]: (128, 64, 128),           # Purple-ish
    SIMPLIFIED_IDS["Agricultural"]: (218, 165, 32),    # Goldenrod
    SIMPLIFIED_IDS["Swimming Pool"]: (0, 191, 255),    # Deep Sky Blue
    SIMPLIFIED_IDS["Bare Soil"]: (139, 69, 19),        # Saddle Brown
}



# --- Mapping 15 model output classes to simplified categories ---
MODEL_TO_SIMPLIFIED = {
    0: SIMPLIFIED_IDS["Other"],            # Background / undefined
    1: SIMPLIFIED_IDS["Buildings"],        # Building
    2: SIMPLIFIED_IDS["Roads"],            # Pervious surface
    3: SIMPLIFIED_IDS["Roads"],            # Impervious surface
    4: SIMPLIFIED_IDS["Bare Soil"],        # Bare soil (moved from 'Other')
    5: SIMPLIFIED_IDS["Vegetation"],       # Coniferous
    6: SIMPLIFIED_IDS["Vegetation"],       # Deciduous
    7: SIMPLIFIED_IDS["Vegetation"],       # Brushwood
    8: SIMPLIFIED_IDS["Agricultural"],     # Vineyard
    9: SIMPLIFIED_IDS["Agricultural"],     # Herbaceous
    10: SIMPLIFIED_IDS["Agricultural"],    # Agricultural land
    11: SIMPLIFIED_IDS["Agricultural"],    # Plowed land
    12: SIMPLIFIED_IDS["Swimming Pool"],   # Swimming pool
    13: SIMPLIFIED_IDS["Other"],           # Snow
    14: SIMPLIFIED_IDS["Other"],           # Greenhouse
}




def map_to_simplified(segmentation):
    simplified_mask = np.zeros_like(segmentation)
    for model_class_id, simple_class_id in MODEL_TO_SIMPLIFIED.items():
        simplified_mask[segmentation == model_class_id] = simple_class_id
    return simplified_mask

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
        inputs = feature_extractor(image=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

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
def _blend_overlay(image, mask, color_bgr, alpha=0.5):
    overlay = image.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask, 
                                    (1 - alpha) * image[:, :, c] + alpha * color_bgr[c],
                                    image[:, :, c])
    return overlay.astype(np.uint8)



def _perform_advanced_analysis(image1: Image.Image, image2: Image.Image):
    try:
        seg1_raw = segment_image(image1)
        seg2_raw = segment_image(image2)

        if seg1_raw is None or seg2_raw is None:
            return {'error_message': 'Segmentation failed for one or both images.'}

        seg1 = map_to_simplified(seg1_raw)
        seg2 = map_to_simplified(seg2_raw)

        if seg1.shape != seg2.shape:
            seg2 = cv2.resize(seg2, (seg1.shape[1], seg1.shape[0]), interpolation=cv2.INTER_NEAREST)

        image2_np = np.array(image2.convert("RGB"))
        blended_overlay = image2_np.copy()

        total_pixels = seg1.size
        changed_pixels_total = np.sum(seg1 != seg2)
        total_change_pct = round((changed_pixels_total / total_pixels) * 100, 2)

        unique_labels = sorted(set(np.unique(seg1)) | set(np.unique(seg2)))
        category_changes = {}
        legend_items = []

        # --- Transition breakdown ---
        transition_counts = {}

        for label in unique_labels:
            name = next((k for k, v in SIMPLIFIED_IDS.items() if v == label), f"Class {label}")
            color = SIMPLIFIED_COLORS.get(label, (255, 255, 255))

            mask1 = (seg1 == label)
            mask2 = (seg2 == label)

            change_mask = np.logical_xor(mask1, mask2)

            changed_pixels = np.sum(change_mask)
            logger.info(f"Applied color {color} to {changed_pixels} pixels")

            # Change percent (Jaccard approach)
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            change_pct = 100 - (intersection / union * 100) if union != 0 else 0
            category_changes[name] = round(change_pct, 2)

            legend_items.append({
                'label': name,
                'color': '#{:02x}{:02x}{:02x}'.format(*color),
                'change_percent': round(change_pct, 2)
            })

            if np.any(change_mask):
                bgr_color = tuple(reversed(color))
                blended_overlay = _blend_overlay(blended_overlay, change_mask, bgr_color, alpha=0.5)

        # --- Compute all transitions: from -> to ---
        transition_matrix = {}
        flat_seg1 = seg1.flatten()
        flat_seg2 = seg2.flatten()

        for from_class, to_class in zip(flat_seg1, flat_seg2):
            if from_class != to_class:
                key = (from_class, to_class)
                transition_matrix[key] = transition_matrix.get(key, 0) + 1

        transition_breakdown = []
        for (from_class, to_class), count in sorted(transition_matrix.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_pixels) * 100
            from_label = next((k for k, v in SIMPLIFIED_IDS.items() if v == from_class), f"Class {from_class}")
            to_label = next((k for k, v in SIMPLIFIED_IDS.items() if v == to_class), f"Class {to_class}")
            transition_breakdown.append({
                'from': from_label,
                'to': to_label,
                'percent': round(pct, 2)
            })

        # Encode overlay
        _, buffer = cv2.imencode('.png', blended_overlay)
        advanced_change_map = base64.b64encode(buffer).decode('utf-8')

        return {
            'advanced_change_map': advanced_change_map,
            'total_change_percent': total_change_pct,
            'category_changes': category_changes,
            'legend_items': legend_items,
            'transition_breakdown': transition_breakdown
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
                    context['error_message'] = "Invalid analysis type selected."
                    context['form'] = form
                    return render(request, 'land_use_app/index.html', context)

                context.update(analysis_results)
                context['form'] = form

            except Exception as e:
                logger.error(f"Exception during image analysis: {e}")
                traceback.print_exc()
                context['error_message'] = "An unexpected error occurred during analysis."
                context['form'] = form
        else:
            context['error_message'] = "Form data is not valid."
            context['form'] = form
    else:
        context['form'] = ImageUploadForm()

    return render(request, 'land_use_app/index.html', context)
