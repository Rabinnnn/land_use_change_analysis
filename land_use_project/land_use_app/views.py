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

logger = logging.getLogger(__name__) # Get a logger for this module

# --- Advanced Analysis (Segmentation) Category Mapping Definitions ---
SIMPLIFIED_IDS = {
    "Other": 0, "Buildings": 1, "Vegetation": 2, "Water": 3,
}
ADE20K_CATEGORY_NAMES = {
    1: "Buildings", 8: "Buildings",
    5: "Vegetation", 9: "Vegetation", 12: "Vegetation", 13: "Vegetation",
    14: "Water", 18: "Water", 43: "Water",
}
SIMPLIFIED_COLORS = {
    SIMPLIFIED_IDS["Other"]: (100, 100, 100), # Grey
    SIMPLIFIED_IDS["Buildings"]: (255, 255, 0), # Yellow
    SIMPLIFIED_IDS["Vegetation"]: (0, 255, 0), # Green
    SIMPLIFIED_IDS["Water"]: (0, 0, 255), # Blue
}

# --- Helper Functions ---
def center_crop(img, target_size):
    """Crops the center of a PIL Image to the target size."""
    width, height = img.size
    new_w, new_h = target_size
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    return img.crop((left, top, right, bottom))

# Modify segment_image to get model/feature_extractor from AppConfig
def segment_image(image):
    """Performs semantic segmentation using the loaded Segformer model."""
    # Access the loaded model and feature extractor from the AppConfig
    app_config = apps.get_app_config('land_use_app') 

    if app_config.model_load_error:
        logger.error("Model loading failed during app startup. Segmentation is not possible.")
        return None 

    feature_extractor = app_config.feature_extractor
    model = app_config.model

    if feature_extractor is None or model is None:
        logger.error("Model or feature extractor is None. Segmentation is not possible.")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device) # Ensure model is on the correct device

        inputs = feature_extractor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )
        segmentation = logits.argmax(dim=1)[0].cpu().numpy()

        logger.debug(f"Raw segmentation output shape: {segmentation.shape}")
        return segmentation.astype(np.uint8)

    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        traceback.print_exc() # Log traceback
        return None

# --- Analysis Logic Functions ---
def _perform_basic_analysis(image1_cropped, image2_cropped):
    """
    Performs the basic pixel difference analysis using OpenCV.
    """
    results = {}

    logger.info("Starting Basic Analysis (Pixel Difference)...")

    try:
        # Convert PIL Images (cropped) to OpenCV BGR format
        # cv2 expects BGR order, PIL is RGB
        img1 = cv2.cvtColor(np.array(image1_cropped), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2_cropped), cv2.COLOR_RGB2BGR)

        # Difference analysis (Absolute difference between the two images)
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale difference to find pixels that changed "enough"
        # Pixels with a difference greater than 50 become white (255), others black (0)
        # The threshold value (50) is arbitrary and might need tuning
        _, thresh_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img2_with_contours = img2.copy()
        # Draw red rectangles around the bounding box of each contour
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img2_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Highlight changes with a semi-transparent red overlay
        red_mask = np.zeros_like(img2)
        red_mask[:, :, 2] = thresh_diff
        highlighted_image = cv2.addWeighted(img2_with_contours, 0.7, red_mask, 0.3, 0)

        # Calculate percentage change based on the thresholded difference mask
        changed_pixels = np.sum(thresh_diff > 0)
        total_pixels = thresh_diff.size
        percent_change = (changed_pixels / total_pixels) * 100

        results['basic_message'] = "Basic pixel difference analysis results:"
        results['basic_change_percent'] = round(percent_change, 2)

        # Convert the final highlighted image to PNG bytes for display in the template
        _, buffer = cv2.imencode('.png', highlighted_image)
        results['basic_change_map'] = buffer.tobytes()

        logger.info("Basic Analysis complete.")
        return results

    except Exception as e:
        logger.error(f"Error during Basic Analysis: {e}")
        traceback.print_exc()
        return {'error_message': f"An error occurred during Basic Analysis: {e}"}


def _perform_segmentation_analysis(image1_cropped, image2_cropped):
    """
    Performs the advanced segmentation change detection using Segformer.
    """
    results = {}

    logger.info("Starting Advanced Analysis (Segmentation Change Detection)...")

    # 1. Get raw segmentation masks (calls segment_image which handles model access)
    output1_raw = segment_image(image1_cropped)
    output2_raw = segment_image(image2_cropped)

    if output1_raw is None or output2_raw is None:
        logger.error("Segmentation output was None. Cannot proceed with Segmentation Analysis.")
        return {'error_message': "Segmentation failed for Advanced Analysis. Check server logs."}

    logger.debug("Output1 raw shape: %s", output1_raw.shape)
    logger.debug("Output2 raw shape: %s", output2_raw.shape)
    logger.debug("Unique raw labels in image1: %s", np.unique(output1_raw))
    logger.debug("Unique raw labels in image2: %s", np.unique(output2_raw))

    # 2. Apply the improved mapping from raw IDs to single simplified IDs
    max_raw_id = 0
    if ADE20K_CATEGORY_NAMES:
        max_raw_id = max(output1_raw.max(), output2_raw.max(), max(ADE20K_CATEGORY_NAMES.keys()))
    else:
         max_raw_id = max(output1_raw.max(), output2_raw.max())

    simplified_map = np.full(max_raw_id + 1, SIMPLIFIED_IDS["Other"], dtype=np.uint8)

    for raw_id, category_name in ADE20K_CATEGORY_NAMES.items():
        if category_name in SIMPLIFIED_IDS:
            simplified_id = SIMPLIFIED_IDS[category_name]
            if raw_id <= max_raw_id:
               simplified_map[raw_id] = simplified_id
            else:
               logger.warning(f"Raw ID {raw_id} from ADE20K_CATEGORY_NAMES exceeds map size {max_raw_id}. Skipping map entry.")
        else:
             logger.warning(f"Category name '{category_name}' found in ADE20K_CATEGORY_NAMES but not in SIMPLIFIED_IDS. Raw ID {raw_id} maps to 'Other'.")

    output1_simplified = simplified_map[output1_raw]
    output2_simplified = simplified_map[output2_raw]

    logger.debug("Unique simplified labels in image1: %s", np.unique(output1_simplified))
    logger.debug("Unique simplified labels in image2: %s", np.unique(output2_simplified))

    # 3. Improved Change Detection based on simplified IDs
    changed_mask_simplified = (output1_simplified != output2_simplified)

    # 4. Calculate total change percentage
    total_change_percent = round((np.sum(changed_mask_simplified) / changed_mask_simplified.size) * 100, 2)
    results['segmentation_change_percent'] = total_change_percent
    logger.info(f"Total Changed Pixels (Segmentation Simplified): {np.sum(changed_mask_simplified)} | Total Percent: {total_change_percent:.2f}%")

    # 5. Calculate Change Summary by Transition Type (Segmentation specific)
    change_summary = {}
    total_pixels = changed_mask_simplified.size
    all_possible_simplified_ids = np.unique(np.concatenate((np.unique(output1_simplified), np.unique(output2_simplified))))
    simplified_id_to_name = {v: k for k, v in SIMPLIFIED_IDS.items()}

    logger.debug("Calculating segmentation change summary...")
    for id1 in all_possible_simplified_ids:
        for id2 in all_possible_simplified_ids:
            if id1 != id2:
                transition_mask = (output1_simplified == id1) & (output2_simplified == id2)
                count = np.sum(transition_mask)
                if count > 0:
                    name1 = simplified_id_to_name.get(id1, f"Unknown_{id1}")
                    name2 = simplified_id_to_name.get(id2, f"Unknown_{id2}")
                    transition_key = f"{name1} -> {name2}"
                    change_summary[transition_key] = round((count / total_pixels) * 100, 2)
                    logger.debug(f"Transition {transition_key}: {count} pixels ({change_summary[transition_key]:.2f}%)")

    results['segmentation_change_summary'] = dict(sorted(change_summary.items(), key=lambda item: item[1], reverse=True))
    logger.info("Segmentation Change summary calculated.")

    # 6. Visualization (Coloring changes by final simplified category - Segmentation specific)
    output_image_base = np.array(image2_cropped)
    overlay_color_image = np.zeros_like(output_image_base)
    changed_mask_3channel = np.stack([changed_mask_simplified] * 3, axis=-1)

    if np.sum(changed_mask_simplified) > 0:
        simplified_ids_in_changed_areas = np.unique(output2_simplified[changed_mask_simplified])
    else:
        simplified_ids_in_changed_areas = []

    logger.debug(f"Simplified IDs in changed areas (Image 2, will be colored): {simplified_ids_in_changed_areas}")

    for simplified_id in simplified_ids_in_changed_areas:
        color = SIMPLIFIED_COLORS.get(simplified_id, (128, 128, 128))
        logger.debug(f"Preparing color {color} for changes ending in simplified ID {simplified_id}")
        mask_for_this_color = changed_mask_simplified & (output2_simplified == simplified_id)
        overlay_color_image[mask_for_this_color] = color 

    final_output_image = np.where(
        changed_mask_3channel,
        (0.7 * output_image_base + 0.3 * overlay_color_image).astype(np.uint8),
        output_image_base
    )

    # 7. Encode the resulting image (Segmentation specific map)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(final_output_image, cv2.COLOR_RGB2BGR))
    results['segmentation_change_map'] = buffer.tobytes()

    # 8. Prepare legend data for template (Segmentation specific legend)
    simplified_legend_items = []
    for name, sim_id in SIMPLIFIED_IDS.items():
        color_rgb = SIMPLIFIED_COLORS.get(sim_id, (128, 128, 128))
        color_hex = '#%02x%02x%02x' % color_rgb
        simplified_legend_items.append({'name': name, 'color': color_hex})
    simplified_legend_items.sort(key=lambda item: item['name'])
    results['segmentation_legend_items'] = simplified_legend_items

    logger.info("Advanced Analysis (Segmentation) complete.")
    return results

# --- Main Django View Function ---
def analyze_images(request):
   
    context = {}

    if request.method == 'POST':
    
        # if 'reset' in request.POST:
        #     logger.info("Reset button clicked. Returning empty form.")
        #     # Create a fresh, empty form
        #     form = ImageUploadForm()
        #     context = {'form': form}
        #     return render(request, 'land_use_app/input_images.html', context)
        
        form = ImageUploadForm(request.POST, request.FILES)
       
        analysis_type = request.POST.get('analysis_type')
        logger.info(f"Received POST request. Analysis type: {analysis_type}")
        logger.debug(f"Value received for 'analysis_type': '{analysis_type}' (Type: {type(analysis_type)})") 


        if form.is_valid():
            try:
                # Load and crop images for uniformity
                image1 = Image.open(request.FILES['image1']).convert('RGB')
                image2 = Image.open(request.FILES['image2']).convert('RGB')

                common_width = min(image1.width, image2.width)
                common_height = min(image1.height, image2.height)

                image1_cropped = center_crop(image1, (common_width, common_height))
                image2_cropped = center_crop(image2, (common_width, common_height))

                logger.info(f"Images loaded and cropped to {common_width}x{common_height} for analysis.")

                # --- Call the appropriate analysis function ---
                analysis_results = {}
                if analysis_type == 'basic':
                    # Call the function for the pixel difference analysis
                    analysis_results = _perform_basic_analysis(image1_cropped, image2_cropped)
                    context['analysis_type_performed'] = 'basic'

                elif analysis_type == 'advanced':
                    # Call the function for the segmentation analysis
                    # First, check if the model loaded successfully during app startup
                    app_config = apps.get_app_config('land_use_app') # <-- Use your actual app name
                    if app_config.model_load_error:
                         logger.error("Cannot perform Advanced Analysis: Model failed to load during startup.")
                         context['error_message'] = "Advanced Analysis requires the Segformer model, which failed to load on the server. Please contact the administrator."
                         context['analysis_type_performed'] = 'advanced'
                         context['form'] = form 
                         return render(request, 'land_use_app/input_images.html', context)

                    analysis_results = _perform_segmentation_analysis(image1_cropped, image2_cropped)
                    context['analysis_type_performed'] = 'advanced'

                else:
                    logger.error(f"Invalid analysis type received: {analysis_type}")
                    context['error_message'] = "Invalid analysis type specified."
                    context['form'] = form
                    return render(request, 'land_use_app/input_images.html', context) 

                # Update the context with the results returned by the chosen analysis function
                context.update(analysis_results)

            except Exception as e:
                 logger.error(f"An unexpected error occurred in analyze_images view: {e}")
                 traceback.print_exc() # Log traceback
                 context['error_message'] = f"An unexpected server error occurred: {e}. Check server logs."

        context['form'] = form

    else: # GET request
        form = ImageUploadForm()
        context['form'] = form

    return render(request, 'land_use_app/input_images.html', context)