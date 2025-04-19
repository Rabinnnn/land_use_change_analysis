from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import base64
import traceback # Import for detailed error logging

# --- Model Loading (Load once when app starts) ---
feature_extractor = None
model = None
try:
    print("Loading Segformer model...")
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512").eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Segformer model: {e}")
    # In a real application, you might want to log this error and show a user-friendly message


# --- Simplified Category Mapping Definitions ---
# Define unique simplified ID values for your target conceptual categories + "Other"
SIMPLIFIED_IDS = {
    "Other": 0,
    "Buildings": 1,
    "Vegetation": 2,
    "Water": 3,
    # Add other conceptual categories if needed (e.g., "Roads": 4)
}

# Mapping from raw ADE20K class IDs to conceptual category NAMES
ADE20K_CATEGORY_NAMES = {
    1: "Buildings", 8: "Buildings",
    5: "Vegetation", 9: "Vegetation", 12: "Vegetation", 13: "Vegetation",
    14: "Water", 18: "Water", 43: "Water",
    # Add other raw ADE20K IDs and their corresponding conceptual names if necessary
    # Example: 4: "Roads", 6: "Roads",
}

# Define colors for visualization based on simplified IDs
SIMPLIFIED_COLORS = {
    SIMPLIFIED_IDS["Other"]: (100, 100, 100), # Grey
    SIMPLIFIED_IDS["Buildings"]: (255, 255, 0), # Yellow
    SIMPLIFIED_IDS["Vegetation"]: (0, 255, 0), # Green
    SIMPLIFIED_IDS["Water"]: (0, 0, 255), # Blue
    # Add colors for other simplified IDs
    # Example: SIMPLIFIED_IDS["Roads"]: (150, 100, 50), # Brown/Tan
}

# --- Helper Functions (Remain the same) ---

def center_crop(img, target_size):
    """Crops the center of a PIL Image to the target size."""
    width, height = img.size
    new_w, new_h = target_size
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    return img.crop((left, top, right, bottom))

def segment_image(image):
    """Performs semantic segmentation using the loaded Segformer model."""
    if feature_extractor is None or model is None:
        print("Model not loaded. Cannot perform segmentation.")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

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

        print(f"[DEBUG] Raw segmentation output shape: {segmentation.shape}")
        return segmentation.astype(np.uint8)

    except Exception as e:
        print(f"Error during segmentation: {e}")
        traceback.print_exc()
        return None

# --- Analysis Logic Functions ---

def _perform_advanced_analysis(image1_cropped, image2_cropped):
    """Performs the semantic segmentation change detection."""
    results = {} # Dictionary to store results for context

    print("[DEBUG] Starting Advanced Analysis (Segmentation Change Detection)...")

    # 1. Get raw segmentation masks
    output1_raw = segment_image(image1_cropped)
    output2_raw = segment_image(image2_cropped)

    if output1_raw is None or output2_raw is None:
        print("[DEBUG] Segmentation failed for Advanced Analysis.")
        return {'error_message': "Segmentation failed for Advanced Analysis."}

    print("[DEBUG] Output1 raw shape:", output1_raw.shape)
    print("[DEBUG] Output2 raw shape:", output2_raw.shape)
    print("[DEBUG] Unique raw labels in image1:", np.unique(output1_raw))
    print("[DEBUG] Unique raw labels in image2:", np.unique(output2_raw))

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
               print(f"[WARNING] Raw ID {raw_id} from ADE20K_CATEGORY_NAMES exceeds map size. Skipping.")
        else:
             print(f"[WARNING] Category name '{category_name}' not in SIMPLIFIED_IDS. Raw ID {raw_id} maps to 'Other'.")

    output1_simplified = simplified_map[output1_raw]
    output2_simplified = simplified_map[output2_raw]

    print("[DEBUG] Unique simplified labels in image1:", np.unique(output1_simplified))
    print("[DEBUG] Unique simplified labels in image2:", np.unique(output2_simplified))

    # 3. Improved Change Detection based on simplified IDs
    changed_mask_simplified = (output1_simplified != output2_simplified)

    # 4. Calculate total change percent
    total_change_percent = round((np.sum(changed_mask_simplified) / changed_mask_simplified.size) * 100, 2)
    results['change_percent'] = total_change_percent
    print(f"[DEBUG] Total Changed Pixels (Simplified): {np.sum(changed_mask_simplified)} | Total Percent: {total_change_percent:.2f}%")

    # 5. Calculate Change Summary by Transition Type
    change_summary = {}
    total_pixels = changed_mask_simplified.size
    all_possible_simplified_ids = np.unique(np.concatenate((np.unique(output1_simplified), np.unique(output2_simplified))))
    simplified_id_to_name = {v: k for k, v in SIMPLIFIED_IDS.items()}

    print("[DEBUG] Calculating change summary...")
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
                    print(f"[DEBUG] Transition {transition_key}: {count} pixels ({change_summary[transition_key]:.2f}%)")

    results['change_summary'] = dict(sorted(change_summary.items(), key=lambda item: item[1], reverse=True))
    print("[DEBUG] Change summary calculated.")

    # 6. Visualization (Coloring changes by final simplified category)
    output_image_base = np.array(image2_cropped)
    overlay_color_image = np.zeros_like(output_image_base)
    changed_mask_3channel = np.stack([changed_mask_simplified] * 3, axis=-1)

    if np.sum(changed_mask_simplified) > 0:
        simplified_ids_in_changed_areas = np.unique(output2_simplified[changed_mask_simplified])
    else:
        simplified_ids_in_changed_areas = []

    print(f"[DEBUG] Simplified IDs in changed areas (Image 2, will be colored): {simplified_ids_in_changed_areas}")

    for simplified_id in simplified_ids_in_changed_areas:
        color = SIMPLIFIED_COLORS.get(simplified_id, (128, 128, 128))
        print(f"[DEBUG] Preparing color {color} for changes ending in simplified ID {simplified_id}")
        mask_for_this_color = changed_mask_simplified & (output2_simplified == simplified_id)
        # Corrected line: use the 2D mask to index the 3D array
        overlay_color_image[mask_for_this_color] = color

    final_output_image = np.where(
        changed_mask_3channel,
        (0.7 * output_image_base + 0.3 * overlay_color_image).astype(np.uint8),
        output_image_base
    )

    # 7. Encode the resulting image
    _, buffer = cv2.imencode('.png', cv2.cvtColor(final_output_image, cv2.COLOR_RGB2BGR))
    results['change_map'] = buffer.tobytes()

    # 8. Prepare legend data for template
    simplified_legend_items = []
    for name, sim_id in SIMPLIFIED_IDS.items():
        color_rgb = SIMPLIFIED_COLORS.get(sim_id, (128, 128, 128))
        color_hex = '#%02x%02x%02x' % color_rgb
        simplified_legend_items.append({'name': name, 'color': color_hex})
    simplified_legend_items.sort(key=lambda item: item['name'])
    results['simplified_legend_items'] = simplified_legend_items

    print("[DEBUG] Advanced Analysis complete.")
    return results # Return the dictionary of results

def _perform_basic_analysis(image1_cropped, image2_cropped):
    
    print("[DEBUG] Starting Basic Analysis...")

    
    try:
        
        # Calculate average pixel value difference (dummy calculation)
        img1_gray = np.array(image1_cropped.convert('L'))
        img2_gray = np.array(image2_cropped.convert('L'))
        avg_diff = np.mean(np.abs(img1_gray - img2_gray))

        results = {
            'basic_message': 'Basic Analysis Results:',
            'average_pixel_difference': round(avg_diff, 2),
            
        }
        print("[DEBUG] Basic Analysis complete (using dummy calculation).")
        return results

    except Exception as e:
        print(f"Error during Basic Analysis: {e}")
        traceback.print_exc()
        return {'error_message': f"Error during Basic Analysis: {e}"}

# --- Main Django View Function ---

def analyze_images(request):
    context = {} # Initialize context dictionary

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        # Determine which analysis button was clicked
        analysis_type = request.POST.get('analysis_type')
        print(f"[DEBUG] Received POST request. Analysis type: {analysis_type}")

        if form.is_valid():
            try:
                # Load and crop images (This step is common for both analyses)
                image1 = Image.open(request.FILES['image1']).convert('RGB')
                image2 = Image.open(request.FILES['image2']).convert('RGB')

                common_width = min(image1.width, image2.width)
                common_height = min(image1.height, image2.height)

                image1_cropped = center_crop(image1, (common_width, common_height))
                image2_cropped = center_crop(image2, (common_width, common_height))

                print("[DEBUG] Images loaded and cropped for analysis.")

                # --- Call the appropriate analysis function based on button clicked ---
                analysis_results = {}
                if analysis_type == 'basic':
                    analysis_results = _perform_basic_analysis(image1_cropped, image2_cropped)
                    context['analysis_type_performed'] = 'basic'
                elif analysis_type == 'advanced':
                     # Pass cropped images to the advanced analysis function
                    analysis_results = _perform_advanced_analysis(image1_cropped, image2_cropped)
                    context['analysis_type_performed'] = 'advanced'
                else:
                    # Handle case where no or an unknown analysis type was sent
                    context['error_message'] = "Invalid analysis type specified."
                    print(f"[ERROR] Invalid analysis type received: {analysis_type}")
                    # Keep form for resubmission
                    context['form'] = form
                    return render(request, 'land_use_app/input_images.html', context) # Use your app name


                # Update the context with the results from the chosen analysis
                context.update(analysis_results)

            except Exception as e:
                 # Catch any unexpected errors during file processing or analysis function calls
                 print(f"[ERROR] An unexpected error occurred in analyze_images view: {e}")
                 traceback.print_exc()
                 context['error_message'] = f"An unexpected server error occurred: {e}. Check server logs."

        # If form is not valid, context['form'] already contains the form with errors
        context['form'] = form

    else: # GET request (initial page load)
        form = ImageUploadForm()
        context['form'] = form
        # No analysis performed on GET, results section will not display

    # Render the template with the form and results/errors (if any)
    return render(request, 'land_use_app/input_images.html', context) 