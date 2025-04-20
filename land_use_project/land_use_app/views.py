from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import base64
import traceback # Import for detailed error logging
import io 


# --- Model Loading (Load once when app starts) ---
feature_extractor = None
model = None
try:
    print("Loading Segformer model for Advanced Analysis...")
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512").eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Segformer model: {e}")


# --- Advanced Analysis (Segmentation) Category Mapping Definitions ---
# These are only used by the _perform_segmentation_analysis function
SIMPLIFIED_IDS = {
    "Other": 0,
    "Buildings": 1,
    "Vegetation": 2,
    "Water": 3,
    # Add other conceptual categories if needed (e.g., "Roads": 4)
}

ADE20K_CATEGORY_NAMES = {
    1: "Buildings", 8: "Buildings",
    5: "Vegetation", 9: "Vegetation", 12: "Vegetation", 13: "Vegetation",
    14: "Water", 18: "Water", 43: "Water",
    # Add other raw ADE20K IDs and their corresponding conceptual names if necessary
    # Example: 4: "Roads", 6: "Roads",
}

SIMPLIFIED_COLORS = {
    SIMPLIFIED_IDS["Other"]: (100, 100, 100), # Grey
    SIMPLIFIED_IDS["Buildings"]: (255, 255, 0), # Yellow
    SIMPLIFIED_IDS["Vegetation"]: (0, 255, 0), # Green
    SIMPLIFIED_IDS["Water"]: (0, 0, 255), # Blue
    # Add colors for other simplified IDs
    # Example: SIMPLIFIED_IDS["Roads"]: (150, 100, 50), # Brown/Tan
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

def _perform_basic_analysis(image1_cropped, image2_cropped):
    """
    Performs the basic pixel difference analysis using OpenCV.
    Based on the user's provided code.
    """
    results = {} # Dictionary to store results for context

    print("[DEBUG] Starting Basic Analysis (Pixel Difference)...")

    try:
        # Convert PIL Images (cropped) to OpenCV BGR format
        # cv2 expects BGR order, PIL is RGB
        img1 = cv2.cvtColor(np.array(image1_cropped), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2_cropped), cv2.COLOR_RGB2BGR)

        # Note: Images are already cropped to the same size in the main view,
        # so explicit resizing here is not needed and can be removed.


        # Difference analysis (Absolute difference between the two images)
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Convert difference to grayscale
        # Threshold the grayscale difference to find pixels that changed "enough"
        # Pixels with a difference greater than 50 become white (255), others black (0)
        # The threshold value (50) is arbitrary and might need tuning
        _, thresh_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

        # Find contours around changed regions (optional visualization)
        # RETR_EXTERNAL retrieves only the outer contours
        # CHAIN_APPROX_SIMPLE compresses horizontal/vertical/diagonal segments
        contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare an image to draw contours on (e.g., a copy of the second image)
        img2_with_contours = img2.copy()
        # Draw red rectangles around the bounding box of each contour
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt) # Get bounding box (top-left corner and size)
            cv2.rectangle(img2_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draw rectangle in Red (BGR)


        # Highlight changes with a semi-transparent red overlay
        # Create a red mask where changed pixels are solid red
        red_mask = np.zeros_like(img2) # Start with a black image
        # Set the red channel (index 2 in BGR) to 255 wherever thresh_diff is > 0 (i.e., 255)
        red_mask[:, :, 2] = thresh_diff

        # Blend the image with contours (70%) with the red mask (30%)
        # This creates a semi-transparent red overlay on the changed areas
        highlighted_image = cv2.addWeighted(img2_with_contours, 0.7, red_mask, 0.3, 0)

        # Calculate percentage change based on the thresholded difference mask
        changed_pixels = np.sum(thresh_diff > 0) # Count pixels that are white (changed)
        total_pixels = thresh_diff.size # Total number of pixels in the image
        percent_change = (changed_pixels / total_pixels) * 100

        # Store results in the dictionary using keys specific to basic analysis
        results['basic_message'] = "Basic pixel difference analysis results:"
        results['basic_change_percent'] = round(percent_change, 2)

        # Convert the final highlighted image to PNG bytes for display in the template
        # cv2.imencode expects BGR format
        _, buffer = cv2.imencode('.png', highlighted_image)
        results['basic_change_map'] = buffer.tobytes()

        # Note: Basic analysis doesn't have category-specific results or legends like advanced
        # You could add dummy legend info if you want a consistent legend section structure
        # For example:
        # results['basic_legend_items'] = [{'name': 'Change Detected', 'color': '#ff0000'}] # Red color hex

        print("[DEBUG] Basic Analysis complete.")
        return results # Return the dictionary of results

    except Exception as e:
        print(f"[ERROR] Error during Basic Analysis: {e}")
        traceback.print_exc() # Print traceback for debugging on the server side
        return {'error_message': f"An error occurred during Basic Analysis: {e}"}


def _perform_segmentation_analysis(image1_cropped, image2_cropped):
    """
    Performs the advanced segmentation change detection using Segformer.
    This was the 'advanced' analysis placeholder previously, now the actual segmentation.
    """
    results = {} # Dictionary to store results for context

    print("[DEBUG] Starting Advanced Analysis (Segmentation Change Detection)...")

    # 1. Get raw segmentation masks
    output1_raw = segment_image(image1_cropped)
    output2_raw = segment_image(image2_cropped)

    if output1_raw is None or output2_raw is None:
        print("[DEBUG] Segmentation failed for Advanced Analysis.")
        return {'error_message': "Segmentation failed for Advanced Analysis. Check model loading."}

    print("[DEBUG] Output1 raw shape:", output1_raw.shape)
    print("[DEBUG] Output2 raw shape:", output2_raw.shape)
    print("[DEBUG] Unique raw labels in image1:", np.unique(output1_raw))
    print("[DEBUG] Unique raw labels in image2:", np.unique(output2_raw))

    # 2. Apply the improved mapping from raw IDs to single simplified IDs
    max_raw_id = 0
    if ADE20K_CATEGORY_NAMES:
        # Ensure max_raw_id is at least the max ID in the raw outputs or the max ID in the mapping
        max_raw_id = max(output1_raw.max() if output1_raw is not None else 0,
                         output2_raw.max() if output2_raw is not None else 0,
                         max(ADE20K_CATEGORY_NAMES.keys()) if ADE20K_CATEGORY_NAMES else 0)
    else:
         max_raw_id = max(output1_raw.max() if output1_raw is not None else 0,
                         output2_raw.max() if output2_raw is not None else 0)

    simplified_map = np.full(max_raw_id + 1, SIMPLIFIED_IDS["Other"], dtype=np.uint8)

    for raw_id, category_name in ADE20K_CATEGORY_NAMES.items():
        if category_name in SIMPLIFIED_IDS:
            simplified_id = SIMPLIFIED_IDS[category_name]
            if raw_id <= max_raw_id:
               simplified_map[raw_id] = simplified_id
            else:
               print(f"[WARNING] Raw ID {raw_id} from ADE20K_CATEGORY_NAMES exceeds map size {max_raw_id}. Skipping map entry.")
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
    results['segmentation_change_percent'] = total_change_percent # Use distinct key
    print(f"[DEBUG] Total Changed Pixels (Segmentation Simplified): {np.sum(changed_mask_simplified)} | Total Percent: {total_change_percent:.2f}%")

    # 5. Calculate Change Summary by Transition Type (Segmentation specific)
    change_summary = {}
    total_pixels = changed_mask_simplified.size
    all_possible_simplified_ids = np.unique(np.concatenate((np.unique(output1_simplified), np.unique(output2_simplified))))
    simplified_id_to_name = {v: k for k, v in SIMPLIFIED_IDS.items()}

    print("[DEBUG] Calculating segmentation change summary...")
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

    results['segmentation_change_summary'] = dict(sorted(change_summary.items(), key=lambda item: item[1], reverse=True)) # Use distinct key
    print("[DEBUG] Segmentation Change summary calculated.")

    # 6. Visualization (Coloring changes by final simplified category - Segmentation specific)
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

    # 7. Encode the resulting image (Segmentation specific map)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(final_output_image, cv2.COLOR_RGB2BGR))
    results['segmentation_change_map'] = buffer.tobytes() # Use distinct key

    # 8. Prepare legend data for template (Segmentation specific legend)
    simplified_legend_items = []
    for name, sim_id in SIMPLIFIED_IDS.items():
        color_rgb = SIMPLIFIED_COLORS.get(sim_id, (128, 128, 128))
        color_hex = '#%02x%02x%02x' % color_rgb
        simplified_legend_items.append({'name': name, 'color': color_hex})
    simplified_legend_items.sort(key=lambda item: item['name'])
    results['segmentation_legend_items'] = simplified_legend_items # Use distinct key

    print("[DEBUG] Advanced Analysis (Segmentation) complete.")
    return results # Return the dictionary of results

# --- Main Django View Function ---

def analyze_images(request):
    context = {} # Initialize context dictionary

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        # Determine which analysis button was clicked using the button's 'name' and 'value'
        analysis_type = request.POST.get('analysis_type')
        print(f"[DEBUG] Received POST request. Analysis type: {analysis_type}")

        if form.is_valid():
            try:
                # Load and crop images (This step is common for both analyses)
                # Cropping ensures both images are the same size for comparison/analysis
                image1 = Image.open(request.FILES['image1']).convert('RGB')
                image2 = Image.open(request.FILES['image2']).convert('RGB')

                common_width = min(image1.width, image2.width)
                common_height = min(image1.height, image2.height)

                image1_cropped = center_crop(image1, (common_width, common_height))
                image2_cropped = center_crop(image2, (common_width, common_height))

                print(f"[DEBUG] Images loaded and cropped to {common_width}x{common_height} for analysis.")

                # --- Call the appropriate analysis function based on button clicked ---
                analysis_results = {}
                if analysis_type == 'basic':
                    # Call the function for the pixel difference analysis
                    analysis_results = _perform_basic_analysis(image1_cropped, image2_cropped)
                    # Add a marker to context to indicate which analysis was performed
                    context['analysis_type_performed'] = 'basic'

                elif analysis_type == 'advanced':
                    # Call the function for the segmentation analysis
                    analysis_results = _perform_segmentation_analysis(image1_cropped, image2_cropped)
                     # Add a marker to context to indicate which analysis was performed
                    context['analysis_type_performed'] = 'advanced'

                else:
                    # Handle case where no or an unknown analysis type was sent (shouldn't happen with correct buttons)
                    context['error_message'] = "Invalid analysis type specified."
                    print(f"[ERROR] Invalid analysis type received: {analysis_type}")
                    # Keep form for resubmission if necessary
                    context['form'] = form
                    # Render the template immediately with the error message
                    return render(request, 'land_use_app/input_images.html', context) 


                # Update the context dictionary with the results returned by the chosen analysis function
                # This merges the key-value pairs from analysis_results into context
                context.update(analysis_results)

            except Exception as e:
                 # Catch any unexpected errors during file loading or cropping BEFORE analysis functions are called
                 print(f"[ERROR] An unexpected error occurred during file handling or initial processing: {e}")
                 traceback.print_exc() # Print traceback for debugging on the server side
                 context['error_message'] = f"An unexpected server error occurred: {e}. Check server logs."

        # If form is not valid, context['form'] already contains the form with errors
        # The template will display these errors next to the fields
        context['form'] = form

    else: # GET request (initial page load)
        # Create an empty form instance for the initial page load
        form = ImageUploadForm()
        context['form'] = form
        # No analysis results are available on a GET request, results section will not display

    # Render the template, passing the request, the template path, and the context dictionary
    return render(request, 'land_use_app/input_images.html', context)