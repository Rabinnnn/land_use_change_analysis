from django.shortcuts import render
# Assuming your forms.py is in the same app directory
from .forms import ImageUploadForm
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import base64 # Useful if you don't use a custom template tag for b64encode

# --- Model Loading (Load once when app starts) ---
# This should ideally be done once when the application starts, not on every request.
# In a production Django app, consider using a global variable, a cache, or a specific loading mechanism.
# For simplicity in this example, we load it here.
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
# These will be the values in the final simplified masks.
SIMPLIFIED_IDS = {
    "Other": 0, # Use 0 for any category not explicitly mapped below
    "Buildings": 1,
    "Vegetation": 2,
    "Water": 3,
    # Add other conceptual categories if needed (e.g., "Roads": 4)
}

# Mapping from raw ADE20K class IDs to conceptual category NAMES
# This should include all raw IDs you want to group under a simplified category.
ADE20K_CATEGORY_NAMES = {
    1: "Buildings",    # Common ADE20K ID for 'building'
    8: "Buildings",    # Original ID you had (might be a specific type)
    # Find and add other ADE20K building-related IDs if necessary (e.g., 20: "building part")

    5: "Vegetation", # Common ADE20K ID for 'tree'
    9: "Vegetation", # Common ADE20K ID for 'vegetation'
    12: "Vegetation", # Original ID you had (might be a specific type)
    13: "Vegetation", # Common ADE20K ID for 'grass'
    # Find and add other vegetation-related IDs if necessary

    14: "Water",      # Common ADE20K ID for 'river'
    18: "Water",      # Common ADE20K ID for 'water'
    43: "Water",       # Original ID you had
    # Find and add other water-related IDs if necessary

    # Example of adding another category:
    # 4: "Roads",      # Common ADE20K ID for 'road'
    # 6: "Roads",      # Common ADE20K ID for 'sidewalk'
}

# Define colors for visualization based on simplified IDs
# These colors will be used to highlight changed areas on the map based on the FINAL category.
SIMPLIFIED_COLORS = {
    SIMPLIFIED_IDS["Other"]: (100, 100, 100), # Grey for Other
    SIMPLIFIED_IDS["Buildings"]: (255, 255, 0), # Yellow for Buildings
    SIMPLIFIED_IDS["Vegetation"]: (0, 255, 0), # Green for Vegetation
    SIMPLIFIED_IDS["Water"]: (0, 0, 255), # Blue for Water
    # Add colors for other simplified IDs
    # Example: SIMPLIFIED_IDS["Roads"]: (150, 100, 50), # Brown/Tan for Roads
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
        # In a real application, you might want to return None or raise a specific error
        return None

    try:
        # Ensure model is on the correct device (CPU/GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare image inputs
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the logits and resize them to original image size before argmax
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )
        # Get the class ID with the highest score for each pixel
        segmentation = logits.argmax(dim=1)[0].cpu().numpy()

        print(f"[DEBUG] Raw segmentation output shape: {segmentation.shape}")
        return segmentation.astype(np.uint8) # Ensure it's uint8 for mapping

    except Exception as e:
        print(f"Error during segmentation: {e}")
        # In a real application, log the full traceback
        return None


# --- Django View Function ---

def analyze_images(request):
    context = {}

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            try:
                image1 = Image.open(request.FILES['image1']).convert('RGB')
                image2 = Image.open(request.FILES['image2']).convert('RGB')

                print("[DEBUG] Original image1 size:", image1.size)
                print("[DEBUG] Original image2 size:", image2.size)

                # Ensure both images have compatible sizes for comparison/segmentation
                common_width = min(image1.width, image2.width)
                common_height = min(image1.height, image2.height)

                image1_cropped = center_crop(image1, (common_width, common_height))
                image2_cropped = center_crop(image2, (common_width, common_height))

                print("[DEBUG] Cropped image size:", image1_cropped.size)

                # Get raw segmentation masks (ADE20K IDs)
                output1_raw = segment_image(image1_cropped)
                output2_raw = segment_image(image2_cropped)

                # Check if segmentation failed
                if output1_raw is None or output2_raw is None:
                     context['error_message'] = "Error performing image segmentation. Check model loading logs."
                     context['form'] = form # Return form with potential errors
                     return render(request, 'land_use_app/input_images.html', context) # Use your app name


                print("[DEBUG] Output1 raw shape:", output1_raw.shape)
                print("[DEBUG] Output2 raw shape:", output2_raw.shape)
                print("[DEBUG] Unique raw labels in image1:", np.unique(output1_raw))
                print("[DEBUG] Unique raw labels in image2:", np.unique(output2_raw))

                # --- Apply the improved mapping from raw IDs to single simplified IDs ---

                # Determine the maximum raw ID found or needed for the mapping array size
                max_raw_id = 0
                if ADE20K_CATEGORY_NAMES:
                    max_raw_id = max(output1_raw.max(), output2_raw.max(), max(ADE20K_CATEGORY_NAMES.keys()))
                else:
                     max_raw_id = max(output1_raw.max(), output2_raw.max())

                # Initialize map array with the default "Other" simplified ID (e.g., 0)
                simplified_map = np.full(max_raw_id + 1, SIMPLIFIED_IDS["Other"], dtype=np.uint8) # Ensure dtype is uint8

                # Populate the map: For each raw ADE20K ID in your mapping, find its conceptual name,
                # and then find the corresponding unique simplified ID.
                for raw_id, category_name in ADE20K_CATEGORY_NAMES.items():
                    # Ensure the category name exists in our simplified IDs mapping
                    if category_name in SIMPLIFIED_IDS:
                        simplified_id = SIMPLIFIED_IDS[category_name]
                        # Ensure the raw_id is within the bounds of our map array
                        if raw_id <= max_raw_id:
                           simplified_map[raw_id] = simplified_id
                        else:
                           print(f"[WARNING] Raw ID {raw_id} from ADE20K_CATEGORY_NAMES exceeds max_raw_id {max_raw_id}. Skipping map entry.")
                    else:
                         print(f"[WARNING] Category name '{category_name}' found in ADE20K_CATEGORY_NAMES but not in SIMPLIFIED_IDS. Raw ID {raw_id} will map to 'Other'.")


                # Apply the new mapping to the raw segmentation outputs
                output1_simplified = simplified_map[output1_raw]
                output2_simplified = simplified_map[output2_raw]

                print("[DEBUG] Unique simplified labels in image1:", np.unique(output1_simplified))
                print("[DEBUG] Unique simplified labels in image2:", np.unique(output2_simplified))


                # --- Improved Change Detection based on simplified IDs ---

                # Find changed pixels based on the NEW single simplified ID values
                changed_mask_simplified = (output1_simplified != output2_simplified)

                # Calculate total change percent based on the simplified mask
                total_change_percent = round((np.sum(changed_mask_simplified) / changed_mask_simplified.size) * 100, 2)
                context['change_percent'] = total_change_percent
                print(f"[DEBUG] Total Changed Pixels (Simplified): {np.sum(changed_mask_simplified)} | Total Percent: {total_change_percent:.2f}%")

                # --- Calculate Change Summary by Transition Type ---
                # This counts how many pixels changed from one simplified category to another.
                change_summary = {}
                total_pixels = changed_mask_simplified.size

                # Get the list of all possible simplified IDs present in *either* image for iterating
                all_possible_simplified_ids = np.unique(np.concatenate((np.unique(output1_simplified), np.unique(output2_simplified))))

                # Reverse map simplified IDs back to names for the summary strings
                simplified_id_to_name = {v: k for k, v in SIMPLIFIED_IDS.items()}

                print("[DEBUG] Calculating change summary...")
                # Iterate through all possible pairs of simplified categories found in the images
                for id1 in all_possible_simplified_ids:
                    for id2 in all_possible_simplified_ids:
                        # Only count actual changes between different simplified categories
                        if id1 != id2:
                            # Create a mask for this specific transition type (id1 -> id2)
                            transition_mask = (output1_simplified == id1) & (output2_simplified == id2)
                            count = np.sum(transition_mask)

                            # If this type of transition occurred, add it to the summary
                            if count > 0:
                                # Get the category names for the summary string
                                name1 = simplified_id_to_name.get(id1, f"Unknown_{id1}") # Use get for safety
                                name2 = simplified_id_to_name.get(id2, f"Unknown_{id2}") # Use get for safety
                                transition_key = f"{name1} -> {name2}"
                                change_summary[transition_key] = round((count / total_pixels) * 100, 2)
                                print(f"[DEBUG] Transition {transition_key}: {count} pixels ({change_summary[transition_key]:.2f}%)")

                # Sort the change summary by percentage descending for display
                context['change_summary'] = dict(sorted(change_summary.items(), key=lambda item: item[1], reverse=True))
                print("[DEBUG] Change summary calculated.")


                # --- Visualization (Coloring changes by final simplified category) ---

                # Use image2_cropped converted to numpy array as the base image for visualization
                output_image_base = np.array(image2_cropped)
                # Create a blank canvas (all zeros) of the same size and shape for our colored overlay
                overlay_color_image = np.zeros_like(output_image_base)

                # Ensure the overall changed_mask_simplified is 3-channel for broadcasting with color
                changed_mask_3channel = np.stack([changed_mask_simplified] * 3, axis=-1)

                # Get the simplified IDs that are present in the *changed* areas of image 2.
                # This is more efficient than iterating through *all* possible SIMPLIFIED_IDS.
                simplified_ids_in_changed_areas = np.unique(output2_simplified[changed_mask_simplified])

                print(f"[DEBUG] Simplified IDs in changed areas (Image 2, will be colored): {simplified_ids_in_changed_areas}")

                # Iterate only through the simplified IDs that actually appear in the changed areas of image 2
                for simplified_id in simplified_ids_in_changed_areas:
                    # Get the color for this simplified ID from SIMPLIFIED_COLORS.
                    # Use a default grey color if an ID that shouldn't be there appears, or if a color is missing.
                    color = SIMPLIFIED_COLORS.get(simplified_id, (128, 128, 128))
                    print(f"[DEBUG] Preparing color {color} for changes ending in simplified ID {simplified_id}")

                    # Create a mask for pixels that *changed* AND ended up as this specific simplified_id in image 2
                    mask_for_this_color = changed_mask_simplified & (output2_simplified == simplified_id)

                    # Apply this color to the overlay image for the pixels specified by mask_for_this_color
                    # Need to ensure the specific mask has a dimension for color channels for broadcasting
                    # mask_for_this_color_3channel = np.stack([mask_for_this_color] * 3, axis=-1)
                    overlay_color_image[mask_for_this_color] = color



                # Now blend the base image with the multi-colored overlay, only where overall change occurred
                # We blend the *original* pixel with the *specific color* from the overlay_color_image
                # Adjust the blending factors (0.7 for original, 0.3 for overlay color) as desired.
                # result = (0.7 * output_image_base + 0.3 * overlay_color_image).astype(np.uint8) # This blends everywhere
                # We only want to blend where the overall changed_mask_simplified is True

                # Create the final image: keep original where no change, blend where change
                final_output_image = np.where(
                    changed_mask_3channel, # Condition: where did simplified categories change? (3-channel mask)
                    # Result if True: Blend the original pixel with the color from overlay_color_image
                    (0.7 * output_image_base + 0.3 * overlay_color_image).astype(np.uint8),
                    # Result if False: Keep the original pixel from the base image
                    output_image_base
                )


                # Encode the resulting image for display in the template
                # Using cv2.imencode and base64 encoding
                # cv2 expects BGR order, so convert from RGB numpy array
                _, buffer = cv2.imencode('.png', cv2.cvtColor(final_output_image, cv2.COLOR_RGB2BGR))

                # Depending on how your template tag works:
                # If your template tag 'b64encode' expects bytes:
                context['change_map'] = buffer.tobytes()
                # If your template tag expects a base64 string:
                # context['change_map'] = base64.b64encode(buffer).decode('utf-8')


                # Pass simplified colors and names to the template for the legend
                # Create a list of dictionaries: [{'name': 'Category Name', 'color': '#HEXCOLOR'}]
                simplified_legend_items = []
                # Iterate through the SIMPLIFIED_IDS dictionary to get names and corresponding IDs
                for name, sim_id in SIMPLIFIED_IDS.items():
                    # Get the color tuple (R, G, B) for this simplified ID
                    color_rgb = SIMPLIFIED_COLORS.get(sim_id, (128, 128, 128)) # Default to grey if ID not in SIMPLIFIED_COLORS
                    # Convert the RGB tuple to a hex string for CSS background-color
                    color_hex = '#%02x%02x%02x' % color_rgb
                    simplified_legend_items.append({'name': name, 'color': color_hex})

                # Sort legend items alphabetically by name for consistent display (optional)
                simplified_legend_items.sort(key=lambda item: item['name'])

                context['simplified_legend_items'] = simplified_legend_items


            except Exception as e:
                 # Catch any unexpected errors during file processing or analysis
                 print(f"An unexpected error occurred during analysis: {e}")
                 import traceback
                 traceback.print_exc() # Print traceback for debugging on the server side
                 context['error_message'] = f"An error occurred during analysis: {e}. Check server logs for details."

        # If form is not valid, context['form'] already contains the form with errors
        context['form'] = form

    else: # GET request (initial page load)
        form = ImageUploadForm()
        context['form'] = form
        # No results to display on GET request

    # Render the template with the form and results/errors (if any)
    return render(request, 'land_use_app/input_images.html', context) # Use your app name