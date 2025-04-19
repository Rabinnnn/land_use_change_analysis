from django.shortcuts import render
# Assuming your forms.py is in the same app directory
from .forms import ImageUploadForm
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import base64 # Needed for potential future direct base64 encoding if not using template tag

# Load Hugging Face Segformer model and feature extractor
# This should ideally be done once when the application starts, not on every request.
# In a production Django app, consider using a global variable, a cache, or a specific loading mechanism.
# For simplicity in this example, we load it here.
try:
    print("Loading Segformer model...")
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512").eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Segformer model: {e}")
    # Handle model loading errors appropriately in a real application
    feature_extractor = None
    model = None


# Mapping from raw ADE20K class IDs to conceptual category NAMES
# Expanded based on common ADE20K IDs and your observed labels
ADE20K_CATEGORY_NAMES = {
    1: "Buildings",    # Common ADE20K ID for 'building'
    8: "Buildings",    # Original ID you had (might be a specific type)
    9: "Vegetation", # Common ADE20K ID for 'vegetation'
    12: "Vegetation", # Original ID you had (might be a specific type)
    13: "Vegetation", # Common ADE20K ID for 'grass'
    14: "Water",      # Common ADE20K ID for 'river'
    18: "Water",      # Common ADE20K ID for 'water'
    43: "Water",       # Original ID you had
    # Add other raw ADE20K IDs and their corresponding conceptual names if needed
    # Example: 4: "Roads", 6: "Roads", # If you wanted a "Roads" category
}

# Define unique simplified ID values for your target conceptual categories + "Other"
# These will be the values in the final simplified masks.
SIMPLIFIED_IDS = {
    "Other": 0, # Use 0 for any category not in ADE20K_CATEGORY_NAMES
    "Buildings": 1,
    "Vegetation": 2,
    "Water": 3,
    # Add simplified IDs for other conceptual categories if you added them above
    # Example: "Roads": 4,
}

# (Optional) Define colors for visualization based on simplified IDs if you implement
# color-coding the final map by simplified category or transition type.
# Currently, the visualization just uses magenta for *any* change.
SIMPLIFIED_COLORS = {
    SIMPLIFIED_IDS["Other"]: (100, 100, 100), # Grey for Other
    SIMPLIFIED_IDS["Buildings"]: (255, 255, 0), # Yellow for Buildings
    SIMPLIFIED_IDS["Vegetation"]: (0, 255, 0), # Green for Vegetation
    SIMPLIFIED_IDS["Water"]: (0, 0, 255), # Blue for Water
    # Add colors for other simplified IDs
    # Example: SIMPLIFIED_IDS["Roads"]: (150, 100, 50), # Brown/Tan for Roads
}


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
        return None # Or raise an exception

    try:
        # Ensure model is on the correct device (CPU/GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare image inputs
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the logits and derive the segmentation mask
        logits = outputs.logits
        # Resize logits to original image size before argmax for potentially better results
        # Hugging Face documentation often resizes logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )
        segmentation = logits.argmax(dim=1)[0].cpu().numpy()

        # Note: The original code resized the *final mask*. Resizing logits first might be better.
        # If you prefer resizing the final mask as before:
        # segmentation_raw = logits.argmax(dim=1)[0].cpu().numpy()
        # segmentation = cv2.resize(segmentation_raw.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
        # print(f"[DEBUG] Segmentation resized shape: {segmentation.shape}")

        print(f"[DEBUG] Raw segmentation output shape: {segmentation.shape}")
        return segmentation.astype(np.uint8) # Ensure it's uint8

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None


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

                if output1_raw is None or output2_raw is None:
                     context['error_message'] = "Error performing image segmentation. Please try again."
                     context['form'] = form # Return form with errors
                     return render(request, 'land_use_app/input_images.html', context)


                print("[DEBUG] Output1 raw shape:", output1_raw.shape)
                print("[DEBUG] Output2 raw shape:", output2_raw.shape)
                print("[DEBUG] Unique raw labels in image1:", np.unique(output1_raw))
                print("[DEBUG] Unique raw labels in image2:", np.unique(output2_raw))

                # --- Apply the improved mapping from raw IDs to single simplified IDs ---

                # Create the mapping array: Default to the simplified "Other" ID (0)
                max_raw_id = 0
                if ADE20K_CATEGORY_NAMES:
                    max_raw_id = max(output1_raw.max(), output2_raw.max(), max(ADE20K_CATEGORY_NAMES.keys()))
                else:
                     max_raw_id = max(output1_raw.max(), output2_raw.max())

                # Initialize map with the default "Other" simplified ID
                simplified_map = np.full(max_raw_id + 1, SIMPLIFIED_IDS["Other"], dtype=output1_raw.dtype)

                # Populate the map: For each raw ADE20K ID in your mapping, find its conceptual name,
                # and then find the corresponding unique simplified ID.
                for raw_id, category_name in ADE20K_CATEGORY_NAMES.items():
                    # Ensure the category name exists in our simplified IDs mapping
                    if category_name in SIMPLIFIED_IDS:
                        simplified_id = SIMPLIFIED_IDS[category_name]
                        if raw_id <= max_raw_id: # Prevent indexing errors if max_raw_id is smaller
                           simplified_map[raw_id] = simplified_id
                        else:
                           print(f"[WARNING] Raw ID {raw_id} from ADE20K_CATEGORY_NAMES exceeds max_raw_id {max_raw_id}. Skipping map entry.")
                    else:
                         print(f"[WARNING] Category name '{category_name}' found in ADE20K_CATEGORY_NAMES but not in SIMPLIFIED_IDS. Raw ID {raw_id} will map to 'Other'.")


                # Apply the new mapping
                output1_simplified = simplified_map[output1_raw]
                output2_simplified = simplified_map[output2_raw]

                print("[DEBUG] Unique simplified labels in image1:", np.unique(output1_simplified))
                print("[DEBUG] Unique simplified labels in image2:", np.unique(output2_simplified))


                # --- Improved Change Detection based on simplified IDs ---

                # Find changed pixels based on the NEW single simplified ID values
                changed_mask_simplified = (output1_simplified != output2_simplified)

                # Calculate total change percent based on simplified mask
                total_change_percent = round((np.sum(changed_mask_simplified) / changed_mask_simplified.size) * 100, 2)
                context['change_percent'] = total_change_percent
                print(f"[DEBUG] Total Changed Pixels (Simplified): {np.sum(changed_mask_simplified)} | Total Percent: {total_change_percent:.2f}%")

                # --- Optional: Calculate Change Summary by Type ---
                # This requires iterating through transitions. Let's add a basic example.
                change_summary = {}
                total_pixels = changed_mask_simplified.size

                # Get the list of all possible simplified IDs present in *either* image
                all_simplified_ids = np.unique(np.concatenate((output1_simplified, output2_simplified)))

                # Reverse map simplified IDs back to names for the summary
                simplified_id_to_name = {v: k for k, v in SIMPLIFIED_IDS.items()}

                print("[DEBUG] Calculating change summary...")
                # Iterate through all possible pairs of simplified categories
                for id1 in all_simplified_ids:
                    for id2 in all_simplified_ids:
                        if id1 != id2: # Only count actual changes
                            # Create a mask for this specific transition type
                            transition_mask = (output1_simplified == id1) & (output2_simplified == id2)
                            count = np.sum(transition_mask)

                            if count > 0:
                                # Get the category names for the summary
                                name1 = simplified_id_to_name.get(id1, f"Unknown_{id1}")
                                name2 = simplified_id_to_name.get(id2, f"Unknown_{id2}")
                                transition_key = f"{name1} -> {name2}"
                                change_summary[transition_key] = round((count / total_pixels) * 100, 2)
                                print(f"[DEBUG] Transition {transition_key}: {count} pixels ({change_summary[transition_key]:.2f}%)")

                # Sort the change summary by percentage descending
                context['change_summary'] = dict(sorted(change_summary.items(), key=lambda item: item[1], reverse=True))
                print("[DEBUG] Change summary calculated.")


                # --- Visualization (Using the improved mask) ---

                # Use image2_cropped as base for visualization
                output_image = np.array(image2_cropped)

                # Color to highlight changed areas (based on simplified change)
                highlight_color = (255, 0, 255)  # Magenta

                # Create a color mask
                color_mask = np.zeros_like(output_image)
                color_mask[:, :] = highlight_color

                # Blend where pixels changed (using the simplified mask)
                # Ensure the mask has a dimension for color channels for broadcasting
                changed_mask_3channel = np.stack([changed_mask_simplified] * 3, axis=-1)

                # Perform blending only on the changed pixels
                blended = (0.7 * output_image + 0.3 * color_mask).astype(np.uint8)
                output_image = np.where(changed_mask_3channel, blended, output_image)


                # Encode the resulting image for display in the template
                # Using cv2.imencode and base64 encoding
                _, buffer = cv2.imencode('.png', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)) # Convert back to BGR for cv2
                # context['change_map'] = base64.b64encode(buffer).decode('utf-8') # Use this if not using the template tag filter
                context['change_map'] = buffer.tobytes() # Keep bytes if using template tag filter

            except Exception as e:
                 print(f"An error occurred during analysis: {e}")
                 import traceback
                 traceback.print_exc()
                 context['error_message'] = f"An error occurred during analysis: {e}"

        # Form is invalid, errors will be in context['form'] handled in template
        context['form'] = form

    else: # GET request
        form = ImageUploadForm()
        context['form'] = form

    # Render the template with the form and results (if any)
    return render(request, 'land_use_app/input_images.html', context)