from django.apps import AppConfig
import logging
import torch
from huggingface_hub import hf_hub_download
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Placeholder UNet model - replace with actual FLAIR UNet definition
class UNet(torch.nn.Module):
    def __init__(self, n_channels=4, n_classes=15):
        super().__init__()
        # Placeholder: Replace with the actual FLAIR UNet structure if needed
        self.net = torch.nn.Identity()

    def forward(self, x):
        return self.net(x)

class LandUseAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'land_use_app'

    model = None
    model_load_error = False

    def ready(self):
        import sys
        try:
            from django.core.management import execute_from_command_line
            if 'runserver' not in sys.argv:
                logger.info("Skipping model loading during management command.")
                return
        except Exception as e:
            logger.warning(f"Management command check failed: {e}. Attempting to load model anyway.")

        logger.info("AppConfig ready. Attempting to load FLAIR UNet model...")

        try:
            # Download model weights from Hugging Face Hub
            model_path = hf_hub_download(
                repo_id="IGNF/FLAIR-INC_rgbie_15cl_resnet34-unet",
                filename="FLAIR-INC_rgbie_15cl_resnet34-unet_weights.pth"
            )

            # Initialize the model (placeholder here)
            model = UNet(n_channels=4, n_classes=15)

            # Load the state_dict
            raw_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            # If model is nested inside a dictionary, extract it
            if isinstance(raw_state_dict, dict) and 'model' in raw_state_dict:
                raw_state_dict = raw_state_dict['model']

            # Remove unwanted prefixes from state_dict keys
            cleaned_state_dict = OrderedDict()
            for k, v in raw_state_dict.items():
                new_key = k.replace("model.seg_model.", "")
                cleaned_state_dict[new_key] = v

            model.load_state_dict(cleaned_state_dict, strict=False)
            model.eval()

            # Store in app-level variable
            LandUseAppConfig.model = model
            logger.info("FLAIR UNet model loaded successfully in AppConfig.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load FLAIR UNet model in AppConfig: {e}", exc_info=True)
            LandUseAppConfig.model_load_error = True
            raise e
