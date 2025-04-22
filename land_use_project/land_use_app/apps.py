from django.apps import AppConfig
import logging
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from collections import OrderedDict
from PIL import Image

logger = logging.getLogger(__name__)

# Placeholder UNet model - replace with actual FLAIR UNet definition
class UNet(torch.nn.Module):
    def __init__(self, n_channels=4, n_classes=15):
        super().__init__()
        self.net = torch.nn.Identity()  # Replace with actual architecture

    def forward(self, x):
        return self.net(x)

def default_feature_extractor(*args, **kwargs):
    image = kwargs.get("image")
    if image is None:
        raise ValueError("Image not provided")
    # Continue with the feature extraction logic
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    tensor = transform(image)
    if kwargs.get("return_tensors") == "pt":
        return {"pixel_values": tensor.unsqueeze(0)}  # Add batch dimension
    return tensor




class LandUseAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'land_use_app'

    model = None
    feature_extractor = None
    model_load_error = False

    def ready(self):
        """
        This method is called once per process when the application is ready.
        We load the large model here.
        """
        # Avoid loading if running migrations or other management commands
        # that don't need the model.
        try:
            from django.core.management import execute_from_command_line
            import sys
            # Simple check: if 'runserver' is not in the command line args,
            # or if specific commands that don't need the model are present, skip loading.
            if 'runserver' not in sys.argv:
                logger.info("Skipping model loading during management command.")
                return
        except Exception as e:
             logger.warning(f"Could not perform management command check: {e}. Attempting to load model.")


        logger.info("AppConfig ready. Attempting to load Segformer model...")

        try:
            # Load the model and feature extractor
            LandUseAppConfig.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
            # Set model to evaluation mode immediately
            LandUseAppConfig.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512").eval()

            logger.info("Segformer model loaded successfully in AppConfig.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load FLAIR UNet model in AppConfig: {e}", exc_info=True)
            LandUseAppConfig.model_load_error = True
            raise e
