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
        import sys
        try:
            if 'runserver' not in sys.argv:
                logger.info("Skipping model loading during management command.")
                return
        except Exception as e:
            logger.warning(f"Management command check failed: {e}. Attempting to load model anyway.")

        logger.info("AppConfig ready. Attempting to load FLAIR UNet model...")

        try:
            model_path = hf_hub_download(
                repo_id="IGNF/FLAIR-INC_rgbie_15cl_resnet34-unet",
                filename="FLAIR-INC_rgbie_15cl_resnet34-unet_weights.pth"
            )

            model = UNet(n_channels=4, n_classes=15)
            raw_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            if isinstance(raw_state_dict, dict) and 'model' in raw_state_dict:
                raw_state_dict = raw_state_dict['model']

            cleaned_state_dict = OrderedDict()
            for k, v in raw_state_dict.items():
                new_key = k.replace("model.seg_model.", "")
                cleaned_state_dict[new_key] = v

            model.load_state_dict(cleaned_state_dict, strict=False)
            model.eval()

            LandUseAppConfig.model = model
            LandUseAppConfig.feature_extractor = default_feature_extractor

            logger.info("FLAIR UNet model and custom feature extractor loaded successfully in AppConfig.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load FLAIR UNet model in AppConfig: {e}", exc_info=True)
            LandUseAppConfig.model_load_error = True
            raise e
