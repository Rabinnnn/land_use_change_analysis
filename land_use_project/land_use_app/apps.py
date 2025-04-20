from django.apps import AppConfig
import logging
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

logger = logging.getLogger(__name__)

class LandUseAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'land_use_app'

    # Class attributes to hold the loaded model and feature extractor
    # They will be populated in the ready() method
    feature_extractor = None
    model = None
    model_load_error = False # Flag to indicate if loading failed

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
            # This check isn't foolproof but prevents loading during e.g., 'makemigrations'.
            if 'runserver' not in sys.argv:
                 logger.info("Skipping model loading during management command.")
                 return # Skip loading
        except Exception as e:
             logger.warning(f"Could not perform management command check: {e}. Attempting to load model.")


        logger.info("AppConfig ready. Attempting to load Segformer model...")

        try:
            # Load the model and feature extractor
            # These will be class attributes accessible from views
            LandUseAppConfig.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
            # Set model to evaluation mode immediately
            LandUseAppConfig.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512").eval()

            logger.info("Segformer model loaded successfully in AppConfig.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load Segformer model in AppConfig: {e}", exc_info=True)
            # Set the error flag
            LandUseAppConfig.model_load_error = True
            # Prevent the application from starting if loading the model fails
            raise e 


