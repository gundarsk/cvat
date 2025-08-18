# model_handler.py  (SAM2 version)

import os
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use env vars so you can control them from function.yaml
        self.sam2_checkpoint = os.environ.get(
            "SAM2_CHECKPOINT",
            "/opt/nuclio/sam2.1_hiera_large.pt",
        )
        self.sam2_config = os.environ.get(
            "SAM2_CONFIG",
            "/opt/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        )

        # Build SAM2 and predictor
        sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def handle(self, image):
        # Same flow as SAM v1: set image, get image embedding
        self.predictor.set_image(np.array(image))
        features = self.predictor.get_image_embedding()
        return features
