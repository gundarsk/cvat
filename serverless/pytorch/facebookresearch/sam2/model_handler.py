# model_handler.py

import os
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Config & checkpoint come from env (function.yaml)
        sam2_cfg = os.environ.get("SAM2_CONFIG", "/opt/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
        sam2_ckpt = os.environ.get("SAM2_CHECKPOINT", "/opt/nuclio/weights/sam2.1_hiera_large.pt")

        model = build_sam2(sam2_cfg, sam2_ckpt, device=str(self.device))
        self.predictor = SAM2ImagePredictor(model)

    def handle(self, pil_image):
        # pil_image → numpy RGB
        image_np = np.array(pil_image)
        self.predictor.set_image(image_np)

        # Return image embedding (torch.Tensor)
        emb = self.predictor.get_image_embedding()  # (C, H', W') tensor
        return emb
