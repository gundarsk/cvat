import os
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg  = os.environ.get("SAM2_CONFIG", "/opt/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
        ckpt = os.environ.get("SAM2_CHECKPOINT", "/opt/nuclio/weights/sam2.1_hiera_large.pt")
        model = build_sam2(cfg, ckpt, device=str(self.device))
        self.predictor = SAM2ImagePredictor(model)

    def predict_masks(self, image_np, point_coords=None, point_labels=None, box=None, multimask_output=True):
        # image_np: HxWx3 uint8 RGB
        self.predictor.set_image(image_np)

        # Normalize inputs
        pc = None
        pl = None
        bb = None
        if point_coords is not None and len(point_coords) > 0:
            pc = np.asarray(point_coords, dtype=np.float32)  # Nx2 (x,y) in pixels
            pl = np.asarray(point_labels, dtype=np.int32)    # N
        if box is not None:
            bb = np.asarray(box, dtype=np.float32)  # [x1,y1,x2,y2]

        # SAM2ImagePredictor API mirrors SAM: returns (masks, scores, logits)
        masks, scores, _ = self.predictor.predict(
            point_coords=pc, point_labels=pl, box=bb, multimask_output=multimask_output
        )
        # masks: (K,H,W) boolean/float; scores: (K,)
        return masks, scores
