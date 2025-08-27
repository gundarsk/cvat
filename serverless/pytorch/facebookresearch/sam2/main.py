# main.py
# Copyright (C) CVAT.ai
# SPDX-License-Identifier: MIT

import json
import base64
import io
from PIL import Image

from model_handler import ModelHandler

def _ensure_json(body):
    # Nuclio may pass bytes or dict; normalize to dict
    if isinstance(body, (bytes, bytearray)):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body

def init_context(context):
    context.logger.info("Init context...  0%")
    context.user_data.model = ModelHandler()
    context.logger.info("Init context...100%")

def handler(context, event):
    try:
        data = _ensure_json(event.body)
        if "image" not in data:
            raise ValueError("Missing 'image' field (base64-encoded) in request JSON")

        # Decode image
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf).convert("RGB")

        # Get embedding tensor from model
        features = context.user_data.model.handle(image)

        # Serialize to base64 (CPU tensor for portability)
        import torch
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().contiguous().numpy()

        blob_b64 = base64.b64encode(features).decode("ascii") if isinstance(features, (bytes, bytearray)) \
                   else base64.b64encode(features.tobytes()).decode("ascii")

        return context.Response(
            body=json.dumps({"blob": blob_b64}),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"sam2 handler error: {e}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
