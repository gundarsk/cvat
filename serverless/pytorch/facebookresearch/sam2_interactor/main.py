import json, base64, io
import numpy as np
from PIL import Image

from model_handler import ModelHandler

def _ensure_json(body):
    if isinstance(body, (bytes, bytearray)):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body

def _extract_prompts(d):
    """
    Accept several common forms:
      • point_coords + point_labels
      • positive_points / negative_points
      • points: [{x,y,label|type}]
      • box (xyxy) as list or dict
    """
    pc, pl, bb = None, None, None

    # point_coords + point_labels
    if "point_coords" in d and "point_labels" in d:
        pc = d["point_coords"]
        pl = d["point_labels"]

    # positive_points / negative_points
    if ("positive_points" in d) or ("negative_points" in d):
        pos = d.get("positive_points", []) or []
        neg = d.get("negative_points", []) or []
        pc = (pc or []) + pos + neg
        pl = (pl or []) + [1]*len(pos) + [0]*len(neg)

    # points: [{x,y,label|type}]
    if "points" in d and isinstance(d["points"], list):
        pts = d["points"]
        xs, ys, lbs = [], [], []
        for p in pts:
            x, y = p.get("x"), p.get("y")
            if x is None or y is None:
                continue
            lab = p.get("label")
            if lab is None:
                t = (p.get("type") or "").lower()
                lab = 1 if t in ("pos", "positive", "fg", "fgnd", "foreground") else 0
            xs.append(float(x)); ys.append(float(y)); lbs.append(int(lab))
        if xs:
            pc = (pc or []) + list(map(list, zip(xs, ys)))
            pl = (pl or []) + lbs

    # box (xyxy)
    if "box" in d:
        b = d["box"]
        if isinstance(b, dict):
            bb = [b[k] for k in ("x1","y1","x2","y2") if k in b]
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            bb = list(map(float, b))

    return pc, pl, bb

def init_context(context):
    context.logger.info("Init SAM2 interactor … 0%")
    context.user_data.model = ModelHandler()
    context.logger.info("Init SAM2 interactor … 100%")

def handler(context, event):
    try:
        data = _ensure_json(event.body)

        # decode image
        if "image" not in data:
            raise ValueError("Missing 'image' base64 in request")
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image)

        # parse prompts
        pc, pl, bb = _extract_prompts(data)
        multimask = bool(data.get("multimask_output", True))

        masks, scores = context.user_data.model.predict_masks(
            image_np, point_coords=pc, point_labels=pl, box=bb, multimask_output=multimask
        )

        # Convert masks → COCO RLE for CVAT (size [H,W])
        # pycocotools expects Fortran-ordered uint8
        from pycocotools import mask as cocomask
        results = []
        H, W = masks.shape[-2], masks.shape[-1]
        for m, s in zip(masks, np.array(scores).reshape(-1)):
            m_uint8 = (np.asarray(m) > 0.5).astype(np.uint8, copy=False)  # (H,W)
            rle = cocomask.encode(np.asfortranarray(m_uint8))
            # rle is dict with 'counts' as bytes → decode to str
            rle["counts"] = rle["counts"].decode("utf-8")
            rle["size"] = [H, W]
            results.append({"rle": rle, "score": float(s)})

        return context.Response(
            body=json.dumps({"masks": results}),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"SAM2 interactor error: {e}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
