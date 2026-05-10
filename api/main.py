import time
import io
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ---------- Configuration ----------
MODEL_PATH = "models/best.onnx"
# MODEL_PATH = "models/finetuned/best.onnx"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.4
INPUT_SIZE = (640, 640)          # model expects 640x640
CLASS_NAMES = ["color", "cut", "fold", "glue", "poke"]

# ---------- some helper functions ----------
def letterbox(img, new_shape, color=(114, 114, 114)):
    """Resize and pad image to new_shape while keeping aspect ratio."""
    shape = img.shape[:2]  # current [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # we keep the ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # and compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    # then resize
    if (shape[0] != new_unpad[1]) or (shape[1] != new_unpad[0]):
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, (r, dw, dh)

def preprocess(image_bytes, input_size=INPUT_SIZE):
    """Convert image bytes -> letterboxed numpy array (1,3,640,640) float32 [0,1]."""
    # Read image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    original_shape = img.shape[:2]  # (height, width)

    # convert BGR to RGB thank you cv2 (;
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Letterbox
    img, (ratio, dw, dh) = letterbox(img, input_size)

    # we normalize to [0,1] and HWC -> CHW
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)          # (3, 640, 640)
    img = np.expand_dims(img, axis=0)     # (1, 3, 640, 640)

    return img, original_shape, ratio, dw, dh


def parse_output(output, original_shape, ratio, dw, dh,
                 conf_thresh=CONFIDENCE_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    # output shape: (1, 9, N) -> squeeze -> (9, N)
    output = output[0]

    # First 4 rows = box coordinates (cx, cy, w, h)
    boxes = output[:4, :]

    # Rows 4..8 = class scores (already include objectness)
    class_scores = output[4:, :]          # (5, N)

    # Get max confidence and class ID for each prediction
    scores = class_scores.max(axis=0)     # (N,)
    class_ids = class_scores.argmax(axis=0)  # (N,)

    # Filter by threshold
    mask = scores > conf_thresh
    if not mask.any():
        return {
            "defective": False,
            "confidence": 0.0,
            "defect_type": None,
            "bbox": [],
            "inference_time_ms": 0.0
        }

    cx = boxes[0, mask]
    cy = boxes[1, mask]
    w  = boxes[2, mask]
    h  = boxes[3, mask]
    confs = scores[mask]
    cls   = class_ids[mask]

    # Convert center to corners
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Rescale to original image
    gain = ratio
    pad_x = dw
    pad_y = dh
    x1_orig = (x1 - pad_x) / gain
    y1_orig = (y1 - pad_y) / gain
    x2_orig = (x2 - pad_x) / gain
    y2_orig = (y2 - pad_y) / gain

    orig_h, orig_w = original_shape
    x1_orig = np.clip(x1_orig, 0, orig_w)
    y1_orig = np.clip(y1_orig, 0, orig_h)
    x2_orig = np.clip(x2_orig, 0, orig_w)
    y2_orig = np.clip(y2_orig, 0, orig_h)

    boxes_rescaled = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_rescaled.tolist(), confs.tolist(),
                               conf_thresh, iou_thresh)
    if len(indices) == 0:
        return {
            "defective": False,
            "confidence": 0.0,
            "defect_type": None,
            "bbox": [],
            "inference_time_ms": 0.0
        }

    best_idx = indices[0]
    best_bbox = boxes_rescaled[best_idx].tolist()
    best_conf = float(confs[best_idx])
    defect_type = CLASS_NAMES[int(cls[best_idx])]

    return {
        "defective": True,
        "confidence": round(best_conf, 4),
        "defect_type": defect_type,
        "bbox": [round(x, 1) for x in best_bbox],
        "inference_time_ms": 0.0
    }

# ---------- for lifespan we load ONNX model once ----------
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ONNX model at startup
    ml_model["session"] = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    ml_model["input_name"] = ml_model["session"].get_inputs()[0].name
    print("Model loaded successfully.")
    yield
    # Clean up
    ml_model.clear()

# ---------- FastAPI app ----------
app = FastAPI(title="Industrial Defect Detection API",
              lifespan=lifespan)

# we add CORS to be able to access from React frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    # TODO: in production, restrict this to our frontend domain
    allow_origins=["*"],   # important, we might restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file")

    # Preprocess our image to get model input
    try:
        img_tensor, original_shape, ratio, dw, dh = preprocess(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Inference
    start = time.perf_counter()
    outputs = ml_model["session"].run(
        None, {ml_model["input_name"]: img_tensor}
    )
    inference_time = (time.perf_counter() - start) * 1000  # ms

    # we parse results before return it
    result = parse_output(outputs[0], original_shape, ratio, dw, dh)
    result["inference_time_ms"] = round(inference_time, 2)

    return result
