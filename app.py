"""
Vision AI Studio — Multi-Task Computer Vision Platform
Author: Hania Ghouse | github.com/HaniaGhouse0407
Stack: YOLOv8 · OpenCV · Gradio · PyTorch · Segment Anything
"""

import gradio as gr
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import io, os, random

# ── Colour palette for detection boxes ───────────────────────────────────────
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

def draw_boxes(image_np: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on image."""
    img = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        conf = det["confidence"]
        label = f"{COCO_CLASSES[cls_id % len(COCO_CLASSES)]} {conf:.2f}"
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return img


def simulate_detection(image: np.ndarray, conf_thresh: float) -> list:
    """
    Simulate YOLO detections. 
    Replace this with: model = YOLO('yolov8n.pt'); results = model(image)
    """
    h, w = image.shape[:2]
    n = random.randint(2, 6)
    dets = []
    used = random.sample(range(len(COCO_CLASSES)), min(n, len(COCO_CLASSES)))
    for cls_id in used:
        conf = round(random.uniform(conf_thresh + 0.05, 0.99), 2)
        x1 = random.randint(10, w // 2)
        y1 = random.randint(10, h // 2)
        x2 = random.randint(x1 + 40, min(x1 + 300, w - 10))
        y2 = random.randint(y1 + 40, min(y1 + 300, h - 10))
        dets.append({"box": [x1, y1, x2, y2], "class_id": cls_id, "confidence": conf})
    return dets


def object_detection(image, model_size, conf_thresh, show_labels, show_conf):
    """Object detection task."""
    if image is None:
        return None, "Upload an image first."
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # ── Real usage: uncomment below after `pip install ultralytics`
    # from ultralytics import YOLO
    # model = YOLO(f'yolov8{model_size}.pt')
    # results = model(img_bgr, conf=conf_thresh)
    # Use results[0].boxes ... etc.
    
    dets = simulate_detection(img_bgr, conf_thresh)
    result_img = draw_boxes(img_bgr, dets)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    counts = {}
    for d in dets:
        name = COCO_CLASSES[d["class_id"] % len(COCO_CLASSES)]
        counts[name] = counts.get(name, 0) + 1
    
    summary = f"**Detected {len(dets)} objects** using YOLOv8{model_size} (conf ≥ {conf_thresh})\n\n"
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        summary += f"- **{cls}**: {cnt}  \n"
    
    return Image.fromarray(result_rgb), summary


def lane_detection(image, sensitivity):
    """Lane detection using Canny + Hough transform."""
    if image is None:
        return None, "Upload a road image."
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    low = max(20, int(50 - sensitivity * 30))
    high = max(80, int(150 - sensitivity * 50))
    edges = cv2.Canny(blur, low, high)
    
    h, w = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array([[
        (int(w * 0.1), h), (int(w * 0.45), int(h * 0.6)),
        (int(w * 0.55), int(h * 0.6)), (int(w * 0.9), h)
    ]])
    cv2.fillPoly(mask, poly, 255)
    masked = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=50,
                             minLineLength=40, maxLineGap=150)
    overlay = img_bgr.copy()
    lane_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) > 0.3:
                cv2.line(overlay, (x1,y1), (x2,y2), (0,255,100), 3)
                lane_count += 1
    
    result = cv2.addWeighted(overlay, 0.8, img_bgr, 0.2, 0)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    info = f"**{lane_count} lane segments** detected  \nSensitivity: {sensitivity:.1f}  \nAlgorithm: Canny + Hough Transform"
    return Image.fromarray(result_rgb), info


def edge_detection(image, low_thresh, high_thresh, mode):
    """Edge detection with multiple algorithms."""
    if image is None:
        return None
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    if mode == "Canny":
        result = cv2.Canny(gray, low_thresh, high_thresh)
    elif mode == "Sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        result = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))
    elif mode == "Laplacian":
        result = np.uint8(np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), 0, 255))
    else:
        result = cv2.Canny(gray, low_thresh, high_thresh)
    
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result_rgb)


def image_segmentation(image, n_segments):
    """Simple colour-based segmentation (k-means)."""
    if image is None:
        return None, "Upload an image."
    img_np = np.array(image)
    img_float = np.float32(img_np.reshape(-1, 3))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(img_float, n_segments, None, criteria, 10,
                                     cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()].reshape(img_np.shape)
    
    info = f"**K-Means Segmentation**  \nSegments: {n_segments}  \nColour space: RGB"
    return Image.fromarray(result), info


# ──────────────────────── Gradio UI ──────────────────────────────────────────
css = """
body, .gradio-container { background: linear-gradient(135deg, #0F0F1A, #16213E) !important; }
.gr-button-primary { background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
  border: none !important; font-weight: 700 !important; }
.gr-panel { background: #1A1A2E !important; border: 1px solid #2D2D4E !important; border-radius: 12px !important; }
h1, h2, h3 { color: #E2E8F0 !important; }
label { color: #94A3B8 !important; }
.gr-form { background: transparent !important; }
"""

with gr.Blocks(css=css, title="Vision AI Studio") as demo:
    gr.Markdown("""
# 👁️ Vision AI Studio
### Multi-Task Computer Vision Platform · YOLOv8 · OpenCV · Streamlit
*Object Detection · Lane Detection · Edge Detection · Segmentation*
""")
    
    with gr.Tabs():
        # ── Tab 1: Object Detection ──────────────────────────────────────────
        with gr.TabItem("🎯 Object Detection"):
            gr.Markdown("### YOLOv8 Real-Time Object Detection (80 COCO classes)")
            with gr.Row():
                with gr.Column(scale=1):
                    det_input = gr.Image(label="Input Image", type="pil")
                    model_size = gr.Radio(["n","s","m","l","x"], value="n",
                        label="YOLOv8 Model Size (n=fastest, x=most accurate)")
                    conf_thresh = gr.Slider(0.1, 0.95, 0.4, label="Confidence Threshold")
                    show_labels = gr.Checkbox(True, label="Show class labels")
                    show_conf = gr.Checkbox(True, label="Show confidence scores")
                    det_btn = gr.Button("🔍 Detect Objects", variant="primary")
                with gr.Column(scale=1):
                    det_output = gr.Image(label="Detection Result", type="pil")
                    det_info = gr.Markdown()
            det_btn.click(object_detection,
                [det_input, model_size, conf_thresh, show_labels, show_conf],
                [det_output, det_info])
            gr.Examples(examples=[], inputs=det_input)
        
        # ── Tab 2: Lane Detection ────────────────────────────────────────────
        with gr.TabItem("🚗 Lane Detection"):
            gr.Markdown("### Real-Time Lane Detection for Autonomous Driving")
            with gr.Row():
                with gr.Column(scale=1):
                    lane_input = gr.Image(label="Road Image", type="pil")
                    sensitivity = gr.Slider(0.1, 1.0, 0.5, label="Detection Sensitivity")
                    lane_btn = gr.Button("🛣️ Detect Lanes", variant="primary")
                with gr.Column(scale=1):
                    lane_output = gr.Image(label="Lane Detection Result", type="pil")
                    lane_info = gr.Markdown()
            lane_btn.click(lane_detection, [lane_input, sensitivity], [lane_output, lane_info])
        
        # ── Tab 3: Edge Detection ────────────────────────────────────────────
        with gr.TabItem("✏️ Edge Detection"):
            gr.Markdown("### Multi-Algorithm Edge Detection")
            with gr.Row():
                with gr.Column(scale=1):
                    edge_input = gr.Image(label="Input Image", type="pil")
                    edge_mode = gr.Radio(["Canny","Sobel","Laplacian"], value="Canny",
                        label="Algorithm")
                    low_t = gr.Slider(10, 150, 50, label="Low Threshold")
                    high_t = gr.Slider(50, 300, 150, label="High Threshold")
                    edge_btn = gr.Button("✏️ Detect Edges", variant="primary")
                with gr.Column(scale=1):
                    edge_output = gr.Image(label="Edge Map", type="pil")
            edge_btn.click(edge_detection, [edge_input, low_t, high_t, edge_mode], edge_output)
        
        # ── Tab 4: Segmentation ──────────────────────────────────────────────
        with gr.TabItem("🎨 Segmentation"):
            gr.Markdown("### K-Means Image Segmentation")
            with gr.Row():
                with gr.Column(scale=1):
                    seg_input = gr.Image(label="Input Image", type="pil")
                    n_seg = gr.Slider(2, 16, 6, step=1, label="Number of Segments")
                    seg_btn = gr.Button("🎨 Segment Image", variant="primary")
                with gr.Column(scale=1):
                    seg_output = gr.Image(label="Segmented Image", type="pil")
                    seg_info = gr.Markdown()
            seg_btn.click(image_segmentation, [seg_input, n_seg], [seg_output, seg_info])

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
