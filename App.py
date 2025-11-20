import os
import cv2
import numpy as np
import gradio as gr
from functools import lru_cache
from rfdetr import RFDETRBase

# ---------- Modelo (cached) ----------
@lru_cache(maxsize=2)
def load_model(weights: str):
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Pesos no encontrados: {weights}")
    print(f"[INFO] Cargando modelo con pesos: {weights}")
    model = RFDETRBase(pretrain_weights=weights)
    return model

# ---------- Utilidad IoU / NMS ----------
def box_iou(box, boxes):
    """
    IoU entre una caja (4,) y un array de cajas (N, 4).
    Formato esperado: [x1, y1, x2, y2]
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter + 1e-6
    return inter / union


def nms_xyxy(xyxy, conf, iou_th):
    """
    Non-Maximum Suppression sencillo sobre arrays numpy.
    xyxy: (N, 4), conf: (N,), iou_th: float
    Devuelve índices seleccionados.
    """
    if len(xyxy) == 0:
        return np.array([], dtype=int)

    order = np.argsort(conf)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = box_iou(xyxy[i], xyxy[order[1:]])
        remaining = np.where(ious <= iou_th)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=int)

# ---------- Dibujar detecciones ----------
def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

def draw_detections(frame: np.ndarray, result, iou_threshold: float) -> np.ndarray:
    """
    Dibuja todas las detecciones del objeto Detections sobre el frame,
    aplicando NMS con el IoU dado.
    """
    if result is None:
        return frame

    xyxy = getattr(result, "xyxy", None)
    conf = getattr(result, "confidence", None)
    class_ids = getattr(result, "class_id", None)
    names = getattr(result, "names", None)

    if xyxy is None or conf is None or class_ids is None:
        return frame

    xyxy = np.array(xyxy)
    conf = np.array(conf)
    class_ids = np.array(class_ids)

    # NMS con el umbral de IoU del slider
    if iou_threshold is not None:
        keep = nms_xyxy(xyxy, conf, float(iou_threshold))
        xyxy = xyxy[keep]
        conf = conf[keep]
        class_ids = class_ids[keep]

    # Iterar sobre TODAS las detecciones filtradas
    for i in range(xyxy.shape[0]):
        coords = xyxy[i]
        if len(coords) != 4:
            continue

        x1, y1, x2, y2 = map(int, coords)
        cls = int(class_ids[i])
        c = float(conf[i])
        label = f"{names.get(cls, str(cls))} {c:.2f}" if names else f"{cls} {c:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_label(frame, label, x1, y1)

    return frame


# ---------- Inferencia sobre imagen subida ----------
def detect_image(uploaded_image, weights, conf, iou, max_side):
    if uploaded_image is None:
        return None

    # RGB -> BGR
    frame = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)

    # Redimensionar si es necesario
    h, w = frame.shape[:2]
    if max(h, w) > max_side > 0:
        scale = max_side / float(max(h, w))
        frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    # Cargar modelo según los pesos seleccionados
    model = load_model(weights)

    # IMPORTANTE:
    # para una sola imagen, predict() devuelve UN objeto Detections,
    # NO una lista. No se hace [0].
    result = model.predict(frame, threshold=float(conf))

    annotated = draw_detections(frame, result, iou_threshold=float(iou))
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ---------- UI ----------
MODEL_CHOICES = [
    "Basket-Model/checkpoint_best_total.pth",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt"
]

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Detección de objetos en captura de pantalla")

        uploaded_img = gr.Image(label="Sube captura de pantalla", type="numpy")
        out = gr.Image(label="Detections")

        with gr.Row():
            weights = gr.Dropdown(
                MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="Model weights"
            )
            conf = gr.Slider(
                0.05, 0.95,
                value=0.25,
                step=0.05,
                label="Confidence"
            )
            iou = gr.Slider(
                0.10, 0.90,
                value=0.45,
                step=0.05,
                label="IoU (NMS threshold)"
            )
            max_side = gr.Slider(
                480, 1920,
                value=960,
                step=16,
                label="Max image side (downscale)"
            )

        btn = gr.Button("Detectar objetos")

        # Ahora se pasan pesos, conf e IoU al backend
        btn.click(
            fn=detect_image,
            inputs=[uploaded_img, weights, conf, iou, max_side],
            outputs=out
        )

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", 8080)))
    )
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=False)
