# coding: utf-8

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

# MediaPipe singletons
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

yolo_model = YOLO("./yolov8n-face.pt")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def select_uniform_frames(frames: List[int], N: int) -> List[int]:
    if len(frames) <= N:
        return frames
    idx = np.linspace(0, len(frames) - 1, num=N, dtype=int)
    return [frames[i] for i in idx]

# ---------------------------------------------------------------------------
# Detectors: return List[(x1, y1, x2, y2, area)] in global coordinates
# ---------------------------------------------------------------------------

def detect_faces_mediapipe_facedetection(frame: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """MediaPipe FaceDetection (loose bbox)."""
    results = face_detection.process(frame)
    faces: List[Tuple[int, int, int, int, int]] = []
    if results.detections:
        h, w = frame.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(int(bb.xmin * w), 0)
            y1 = max(int(bb.ymin * h), 0)
            x2 = min(int((bb.xmin + bb.width) * w), w)
            y2 = min(int((bb.ymin + bb.height) * h), h)
            if x2 > x1 and y2 > y1:
                area = (x2 - x1) * (y2 - y1)
                faces.append((x1, y1, x2, y2, area))
    return faces


def detect_faces_mediapipe_facedetection_facemesh(frame: np.ndarray) -> list:
    """
    Hybrid face detector:
    1. Use MediaPipe FaceDetection to find face regions (even small/distant ones).
    2. For each detection, expand the bounding box slightly and crop the region.
    3. Run MediaPipe FaceMesh on the cropped region to get precise landmarks.
    4. Compute tight bounding box from landmarks and convert back to global coordinates.

    This combines high recall (FaceDetection) with accurate, tight crops (FaceMesh).

    Args:
        frame (np.ndarray): Input RGB image.

    Returns:
        list: List of (x1, y1, x2, y2, area) in global frame coordinates.
    """
    h, w = frame.shape[:2]
    faces = []

    # --- Stage 1: detect face regions using FaceDetection ---
    results_bbox = face_detection.process(frame)
    if not results_bbox.detections:
        return []  # No faces found

    # Process each detected face region
    for det in results_bbox.detections:
        bb = det.location_data.relative_bounding_box
        rx1 = int(bb.xmin * w)
        ry1 = int(bb.ymin * h)
        rx2 = int((bb.xmin + bb.width) * w)
        ry2 = int((bb.ymin + bb.height) * h)

        # Expand region to ensure full face is visible for FaceMesh
        margin_x = int((rx2 - rx1) * 0.3)  # 30% margin
        margin_y = int((ry2 - ry1) * 0.4)  # more vertical (head)

        reg_x1 = max(rx1 - margin_x, 0)
        reg_y1 = max(ry1 - margin_y, 0)
        reg_x2 = min(rx2 + margin_x, w)
        reg_y2 = min(ry2 + margin_y, h)

        # Extract region for FaceMesh
        region = frame[reg_y1:reg_y2, reg_x1:reg_x2]

        # --- Stage 2: run FaceMesh on the expanded region ---
        results_mesh = face_mesh.process(region)

        if results_mesh.multi_face_landmarks:
            # Use only the first detected face in this region
            landmarks = results_mesh.multi_face_landmarks[0]
            lh, lw = region.shape[:2]
            xs = [int(land.x * lw) for land in landmarks.landmark]
            ys = [int(land.y * lh) for land in landmarks.landmark]

            # Local tight crop
            lx1, lx2 = max(min(xs), 0), min(max(xs), lw)
            ly1, ly2 = max(min(ys), 0), min(max(ys), lh)

            # Convert back to global coordinates
            gx1 = reg_x1 + lx1
            gy1 = reg_y1 + ly1
            gx2 = reg_x1 + lx2
            gy2 = reg_y1 + ly2

            if gx2 > gx1 and gy2 > gy1:
                area = (gx2 - gx1) * (gy2 - gy1)
                faces.append((gx1, gy1, gx2, gy2, area))

    return faces


def detect_faces_yolo(im_rgb: np.ndarray) -> list:
    """
    Detect faces using YOLOv8 with tracking (ByteTrack).
    Uses tracking for stable bounding boxes.
    Resets tracker internally at the start of a new video.
    """
    h, w = im_rgb.shape[:2]

    # Reset tracker before the first frame (call will be from outside, but keep it here)
    if hasattr(yolo_model.predictor, "trackers"):
        yolo_model.predictor.trackers[0].reset()

    results = yolo_model.track(
                im_rgb, persist=True, imgsz=640, conf=0.01, iou=0.5,
                augment=False, device=0, verbose=False
            )

    faces = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w)
            y2 = min(y2, h)
            if x2 > x1 and y2 > y1:
                area = (x2 - x1) * (y2 - y1)
                faces.append((x1, y1, x2, y2, area))
    return faces


# ---------------------------------------------------------------------------
# Main pipeline: get_face_crops
# ---------------------------------------------------------------------------

def _normalize_detector(name: str) -> str:
    n = (name or "").lower()
    if n in ("mp_fd", "mpfd", "mediapipe", "mp"): return "mp_fd"
    if n in ("mp_hybrid", "mph", "hybrid"):      return "mp_hybrid"
    if n in ("yolo", "yolov8", "yolo8"):         return "yolo"
    return "mp_fd"


def get_face_crops(
    video_path: str,
    segment_length: int,
    *,
    detector: str = "mp_fd",              # "mp_fd" | "mp_hybrid" | "yolo"
    relative_threshold: float = 0.3,      # 0..1, filter by bbox area relative to max
    reuse_last: bool = True,              # if no faces — reuse the last valid crop
    fallback_fullframe: bool = True,      # before the first valid face — use full frame
    average_multi_face: bool = True,      # average multiple faces in a frame
) -> Tuple[str, List[np.ndarray]]:
    det = _normalize_detector(detector)

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need_frames = select_uniform_frames(list(range(total_frames)), segment_length)

    face_images: List[np.ndarray] = []
    last_valid_face: np.ndarray | None = None
    fallback_count = 0
    t = 0

    while True:
        ret, im0 = cap.read()
        if not ret:
            break

        if t in need_frames:
            im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            # Detection
            if det == "mp_fd":
                faces = detect_faces_mediapipe_facedetection(im_rgb)
            elif det == "mp_hybrid":
                faces = detect_faces_mediapipe_facedetection_facemesh(im_rgb)
            else:  # yolo
                faces = detect_faces_yolo(im_rgb)

            if faces:
                max_area = max(a for *_, a in faces)
                thr = float(relative_threshold) * max_area
                valid = [(x1, y1, x2, y2) for (x1, y1, x2, y2, a) in faces if a >= thr]

                if valid:
                    crops = [im_rgb[y1:y2, x1:x2] for (x1, y1, x2, y2) in valid if (x2 > x1 and y2 > y1)]
                    if crops:
                        if average_multi_face and len(crops) > 1:
                            max_h = max(c.shape[0] for c in crops)
                            max_w = max(c.shape[1] for c in crops)
                            resized = [
                                cv2.resize(c, (max_w, max_h), interpolation=cv2.INTER_AREA)
                                if (c.shape[0] != max_h or c.shape[1] != max_w) else c
                                for c in crops
                            ]
                            avg = np.mean(np.stack(resized), axis=0)
                            avg = np.clip(avg, 0, 255).astype(np.uint8)
                            face_images.append(avg)
                            last_valid_face = avg
                        else:
                            biggest = max(crops, key=lambda c: c.shape[0] * c.shape[1])
                            face_images.append(biggest)
                            last_valid_face = biggest
                        t += 1
                        continue

            # Fallbacks
            if reuse_last and last_valid_face is not None:
                face_images.append(last_valid_face)
            elif fallback_fullframe:
                face_images.append(im_rgb)
                fallback_count += 1
            else:
                pass

        t += 1

    cap.release()

    # Trim leading full frames up to the first valid detection
    if fallback_count and last_valid_face is not None:
        face_images = face_images[fallback_count:]

    return video_name, face_images


# ---------------------------------------------------------------------------
# (optional) Helper — video duration, if needed
# ---------------------------------------------------------------------------

def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        raise ValueError("FPS is zero, cannot compute duration.")
    return total_frames / fps
