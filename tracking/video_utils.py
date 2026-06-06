"""Video validation helpers for uploaded CCTV footage."""
from __future__ import annotations

import cv2
from pathlib import Path


def validate_video(path: str) -> tuple[bool, str]:
    """Check if OpenCV can open and read the video."""
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False, (
            "Cannot open video. Try converting to MP4 (H.264). "
            "Some phone/HEVC videos are not supported by OpenCV."
        )

    ret, frame = cap.read()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ret or frame is None:
        return False, "Video opened but no frames could be read."

    return True, f"{p.name} — {w}x{h}, {frames} frames @ {fps:.1f} FPS"
