"""Video source utilities — VisDrone, UA-DETRAC, webcam, file."""
from __future__ import annotations

import os
from pathlib import Path

from config.settings import DEFAULT_DEMO_SEQUENCE, UA_DETRAC, VISDRONE_TRAIN, VISDRONE_VAL


def list_visdrone_sequences(split: str = "train") -> list[str]:
    root = VISDRONE_TRAIN if split == "train" else VISDRONE_VAL
    seq_dir = root / "sequences"
    if not seq_dir.exists():
        return []
    return sorted([d.name for d in seq_dir.iterdir() if d.is_dir()])


def get_visdrone_sequence_path(sequence: str, split: str = "train") -> str | None:
    root = VISDRONE_TRAIN if split == "train" else VISDRONE_VAL
    seq_path = root / "sequences" / sequence
    if not seq_path.exists():
        return None
    frames = sorted(seq_path.glob("*.jpg"))
    if not frames:
        return None
    return str(frames[0].parent)


def frames_to_video_writer(
    frames_dir: str, output_path: str, fps: float = 25.0, max_frames: int | None = 300
) -> str:
    """Create a temporary video from image sequence for OpenCV processing."""
    import cv2

    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    if max_frames:
        frames = frames[:max_frames]

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
    writer.release()
    return output_path


def resolve_video_source(source: str) -> str:
    """Resolve demo sequence names to actual paths."""
    if source.startswith("visdrone:"):
        seq = source.split(":", 1)[1]
        path = get_visdrone_sequence_path(seq)
        if path is None:
            raise FileNotFoundError(f"VisDrone sequence not found: {seq}")
        tmp = str(Path(__file__).parent.parent / "outputs" / f"demo_{seq}.mp4")
        return frames_to_video_writer(path, tmp)

    if source.startswith("detrac:"):
        seq = source.split(":", 1)[1]
        img_dir = UA_DETRAC / "DETRAC-Images" / seq
        if not img_dir.exists():
            raise FileNotFoundError(f"UA-DETRAC sequence not found: {seq}")
        tmp = str(Path(__file__).parent.parent / "outputs" / f"demo_{seq}.mp4")
        return frames_to_video_writer(str(img_dir), tmp)

    if os.path.exists(source):
        return source

    # Try default demo
    default_path = get_visdrone_sequence_path(DEFAULT_DEMO_SEQUENCE)
    if default_path:
        tmp = str(Path(__file__).parent.parent / "outputs" / f"demo_{DEFAULT_DEMO_SEQUENCE}.mp4")
        if not Path(tmp).exists():
            frames_to_video_writer(default_path, tmp)
        return tmp

    raise FileNotFoundError(f"Cannot resolve video source: {source}")
