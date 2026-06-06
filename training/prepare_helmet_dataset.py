"""Convert Pascal VOC helmet annotations to YOLO format."""
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import HELMET_CLASSES, HELMET_DATASET_YOLO, HELMET_RAW

CLASS_MAP = {
    "With Helmet": 0,
    "Without Helmet": 1,
    "with helmet": 0,
    "without helmet": 1,
}


def parse_voc_annotation(xml_path: Path) -> tuple[str, list[tuple[int, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[name]
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        boxes.append((cls_id, cx, cy, bw, bh))
    return filename, boxes


def prepare(split_ratio: float = 0.8, seed: int = 42) -> Path:
    random.seed(seed)
    ann_dir = HELMET_RAW / "annotations"
    img_dir = HELMET_RAW / "images"

    if HELMET_DATASET_YOLO.exists():
        shutil.rmtree(HELMET_DATASET_YOLO)

    for split in ("train", "val"):
        (HELMET_DATASET_YOLO / "images" / split).mkdir(parents=True)
        (HELMET_DATASET_YOLO / "labels" / split).mkdir(parents=True)

    xml_files = sorted(ann_dir.glob("*.xml"))
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * split_ratio)
    splits = {"train": xml_files[:split_idx], "val": xml_files[split_idx:]}

    for split, files in splits.items():
        for xml_path in files:
            filename, boxes = parse_voc_annotation(xml_path)
            stem = Path(filename).stem
            for ext in (".png", ".jpg", ".jpeg"):
                src = img_dir / f"{stem}{ext}"
                if src.exists():
                    break
            else:
                continue
            shutil.copy2(src, HELMET_DATASET_YOLO / "images" / split / src.name)
            label_path = HELMET_DATASET_YOLO / "labels" / split / f"{stem}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                for cls_id, cx, cy, bw, bh in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    data_yaml = {
        "path": str(HELMET_DATASET_YOLO.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(HELMET_CLASSES),
        "names": HELMET_CLASSES,
    }
    yaml_path = HELMET_DATASET_YOLO / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Dataset prepared: {len(splits['train'])} train, {len(splits['val'])} val")
    print(f"YAML: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    prepare()
