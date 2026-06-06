# Smart Traffic Intelligence System — Bengaluru Gridlock Challenge

Production-quality AI traffic analytics platform for CCTV footage analysis, violation detection, and intelligent decision support for Bengaluru traffic authorities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (UI)                      │
│  Live Feed │ Metrics │ Heatmaps │ AI Advisory Panel │ Exports   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   Traffic Pipeline (Orchestrator)                │
│  YOLO11s Detection → ByteTrack → Analytics Modules              │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
 Vehicle   Helmet    Speed     Wrong-Way   Congestion
 Detector  Detector  Estimator  Detector   Analyzer
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Heatmap      Movement       Risk Score
         Generator    Analysis       Calculator
                             │
                             ▼
                    AI Advisory Engine
                    (Rule-based DSS)
```

## Features

| Module | Description |
|--------|-------------|
| Vehicle Detection | YOLO11s pretrained — car, motorcycle, bus, truck, bicycle |
| Multi-Object Tracking | ByteTrack with persistent IDs and trajectories |
| Vehicle Counting | Per-class counts with live totals |
| Congestion Classification | Low / Medium / High (0–100 score) |
| Helmet Violation Detection | Fine-tuned YOLO11s on helmet dataset |
| Wrong-Way Detection | Configurable flow direction + trajectory analysis |
| Speed Estimation | Perspective-scaled km/h with overspeeding alerts |
| Movement Analysis | Direction statistics and flow imbalance |
| Traffic Risk Score | Weighted composite: congestion + violations |
| Heatmaps | Density hotspots and violation zones |
| AI Advisory Engine | Human-readable recommendations with priority levels |
| Reports | CSV, JSON, and daily analytics summaries |

## Project Structure

```
GRID/
├── datasets/          # Processed YOLO helmet dataset
├── models/            # Trained helmet model weights
├── training/          # Dataset prep + helmet fine-tuning
├── tracking/          # Detection, tracking, pipeline
├── analytics/         # Congestion, speed, heatmap, advisory, reports
├── dashboard/         # Streamlit UI
├── reports/           # Generated exports
├── outputs/           # Videos, heatmaps, screenshots
├── config/            # Settings and paths
├── app.py             # CLI entry point
└── requirements.txt
```

## Installation

```bash
cd D:\CODE\MODELS\GRID

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Hardware:** RTX 4060, Python 3.11, Windows

On first run, YOLO11s pretrained weights (`yolo11s.pt`) download automatically.

**GPU setup (RTX 4060):** Install CUDA-enabled PyTorch for training/inference acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Without CUDA, the system runs on CPU (slower but functional).

## Quick Start

### 1. Train Helmet Model (one-time)

```bash
python app.py --train-helmet
# Or with custom params:
python training/train_helmet.py --epochs 50 --batch 16 --device 0
```

### 2. Run CLI Inference (VisDrone demo)

```bash
python app.py --source visdrone:uav0000078_00401_v --max-frames 150 --export
```

### 3. Launch Dashboard

```bash
python app.py --dashboard
# Or directly:
streamlit run dashboard/app.py
```

### 4. Custom Video / Webcam

```bash
python app.py --source path/to/video.mp4 --max-frames 300 --export
python app.py --source 0 --max-frames 500
```

## Configuration

Edit `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EXPECTED_FLOW_DIRECTION` | `(0, -1)` | Expected traffic flow vector |
| `PIXELS_PER_METER` | `8.0` | Speed calibration factor |
| `SPEED_LIMIT_KMH` | `60` | Overspeeding threshold |
| `CONFIDENCE_THRESHOLD` | `0.35` | Detection confidence |

## Datasets

Located in parent `MODELS/` folder:

- **VisDrone-VID** — `VisDrone2019-VID-train/`, `VisDrone2019-VID-val/`
- **UA-DETRAC** — `ua-detrac/DETRAC-Images/`
- **Helmet Detection** — `helmet detection/` (VOC XML → YOLO conversion automated)

## AI Advisory Engine

The advisory module converts analytics into actionable recommendations:

```
Traffic Status: HIGH CONGESTION

Reason:
• 78 vehicles detected
• Average speed below 12 km/h
• Traffic density increased by 42%

Recommendation:
Extend green signal timing by 20 seconds and redirect traffic through alternate corridor.

Priority: CRITICAL
Confidence: 92%
```

## Export Formats

Reports saved to `reports/`:

- `traffic_YYYYMMDD_HHMMSS.csv` — Per-frame analytics log
- `summary_YYYYMMDD_HHMMSS.json` — Session summary + advisory
- `daily_analytics_YYYYMMDD.json` — Daily rollup
- `violations_YYYYMMDD_HHMMSS.csv` — Wrong-way and overspeeding events

## Screenshots

Sample outputs saved to `outputs/screenshots/` after dashboard runs.

## License

Built for Bengaluru Gridlock Challenge — educational / demonstration purposes.
