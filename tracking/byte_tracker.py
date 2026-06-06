"""ByteTrack integration via Ultralytics built-in tracker.

Ultralytics YOLO provides native ByteTrack support through tracker config.
This module documents and centralizes tracker configuration.
"""
from config.settings import TRACKER_CONFIG

# ByteTrack is invoked in TrafficPipeline via:
#   model.track(frame, persist=True, tracker=TRACKER_CONFIG)
#
# Default config: bytetrack.yaml (bundled with ultralytics)
# Features: two-stage association, low-confidence recovery, ID persistence

TRACKER_NAME = "ByteTrack"
TRACKER_YAML = TRACKER_CONFIG
