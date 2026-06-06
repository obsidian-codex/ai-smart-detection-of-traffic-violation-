"""Tests for app.py CLI entry point."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app
from config.settings import DEFAULT_DEMO_SEQUENCE


class TestParseArgs(unittest.TestCase):
    def test_default_source(self):
        args = app.parse_args([])
        self.assertEqual(args.source, f"visdrone:{DEFAULT_DEMO_SEQUENCE}")
        self.assertEqual(args.max_frames, 200)
        self.assertFalse(args.export)
        self.assertFalse(args.train_helmet)
        self.assertFalse(args.dashboard)

    def test_custom_flags(self):
        args = app.parse_args(
            [
                "--source",
                "0",
                "--max-frames",
                "50",
                "--flow-x",
                "1.0",
                "--flow-y",
                "0.0",
                "--export",
                "--output-video",
                "out.mp4",
            ]
        )
        self.assertEqual(args.source, "0")
        self.assertEqual(args.max_frames, 50)
        self.assertEqual(args.flow_x, 1.0)
        self.assertEqual(args.flow_y, 0.0)
        self.assertTrue(args.export)
        self.assertEqual(args.output_video, "out.mp4")

    def test_train_helmet_flag(self):
        args = app.parse_args(["--train-helmet"])
        self.assertTrue(args.train_helmet)

    def test_dashboard_flag(self):
        args = app.parse_args(["--dashboard"])
        self.assertTrue(args.dashboard)


class TestRun(unittest.TestCase):
    @patch("training.train_helmet.train")
    def test_run_train_helmet(self, mock_train):
        args = app.parse_args(["--train-helmet"])
        result = app.run(args)
        mock_train.assert_called_once()
        self.assertIsNone(result)

    @patch("subprocess.run")
    def test_run_dashboard(self, mock_subprocess):
        args = app.parse_args(["--dashboard"])
        result = app.run(args)
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn("streamlit", call_args)
        self.assertIn("dashboard", call_args[-1])
        self.assertIsNone(result)

    @patch("app.ReportGenerator")
    @patch("app.TrafficAdvisoryEngine")
    @patch("app.TrafficPipeline")
    @patch("app.resolve_video_source")
    def test_run_inference_with_export(
        self, mock_resolve, mock_pipeline_cls, mock_advisory_cls, mock_reporter_cls
    ):
        mock_resolve.return_value = "fake_video.mp4"

        mock_analytics = MagicMock()
        mock_analytics.frame_idx = 0
        mock_analytics.total_vehicles = 5
        mock_analytics.congestion_level = "Low"
        mock_analytics.risk_level = "Low Risk"
        mock_analytics.avg_speed_kmh = 30.0
        mock_analytics.annotated_frame = None

        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.process_video.return_value = [mock_analytics]
        mock_pipeline.get_summary.return_value = {
            "total_vehicles": 5,
            "congestion_level": "Low",
            "avg_speed_kmh": 30.0,
        }
        mock_pipeline.movement.get_flow_imbalance.return_value = {"imbalanced": False}
        mock_pipeline.wrong_way_log = []
        mock_pipeline.overspeed_log = []

        from analytics.advisory_engine import Advisory

        mock_advisory = Advisory(
            status="NORMAL TRAFFIC",
            reasons=["Traffic flowing normally"],
            problems=["No critical issues detected"],
            recommendations=["Continue standard signal timing"],
            priority="Low",
            confidence=0.88,
        )
        mock_advisory_cls.return_value.generate.return_value = mock_advisory

        mock_reporter = mock_reporter_cls.return_value
        mock_reporter.export_frame_log_csv.return_value = Path("traffic.csv")
        mock_reporter.export_summary_json.return_value = Path("summary.json")
        mock_reporter.export_daily_summary.return_value = Path("daily.json")

        args = app.parse_args(["--source", "visdrone:test_seq", "--max-frames", "10", "--export"])
        result = app.run(args)

        mock_resolve.assert_called_once_with("visdrone:test_seq")
        mock_pipeline_cls.assert_called_once()
        mock_pipeline.process_video.assert_called_once()
        mock_reporter.export_frame_log_csv.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(result["summary"]["total_vehicles"], 5)

    @patch("app.TrafficAdvisoryEngine")
    @patch("app.TrafficPipeline")
    def test_run_webcam_skips_resolve(self, mock_pipeline_cls, mock_advisory_cls):
        mock_analytics = MagicMock()
        mock_analytics.frame_idx = 0
        mock_analytics.annotated_frame = None

        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.process_video.return_value = [mock_analytics]
        mock_pipeline.get_summary.return_value = {"total_vehicles": 0}
        mock_pipeline.movement.get_flow_imbalance.return_value = {}

        mock_advisory = MagicMock()
        mock_advisory.status = "NORMAL TRAFFIC"
        mock_advisory.priority = "Low"
        mock_advisory.confidence = 0.9
        mock_advisory.recommendations = []
        mock_advisory_cls.return_value.generate.return_value = mock_advisory

        with patch("app.resolve_video_source") as mock_resolve:
            args = app.parse_args(["--source", "0", "--max-frames", "1"])
            app.run(args)
            mock_resolve.assert_not_called()

        mock_pipeline.process_video.assert_called_once_with(
            "0", max_frames=1, callback=ANY
        )


if __name__ == "__main__":
    unittest.main()
