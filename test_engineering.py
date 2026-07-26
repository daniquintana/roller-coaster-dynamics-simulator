"""Automated numerical checks for the preliminary engineering model."""

from __future__ import annotations

import unittest

import numpy as np

from preliminary_engineering_assessment import (
    ScreeningRule,
    detect_events,
    rider_frame_signals,
)
from roller_coaster_dynamics import Vehicle, simulate, track_profile


class TrackAndDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.track = track_profile()
        cls.results = simulate(
            cls.track,
            Vehicle(),
            samples=900,
            max_step=0.06,
        )
        cls.signals = rider_frame_signals(cls.results)

    def test_nominal_vehicle_completes_track(self) -> None:
        self.assertAlmostEqual(
            float(self.results["q"][-1]),
            self.track.q_end,
            places=6,
        )
        self.assertTrue(np.all(self.results["velocity"] >= -1.0e-9))
        self.assertGreater(float(self.results["distance"][-1]), 390.0)

    def test_geometry_is_finite(self) -> None:
        for key in ("x", "z", "theta", "curvature", "metric"):
            self.assertTrue(np.all(np.isfinite(self.results[key])), key)
        self.assertTrue(np.all(self.results["metric"] > 0.0))

    def test_planar_lateral_component_is_zero(self) -> None:
        np.testing.assert_allclose(self.signals["lateral_g"], 0.0)
        np.testing.assert_allclose(self.signals["lateral_jerk_gps"], 0.0)

    def test_specific_force_components_are_consistent(self) -> None:
        expected = np.sqrt(
            self.signals["longitudinal_g"] ** 2
            + self.signals["lateral_g"] ** 2
            + self.signals["normal_g"] ** 2
        )
        np.testing.assert_allclose(self.signals["resultant_g"], expected)

    def test_nominal_regression_envelope(self) -> None:
        self.assertTrue(31.0 < np.max(self.results["velocity"]) < 32.5)
        self.assertTrue(4.4 < np.max(self.signals["normal_g"]) < 4.9)
        self.assertTrue(0.3 < np.min(self.signals["normal_g"]) < 0.8)


class EventDetectionTests(unittest.TestCase):
    def test_contiguous_event_duration_and_peak(self) -> None:
        time = np.arange(6.0)
        distance = 10.0 * time
        signals = {"normal_g": np.array([1.0, 2.0, 4.5, 5.0, 2.0, 1.0])}
        rule = ScreeningRule("test", "normal_g", "above", 4.0, "g")

        events = detect_events(time, distance, signals, [rule])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_time_s, 2.0)
        self.assertEqual(events[0].end_time_s, 3.0)
        self.assertEqual(events[0].duration_s, 1.0)
        self.assertEqual(events[0].peak_value, 5.0)


if __name__ == "__main__":
    unittest.main()
