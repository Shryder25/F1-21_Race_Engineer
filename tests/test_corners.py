import unittest
from dataclasses import replace
import test_analysis


class CornerTests(unittest.TestCase):
    def setUp(self):
        fixture = test_analysis.LapValidityTests()
        fixture.setUp()
        self.engine, self.lap, self.telemetry = fixture.engine, fixture.lap, fixture.telemetry
        self.engine.corners = [{'name': 'Test Corner', 'start': 200, 'end': 400,
                                'ref_apex': 150, 'ref_brake': 250}]

    def drive(self):
        for distance, elapsed, speed, brake in [(100, 1000, 200, 0), (190, 1900, 180, 0),
                                                (210, 2100, 160, 1), (300, 3000, 100, 1),
                                                (410, 4100, 140, 0)]:
            self.engine.update_telemetry(replace(self.telemetry, speed=speed, brake=brake),
                                         replace(self.lap, lap_distance=distance, current_lap_time_ms=elapsed))

    def test_no_measured_delta_without_recorded_reference(self):
        self.drive()
        diag = self.engine.corner_diagnostics[0]
        self.assertIsNone(diag['delta'])
        self.assertEqual(diag['reference_source'], 'track guide')

    def test_recorded_braking_and_boundary_times_used(self):
        self.engine.best_corner_data = {'Test Corner': {
            'min_speed': 110, 'brake_point': 205, 'start_time_ms': 1000, 'end_time_ms': 2800}}
        self.drive()
        diag = self.engine.corner_diagnostics[0]
        self.assertAlmostEqual(diag['delta'], 0.2)
        self.assertEqual(diag['reference_source'], 'recorded lap')
        self.assertNotIn('early', diag['advice'])

    def test_connecting_mid_corner_does_not_generate_measured_diagnostic(self):
        self.engine.update_telemetry(self.telemetry, replace(self.lap, lap_distance=300, current_lap_time_ms=3000))
        self.engine.update_telemetry(self.telemetry, replace(self.lap, lap_distance=450, current_lap_time_ms=4500))
        self.assertEqual(self.engine.corner_diagnostics, [])
