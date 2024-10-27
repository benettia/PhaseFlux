import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import src.app
from src.helpers import generate_dimensionless_features


class TestApp(unittest.TestCase):

    def setUp(self):
        self.at = AppTest.from_file("src/app.py")
        self.at.run()

    def test_page_title(self):
        self.assertEqual(self.at.title[0].value, "Flow Regime Prediction App")

    def test_input_fields(self):
        expected_fields = ["Inner Diameter (m)", "Liquid Density (kg/m³)", "Liquid Viscosity (Pa·s)",
                           "Gas Density (kg/m³)", "Gas Viscosity (Pa·s)", "Surface Tension (N/m)",
                           "Liquid Velocity (m/s)", "Gas Velocity (m/s)", "Angle (degrees)"]
        for field in expected_fields:
            self.assertTrue(
                any(field in element.label for element in self.at.number_input))

    @patch('streamlit.number_input')
    def test_create_input_field(self, mock_number_input):
        mock_number_input.return_value = 0.05
        result = src.app.create_input_field(
            "Test Label", 0.0, 0.05, "%f", "test_key")
        self.assertEqual(result, 0.05)
        mock_number_input.assert_called_once_with(
            "Test Label", min_value=0.0, value=0.05, format="%f", key="test_key")

    def test_calculate_features(self):
        data = pd.DataFrame({
            "ID": [0.05],
            "DenL": [1000],
            "VisL": [0.001],
            "DenG": [1.5],
            "VisG": [0.00001],
            "Vsl": [1.0],
            "Vsg": [10.0],
            "Ang": [0],
            "ST": [0.00001]
        })
        features = generate_dimensionless_features(data).iloc[0].to_dict()

        expected_features = {
            'Ang': 0,
            'FrL': 1.4291592529215116,
            'FrG': 0.5535109985643248,
            'X_LM_2': 7.28528804,
            'Eo': 2447985.00625000
        }
        for feature, value in features.items():
            self.assertAlmostEqual(value, expected_features[feature], places=5)

    def test_predict_flow_regime(self):
        values = [0.05, 1000, 0.001, 1.5, 0.00001, 0.00001, 1.0, 10.0, 0]
        for i, val in enumerate(values):
            self.at.number_input[i].set_value(val)

        self.at.button[0].click().run()
        self.assertTrue(
            any("Intermittent" in element.value for element in self.at.markdown))

    def test_load_model(self):
        model = src.app.load_model()
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
