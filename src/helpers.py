"""Utility module for feature engineering"""

import warnings

import numpy as np
import pandas as pd
from scipy.constants import g as gravity

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Dictionary for string-int labels
STATES_MAPPING = {
    "Annular": 0,
    "Dispersed Bubbly": 1,
    "Intermittent": 2,
    "Stratified Wavy": 3,
    "Stratified Slug": 4,
    "Bouncing Bubbly": 5,
}
REVERSE_STATES_MAPPING = {v: k for k, v in STATES_MAPPING.items()}

MODEL_FEATURES = [
    "Ang",
    "FrL",
    "FrG",
    "X_LM_2",
    "Eo",
]
MODEL_MARKDOWN_FEATURES = [
    "Angle",
    "Liquid Froude Number",
    "Gas Froude Number",
    "Lockhart-Martinelli parameter",
    "Eotvos Number",
]


def generate_dimensionless_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate and keep only the required dimensionless features for flow regime determination.

    Args:
        data (pd.DataFrame): Input dataframe with dimensional features.

    Returns:
        pd.DataFrame: Dataframe with only the required dimensionless features.
    """
    # Reynolds Numbers
    re_l = data["ID"] * data["DenL"] * data["Vsl"] / data["VisL"]
    re_g = data["ID"] * data["DenG"] * data["Vsg"] / data["VisG"]

    # Fanning friction factors
    fanning_l = calculate_fanning(re_l)
    fanning_g = calculate_fanning(re_g)

    # Froude Number
    den_diff = data["DenL"] - data["DenG"]
    data["FrL"] = (data["DenL"] / (den_diff * gravity * data["ID"])) ** 0.5 * data[
        "Vsl"
    ]
    data["FrG"] = (data["DenG"] / (den_diff * gravity * data["ID"])) ** 0.5 * data[
        "Vsg"
    ]

    # LM
    x_lm = ((fanning_l / fanning_g) ** 0.5) * (data["FrL"] / data["FrG"])
    data["X_LM_2"] = x_lm ** 2

    # Eötvös number
    data["Eo"] = den_diff * gravity * data["ID"] ** 2 / data["ST"]

    return data[MODEL_FEATURES]


def calculate_fanning(reynolds: pd.Series) -> pd.Series:
    """Calculate Fanning friction factor for given Reynolds number."""
    fanning = 16 / reynolds
    mask = reynolds > 2300
    fanning[mask] = 0.0625 / (
        np.log10(
            (150.39 / (reynolds[mask] ** 0.98865)) - (152.66 / reynolds[mask]))
        ** 2
    )
    return fanning
