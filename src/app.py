"""Streamlit app for predicting flow regimes in multiphase flow systems"""

import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.helpers import (MODEL_MARKDOWN_FEATURES, REVERSE_STATES_MAPPING,
                         STATES_MAPPING, generate_dimensionless_features)

st.set_page_config(page_title="PhaseFlux", page_icon="🌊")
st.title("MultiPhase Flow Regime Predictor")


@st.cache_resource
def load_model():
    """Load the trained model"""
    with open("./src/model/optuna_lgbm.pkl", "rb") as f:
        model = pickle.load(f)
    return model


cached_model = load_model()


def create_input_field(label: str, min_value: float,
                       value: float, format_: str, key: str) -> float:
    """Create an input field for a numeric value"""
    return st.number_input(label, min_value=min_value,
                           value=value, format=format_, key=key)


st.header("Input Data")
col1, col2, col3 = st.columns(3)

input_fields: Dict[str, Any] = {}

with col1:
    input_fields["ID"] = create_input_field(
        "Inner Diameter (m)", 0.0, 0.05, "%f", "id")
    input_fields["DenL"] = create_input_field(
        "Liquid Density (kg/m³)", 0.0, 1000.0, "%f", "den_l")
    input_fields["VisL"] = create_input_field(
        "Liquid Viscosity (Pa·s)", 0.0, 1e-3, "%e", "vis_l")

with col2:
    input_fields["DenG"] = create_input_field(
        "Gas Density (kg/m³)", 0.0, 1.2, "%f", "den_g")
    input_fields["VisG"] = create_input_field(
        "Gas Viscosity (Pa·s)", 0.0, 1.8e-5, "%e", "vis_g")
    input_fields["ST"] = create_input_field(
        "Surface Tension (N/m)", 0.0, 0.072, "%f", "st")

with col3:
    input_fields["Vsl"] = create_input_field(
        "Liquid Velocity (m/s)", 0.0, 1.0, "%f", "vsl")
    input_fields["Vsg"] = create_input_field(
        "Gas Velocity (m/s)", 0.0, 0.5, "%f", "vsg")
    input_fields["Ang"] = create_input_field(
        "Angle (degrees)", -90.0, 0.0, "%f", "ang")

if st.button("Predict Flow Regime"):
    # Prepare input data
    input_data = pd.DataFrame({k: [v] for k, v in input_fields.items()})

    # Generate dimensionless features
    features = generate_dimensionless_features(input_data)

    # Make prediction
    prediction = cached_model.predict_proba(features)
    predicted_regime = REVERSE_STATES_MAPPING[np.argmax(prediction)]

    # Display results
    st.markdown("<h2 style='text-align: center;'>Predicted Flow Regime:</h2>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{predicted_regime}</h1>",
                unsafe_allow_html=True)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(cached_model)
    shap_values = explainer.shap_values(features)

    # Display SHAP values
    st.header("Feature Contributions")
    tabs = st.tabs(list(STATES_MAPPING.keys()))

    all_regime_shap = []
    for i, (regime, tab) in enumerate(zip(STATES_MAPPING.keys(), tabs)):
        with tab:
            regime_shap = pd.DataFrame({
                "Feature": MODEL_MARKDOWN_FEATURES,
                "Value": features.values[0],
                "SHAP Value": shap_values[:, :, i][0]
            }).sort_values("SHAP Value", key=abs, ascending=False)

            all_regime_shap.append(regime_shap.assign(Regime=regime))

            # Plot the SHAP values using a horizontal bar plot
            fig = px.bar(
                regime_shap,
                y="Feature",
                x="SHAP Value",
                orientation='h',
                title=f"SHAP Values for {regime}",
                color="SHAP Value",
                color_continuous_scale=["#ff4d4d", "#4d4dff"],
                height=400,
                hover_data=["Value"],
                custom_data=["Value"]
            )
            sum_of_shap = regime_shap["SHAP Value"].sum()
            fig.update_layout(
                xaxis_title=f"SHAP Value (sum: {sum_of_shap:.2f})",
                yaxis_title="Feature",
                yaxis={"autorange": "reversed"},
                coloraxis_showscale=False,
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>Feature Value: %{customdata[0]:.4f}<br>SHAP Value: %{x:.4f}"
                )
            )
            st.plotly_chart(fig, use_container_width=True)


st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a LightGBM model to predict flow regimes in multiphase flow systems
in adiabatic conditions. It takes various input parameters related to fluid properties
and flow conditions, and provides a prediction along with feature importance analysis
using SHAP values.

Please note that this app is for demonstration purposes only.
Do not use it for any real-world applications.
""")

st.sidebar.header("Feature Calculations")
st.sidebar.markdown("""
The model uses the following dimensionless features, calculated from the input parameters:

1. **Angle**: Directly used from input

2. **Liquid Froude Number**:
   $$
   Fr_L = \\frac{V_{sl}}{\\sqrt{\\frac{(\\rho_L - \\rho_G)g}{D}}}
   $$

   Where:
   - $V_{sl}$ is the superficial liquid velocity
   - $\\rho_L$ is the liquid density
   - $\\rho_G$ is the gas density
   - $g$ is the gravitational acceleration
   - $D$ is the pipe diameter

3. **Gas Froude Number**:
   $$
   Fr_G = \\frac{V_{sg}}{\\sqrt{\\frac{(\\rho_L - \\rho_G)g}{D}}}
   $$

   Where:
   - $V_{sg}$ is the superficial gas velocity

4. **Lockhart-Martinelli parameter**:
   $$
   X_{LM}^2 = \\left(\\frac{f_L}{f_G}\\right)^{0.5} \\cdot \\left(\\frac{Fr_L}{Fr_G}\\right)^2
   $$

   Where:
   - $f_L$ and $f_G$ are the Fanning friction factors for liquid and gas, respectively
   - $Fr_L$ and $Fr_G$ are the liquid and gas Froude numbers, respectively

5. **Eotvos Number**:
   $$
   Eo = \\frac{(\\rho_L - \\rho_G)gD^2}{\\sigma}
   $$

   Where:
   - $\\sigma$ is the surface tension

Note: The Fanning friction factors are calculated based on the Reynolds number for each phase.
""")
