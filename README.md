# PhaseFlux

## Description
PhaseFlux is a Streamlit-based web application that predicts flow regimes in multiphase flow systems under adiabatic conditions. It utilizes a LightGBM model to provide predictions based on various input parameters related to fluid properties and flow conditions. The app also offers feature importance analysis using SHAP (SHapley Additive exPlanations) values.

## Features
- Predict flow regimes based on user-input parameters
- Visualize feature contributions using SHAP values
- Interactive interface for easy data input and result interpretation

## Installation

### Prerequisites
- Docker
- Docker Compose

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/benettia/phaseflux.git
   cd phaseflux
   ```

2. Build and run the Docker container:
   ```
   docker-compose up --build
   ```

3. Access the application by opening a web browser and navigating to:
   ```
   http://localhost:8501
   ```

## Usage
1. Enter the required input parameters in the provided fields.
2. Click the "Predict Flow Regime" button.
3. View the predicted flow regime and feature contribution analysis.

## Important Note
This application is for demonstration purposes only and should not be used for real-world applications without proper validation and testing.

## Contributing
Contributions to PhaseFlux are welcome. Please feel free to submit a Pull Request.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.