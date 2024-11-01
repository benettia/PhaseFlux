# PhaseFlux

[![Linting](https://github.com/benettia/phaseflux/actions/workflows/linting.yml/badge.svg)](https://github.com/benettia/phaseflux/actions/workflows/linting.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

## Try it Now!
Visit [www.phaseflux.com](http://www.phaseflux.com) to use PhaseFlux directly in your browser - no installation required!

## Description
PhaseFlux is a Streamlit-based web application that predicts flow regimes in multiphase flow systems under adiabatic conditions. It utilizes a LightGBM model to provide predictions based on various input parameters related to fluid properties and flow conditions. The app also offers feature importance analysis using SHAP (SHapley Additive exPlanations) values.

## Important Note
This app is for demonstration purposes only and should not be used for real-world applications. The predictions should be validated against experimental data or established correlations for critical applications.

## Features
- Instant flow regime predictions based on user-input parameters
- Advanced feature contribution analysis using SHAP values

## Online Usage
1. Navigate to [www.phaseflux.com](http://www.phaseflux.com)
2. Enter your flow parameters in the input fields
3. Click "Predict Flow Regime" to get instant results
4. Explore the SHAP analysis to understand prediction factors

## Local Installation

If you prefer to run PhaseFlux locally, you have two options:

### Option 1: Using Docker (Recommended)

#### Prerequisites
- Docker
- Docker Compose

#### Steps
1. Clone the repository:
   ```
   git clone https://github.com/benettia/phaseflux.git
   cd phaseflux
   ```

2. Build and run the Docker container:
   ```
   docker-compose up --build
   ```

3. Access the application at:
   ```
   http://localhost:8501
   ```

### Option 2: Using Virtual Environment

#### Prerequisites
- Python 3.11+
- pip

#### Steps
1. Clone the repository:
   ```
   git clone https://github.com/benettia/phaseflux.git
   cd phaseflux
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

5. Access the application at:
   ```
   http://localhost:8501
   ```

## Contributing
We welcome contributions to PhaseFlux! Whether it's bug fixes, feature additions, or documentation improvements, please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## How to Cite

If you use this model in your research, please cite it as follows:

```bibtex
@misc{alma9963263908776,
  author = {Benetti, Alessandro},
  title = {Flow regime estimation in two-phase flows employing machine learning: Investigation of the most promising dimensionless feature set},
  year = {2021},
  keywords = {flow regime estimation, two-phase flow, machine learning, feature selection, algorithm optimization},
  language = {English}
}
```

