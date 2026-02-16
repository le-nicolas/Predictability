# Predictability
Predictability demonstrates a core engineering idea: reliable models are useful when they can be tested, measured, and improved.

This project simulates noisy measurements of Newton's law (`F = m * a`) and estimates the unknown mass from data using least squares.

## What Improved in This Version
- Refactored the script into reusable, testable functions.
- Added CLI arguments to control simulation and estimator settings.
- Added two estimation modes:
  - Closed-form least squares
  - Numerical optimization
- Added quality metrics (`MSE`, `RMSE`, `R^2`) and error reporting.
- Added optional plot saving for reproducible outputs.
- Added tests and dependency metadata.

## Quick Start
```powershell
git clone https://github.com/le-nicolas/Predictability.git
cd Predictability
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
python Main.py --estimator numeric
```

Example with custom settings and saved plot:
```powershell
python Main.py --true-mass 6.5 --noise-std 3 --samples 80 --seed 7 --save-plot outputs/run.png --no-plot
```

See all options:
```powershell
python Main.py --help
```

## Example Output
```text
Estimator: numeric
True mass:      5.000000
Estimated mass: 5.082374
Absolute error: 0.082374
Percent error:  1.65%
MSE:            0.230911
RMSE:           0.480532
R^2:            0.998887
```

## Run Tests
```powershell
pytest -q
```

## Project Structure
```text
.
├── Main.py
├── README.md
├── requirements.txt
└── tests/
    └── test_main.py
```

## Why This Matters
The modeling cycle is practical when it is explicit:
1. Model a system (`F = m * a`).
2. Apply it to observed data.
3. Validate predictions against reality.
4. Refine based on measured error.

This repository turns that loop into executable, testable code.

