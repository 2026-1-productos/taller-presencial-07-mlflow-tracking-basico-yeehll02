import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


def test_01():

    # Get the venv Python executable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")

    # Test if the homework script runs without errors
    try:
        for model in ["elasticnet", "knn"]:
            subprocess.run(
                [venv_python, "-m", "homework", "--model", model],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running the homework script: {e}")

    # Ensure the mlruns directory exists
    assert os.path.exists("mlruns"), "mlruns directory does not exist."

    # Check if there are any experiments saved in mlruns/
    experiments = [
        d for d in os.listdir("mlruns") if os.path.isdir(os.path.join("mlruns", d))
    ]
    assert len(experiments) > 0, "No experiments found in mlruns directory."

    # Check if the required file exists
    assert os.path.exists("make_predictions.py")
