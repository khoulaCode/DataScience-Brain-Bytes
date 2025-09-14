# Copilot Instructions for DataScience-Brain-Bytes

## Project Overview
This repository is a collaborative collection of data science notebooks and resources, organized by team member. The main focus is on practical exploration of NumPy, Pandas, image processing, classification, regression, and related data science topics using Python (Jupyter Notebooks).

## Directory Structure
- `Content/`: Markdown guides for core data science topics (NumPy, Pandas, preprocessing, regex).
- `Team_members/`: Subfolders for each contributor, containing their own notebooks and data files. Example: `from_adham/`, `from_ahmed/`, etc.
- Each member's folder may include:
  - Jupyter notebooks (`*.ipynb`) for experiments, tasks, and tutorials
  - Data files in a local `data/` subfolder
  - Occasional images or outputs (e.g., `.png` files)

## Key Patterns & Conventions
- **Notebook-centric workflow:** Most work is done in Jupyter notebooks. Each notebook is self-contained and may use local data files.
- **No central Python package or build system:** There is no unified Python package, requirements.txt, or environment file. Each notebook may have its own dependencies (commonly: numpy, pandas, matplotlib, scikit-learn, cv2).
- **Data loading:** Data files are typically loaded from a local `data/` folder within each member's directory. Example:
  ```python
  import pandas as pd
  df = pd.read_csv('data/mydata.csv')
  ```
- **Image processing:** Notebooks often use OpenCV (`cv2`) and NumPy for image manipulation. Example:
  ```python
  import cv2
  import numpy as np
  img = cv2.imread('data/image.png')
  ```
- **Naming conventions:** Notebooks and data files are named descriptively by task (e.g., `Binary_Classification.ipynb`, `Student_Performance_Regression.ipynb`).
- **No automated tests or CI:** There are no test scripts or CI/CD pipelines. Validation is manual via notebook execution.

## Developer Workflow
- **To run code:** Open notebooks in Jupyter or VS Code and execute cells interactively.
- **To add new work:** Create a new notebook in your member folder. Use a local `data/` subfolder for any datasets.
- **To share results:** Save outputs (plots, images) in your folder. Use markdown cells for explanations and observations.
- **To install dependencies:** Use pip in your environment, e.g.:
  ```sh
  pip install numpy pandas matplotlib scikit-learn opencv-python
  ```
  (No global requirements file; install as needed per notebook.)

## Integration Points
- **No cross-notebook imports:** Notebooks are independent; do not import code from other notebooks or folders.
- **External dependencies:** Standard Python data science libraries only. No custom modules or packages.

## Examples
- See `Team_members/from_adham/` for classification and regression tasks.
- See `Content/` for markdown-based tutorials and explanations.

## Special Notes
- If adding new dependencies, document them in a markdown cell at the top of your notebook.
- If you encounter missing data files, check the relevant `data/` subfolder or contact the notebook author.

---

For questions or unclear conventions, review the main `README.md` or ask the team for clarification.
