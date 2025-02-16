# Scoliosis Chart Review - Model Runner

This project uses machine learning models to analyze scoliosis chart data. It includes a grid search for various classifiers, computes SHAP values to help interpret model predictions, and saves results to CSV files for easy review and editing.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option A: Git Clone/Pull](#option-a-git-clonepull)
  - [Option B: Download Zip](#option-b-download-zip)
- [Running the Code](#running-the-code)
- [Understanding the Outputs](#understanding-the-outputs)
- [Using SHAP Results to Edit the Training Data](#using-shap-results-to-edit-the-training-data)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

1. **Python 3.7 or Later**  
   Download and install Python from [python.org](https://www.python.org/downloads/).

2. **pip (Python Package Installer)**  
   - On Windows, pip is installed automatically with Python.  
   - On macOS/Linux, follow the instructions [here](https://pip.pypa.io/en/stable/installation/).

---

## Installation

### Option A: Git Clone/Pull

1. **Clone the Repository**  
   Open your terminal (or Command Prompt on Windows), then run:
   ```bash
   git clone https://github.com/YourUsername/SCOLIOSIS_CHART_REVIEW.git

1.  **Pull New Changes (If Updating)**\
    If you already have the repository locally, navigate to its folder in your terminal and run:

    bash

    Copy

    `git pull`

    This command fetches the latest updates from GitHub.

2.  **Open in VS Code (Optional)**

    -   Launch Visual Studio Code.
    -   Click **File** > **Open Folder**.
    -   Select the `SCOLIOSIS_CHART_REVIEW` folder you just cloned.
    -   Click **Open** to view/edit files in VS Code.

### Option B: Download Zip

1.  **Download the Project as a Zip**

    -   Go to the project's GitHub page in your web browser. (if you're reading this on a web browser you've already done this)
    -   Click the green **Code** button, then select **Download ZIP**.
    -   Extract the downloaded ZIP file to a folder of your choice.
2.  **Open in VS Code (Optional)**

    -   Launch Visual Studio Code.
    -   Click **File** > **Open Folder**.
    -   Select the folder where you extracted the ZIP.
    -   Click **Open** to view/edit files in VS Code.

* * * * *

### Install Required Packages

From inside your project folder (whether cloned or unzipped), open your terminal and run:

bash

Copy

`pip install -r requirements.txt`

This command installs all necessary packages with the correct versions.

* * * * *

Running the Code
----------------

1.  **Prepare the Data**

    -   Ensure your raw chart data is located in the `raw_chart_data` folder.
    -   The script will generate a training DataFrame CSV (`training_df_tothlos.csv`) in the `results` folder if it does not already exist.
2.  **Run the Model Script**\
    From the project folder, run:

    bash

    Copy

    `python3 run_models.py`

    The script will:

    -   Load (or generate) the training data.
    -   Run grid searches over the specified models.
    -   Compute performance metrics.
    -   Save performance and SHAP summaries as CSV files.
    -   Display a summary of the SHAP values.

* * * * *

Understanding the Outputs
-------------------------

After running the script, you will find the following CSV files in the `results` folder:

-   **training_df_tothlos.csv**: The training data used by the models.
-   **performance_tothlos_<timestamp>.csv**: A CSV file containing performance metrics (e.g., AUC, F1, etc.) for the best model.
-   **shap_summary_tothlos_<timestamp>.csv**: A CSV file with a summary of SHAP values, listing each feature and its mean absolute SHAP value, sorted by importance.

* * * * *

Using SHAP Results to Edit the Training Data
--------------------------------------------

The SHAP summary helps you understand which features have the greatest influence on the model predictions. If you notice a feature with an unexpectedly high mean absolute SHAP value, you can:

1.  Open the **shap_summary_tothlos_<timestamp>.csv** file in Excel.
2.  Review the top features by `mean_abs_shap`.
3.  If a feature appears suspicious (e.g., due to data quality issues), open the **training_df_tothlos.csv** file in Excel.
4.  Edit or remove the problematic column.
5.  Re-run the model script to see how performance changes.

* * * * *

Troubleshooting
---------------

-   **Installation Issues:**\
    Make sure you have installed the correct Python version and that pip is working correctly.

-   **Missing Data Files:**\
    Verify that the `raw_chart_data` folder exists and contains your raw data files.

-   **SHAP Plot Issues:**\
    If the SHAP plot does not display, the script will still save the SHAP summary CSV so you can review the values manually.

If you encounter any issues, please contact the project maintainer.