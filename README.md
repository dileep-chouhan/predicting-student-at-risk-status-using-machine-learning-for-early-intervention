# Predicting Student At-Risk Status using Machine Learning for Early Intervention

## Overview

This project aims to develop a predictive model for identifying students at high risk of academic failure.  By analyzing student performance data, the model aims to proactively identify at-risk students, allowing for timely intervention strategies and ultimately improving student retention rates.  The analysis involves data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation.

## Technologies Used

This project utilizes the following Python libraries:

* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning model training and evaluation.
* **Matplotlib & Seaborn:** For data visualization.


## How to Run

To run this project, you will need to have Python 3 installed.  First, clone this repository to your local machine. Then, navigate to the project directory in your terminal and install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

After installing the dependencies, you can run the main script using:

```bash
python main.py
```

## Example Output

The script will print key analysis results to the console, including details about the chosen model, its performance metrics (e.g., accuracy, precision, recall, F1-score), and any relevant statistics derived from the EDA.  Additionally, the project may generate visualization plots (e.g., showing the distribution of grades, or the performance of different models) saved as PNG files in the `output` directory.  The specific output will depend on the dataset used and the chosen models.  Check the `output` directory for any generated plots.