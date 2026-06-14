Student Performance Prediction
==============================

Project Overview
----------------
This project predicts student final marks from two key features:
- `Attendance`
- `Internal_Marks`

The analysis and model training are performed inside the notebook `student performance.ipynb`.

What it does
------------
- Loads `student_performance_dataset.csv`
- Prepares features and target
- Splits data into training and test sets
- Trains models including:
  - `LinearRegression`
  - `SVR`
  - `SVC`
  - `DecisionTreeClassifier`
  - `KNeighborsClassifier`
- Saves the trained regression model to `model.joblib`

Dataset
-------
The dataset file `student_performance_dataset.csv` is expected to contain at least these columns:
- `Attendance`
- `Internal_Marks`
- `Final_Marks`

Dependencies
------------
Install the required Python packages using:

```bash
python -m pip install -r requirements.txt
```

Usage
-----
1. Open `student performance.ipynb` in Jupyter Notebook or VS Code.
2. Run the notebook cells in order.
3. The final regression model is saved as `model.joblib`.

Notes
-----
- The notebook currently fits a `LinearRegression` model and saves it using `joblib`.
- Some additional classifiers are also imported and trained for experimentation, but the saved model is linear regression.
