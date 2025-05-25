## 🧠 SmartPagingAlgorithm

A machine learning-powered system to predict memory-related actions based on system metrics like RAM usage, swap usage, and CPU usage. It classifies system memory behavior into actionable categories to assist with predictive resource management and diagnostics.

---

## 📌 Features

- Preprocesses raw system logs (`windows.csv`) to extract meaningful memory usage patterns.
- Trains a **Random Forest** classifier to predict memory-related action classes.
- Saves predictions alongside true labels for evaluation.
- Retro-style **Tkinter GUI** to interactively test the classifier and visualize predictions.
- "Go Wild" mode shows class-wise prediction distribution with a custom plot.

---

## 📂 Project Structure

```

PageFaultPredictiveMeasure/
│
├── app\_utils/
│   ├── memory\_action\_classifier.pkl          # Trained model
│   └── test\_data\_with\_predictions.csv        # Test set with true and predicted labels
│
├── data/
│   └── windows.csv                           # Raw input data
│
├── model\_train.py                            # Data loading, training, evaluation, and model saving
├── test\_viz.py                               # Retro GUI for testing and visualization
├── README.md                                 # You are here 📖

````

---

## 🛠️ Setup

### 🔋 Prerequisites

- Python 3.7+
- Install the required packages:

```bash
pip install pandas scikit-learn joblib matplotlib
````

---

## 🚀 Usage

### 1. Train the Model

Make sure the `data/windows.csv` file is present. Then run:

```bash
python model_train.py
```

This script will:

* Load and preprocess the CSV.
* Train a Random Forest classifier.
* Save the trained model as `app_utils/memory_action_classifier.pkl`.
* Generate a CSV with predictions in `app_utils/test_data_with_predictions.csv`.

---

### 2. Launch the GUI

Run the following to start the interactive interface:

```bash
python test_viz.py
```

You’ll see a retro-style GUI where you can:

* Load random test samples.
* Predict their class using the trained model.
* Track your accuracy.
* Visualize model predictions via bar chart using **Go Wild** mode.

---

## 🎯 Suggested Action Classes

| Class | Description              | Page Faults Delta Threshold |
| ----- | ------------------------ | --------------------------- |
| 0     | Low memory activity      | ≤ 3000                      |
| 1     | Moderate memory activity | 3000 < x ≤ 5000             |
| 2     | High memory activity     | 5000 < x ≤ 100000           |
| 3     | Critical memory usage    | > 100000                    |

These classes help map system resource metrics to actionable categories.

---

## 📈 Example Output

* ✅ Correct! Predicted: 1 | True: 1
* ❌ Incorrect. Predicted: 2 | True: 0
* 🌀 "Go Wild" generates class distribution bar charts using matplotlib.
