import tkinter as tk
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load model and test data
clf = joblib.load("app_utils/memory_action_classifier.pkl")
test_data = pd.read_csv("app_utils/test_data_with_predictions.csv")

# Setup window
root = tk.Tk()
root.title("🧠 Memory Classifier Retro UI")
root.geometry("600x650")
root.configure(bg="black")

font_style = ("Courier", 12)
title_font = ("Courier", 16, "bold")
green = "#00FF00"
cyan = "#00FFFF"

score = {'correct': 0, 'total': 0}
current_sample = None
chart_canvas = None

def load_sample():
    global current_sample
    current_sample = test_data.sample(n=1).iloc[0]
    input_text.set(f"RAM: {current_sample['RAM_Usage_MB']} MB\n"
                   f"Swap: {current_sample['Swap_Usage_MB']} MB\n"
                   f"CPU: {current_sample['CPU_Usage']}%")
    output_text.set("")

def predict():
    global score
    if current_sample is None:
        return

    features = pd.DataFrame([{
        'RAM_Usage_MB': current_sample['RAM_Usage_MB'],
        'Swap_Usage_MB': current_sample['Swap_Usage_MB'],
        'CPU_Usage': current_sample['CPU_Usage']
    }])
    prediction = clf.predict(features)[0]
    true_label = str(int(current_sample['True_Label']))

    score['total'] += 1
    if prediction == true_label:
        score['correct'] += 1
        output_text.set(f"✅ Correct! Predicted: {prediction} | True: {true_label}")
    else:
        output_text.set(f"❌ Incorrect. Predicted: {prediction} | True: {true_label}")

    score_text.set(f"Score: {score['correct']} / {score['total']}")

def reset_score():
    global chart_canvas
    score['correct'] = 0
    score['total'] = 0
    score_text.set("Score: 0 / 0")
    output_text.set("")
    input_text.set("")
    if chart_canvas:
        chart_canvas.get_tk_widget().destroy()
        chart_canvas = None

def go_wild():
    global chart_canvas

    # Clear previous chart if exists
    if chart_canvas:
        chart_canvas.get_tk_widget().destroy()

    # Run predictions on the full dataset
    X_all = test_data[['RAM_Usage_MB', 'Swap_Usage_MB', 'CPU_Usage']]
    y_true = test_data['True_Label']
    y_pred = clf.predict(X_all)

    # Count predictions per class
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    # Calculate accuracy
    correct = sum(y_true.astype(str) == y_pred.astype(str))
    total = len(y_true)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='black')
    bars = ax.bar(pred_counts.index.astype(str), pred_counts.values, color=cyan)

    ax.set_title(f"🌀 Prediction Count (Score: {correct} / {total})", color=cyan, fontsize=12, fontname='Courier')
    ax.set_xlabel("Predicted Class", color=green, fontname='Courier')
    ax.set_ylabel("Count", color=green, fontname='Courier')
    ax.tick_params(colors=green, labelsize=10)
    ax.set_facecolor('black')

    # Style ticks and spines
    for spine in ax.spines.values():
        spine.set_edgecolor(green)

    # Embed chart in tkinter
    chart_canvas = FigureCanvasTkAgg(fig, master=root)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(pady=10)

# UI Layout
tk.Label(root, text="Memory Classifier Terminal", font=title_font, fg=green, bg="black").pack(pady=10)

input_text = tk.StringVar()
output_text = tk.StringVar()
score_text = tk.StringVar(value="Score: 0 / 0")

tk.Label(root, textvariable=input_text, font=font_style, fg=green, bg="black").pack(pady=10)
tk.Label(root, textvariable=output_text, font=font_style, fg=green, bg="black").pack(pady=5)
tk.Label(root, textvariable=score_text, font=font_style, fg=green, bg="black").pack(pady=5)

frame = tk.Frame(root, bg="black")
frame.pack(pady=20)

tk.Button(frame, text="🆕 Load Sample", command=load_sample, font=font_style, bg="black", fg=green, activebackground="#222").pack(side=tk.LEFT, padx=10)
tk.Button(frame, text="🧪 Predict", command=predict, font=font_style, bg="black", fg=green, activebackground="#222").pack(side=tk.LEFT, padx=10)
tk.Button(frame, text="🔁 Reset Score", command=reset_score, font=font_style, bg="black", fg=green, activebackground="#222").pack(side=tk.LEFT, padx=10)

tk.Button(root, text="🌀 Go Wild", command=go_wild, font=font_style, bg="black", fg=cyan, activebackground="#222").pack(pady=10)

# Start GUI
root.mainloop()
