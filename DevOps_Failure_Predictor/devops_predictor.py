import pandas as pd
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- LOAD DATA ---------------- #

data = pd.read_csv("dataset.csv")

X = data.drop("failed", axis=1)
y = data["failed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- MODELS ---------------- #

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

trained_models = {}

for name in models:
    model = models[name]
    model.fit(X_train, y_train)
    trained_models[name] = model

# ---------------- GUI ---------------- #

root = tk.Tk()
root.title("Advanced DevOps Failure Predictor")
root.geometry("700x650")
root.configure(bg="#1e1e2f")

# Title
title = tk.Label(
    root,
    text="INTELLIGENT DEVOPS FAILURE PREDICTOR",
    font=("Arial", 18, "bold"),
    bg="#1e1e2f",
    fg="white"
)
title.pack(pady=20)

# Input Card
card = tk.Frame(root, bg="white", padx=20, pady=20)
card.pack(pady=10)

labels = [
    "Number of Changes",
    "Files Modified",
    "Test Pass Rate",
    "Previous Failures",
    "Build Time"
]

entries = []

for label in labels:
    tk.Label(card, text=label, bg="white").pack()
    entry = tk.Entry(card, width=30)
    entry.pack(pady=5)
    entries.append(entry)

# Result Label
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"),
    bg="#1e1e2f"
)
result_label.pack(pady=15)

# ---------------- FUNCTIONS ---------------- #

def predict():
    try:
        values = [[int(entry.get()) for entry in entries]]

        results = ""

        for name in trained_models:
            prediction = trained_models[name].predict(values)
            outcome = "FAIL" if prediction[0] == 1 else "SUCCESS"
            results += f"{name}: {outcome}\n"

        if "FAIL" in results:
            result_label.config(text="BUILD RISK DETECTED", fg="red")
        else:
            result_label.config(text="BUILD IS SAFE", fg="lightgreen")

        messagebox.showinfo("Model Predictions", results)

    except:
        messagebox.showerror("Error", "Please enter valid numbers")

def show_confusion():
    model = trained_models["Decision Tree"]
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def show_tree():
    model = trained_models["Decision Tree"]

    plt.figure(figsize=(12,8))
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

# Buttons
tk.Button(
    root,
    text="PREDICT",
    command=predict,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold"),
    width=20
).pack(pady=10)

tk.Button(
    root,
    text="CONFUSION MATRIX",
    command=show_confusion,
    bg="#2196F3",
    fg="white",
    width=20
).pack(pady=5)

tk.Button(
    root,
    text="SHOW DECISION TREE",
    command=show_tree,
    bg="#9C27B0",
    fg="white",
    width=20
).pack(pady=5)

root.mainloop()