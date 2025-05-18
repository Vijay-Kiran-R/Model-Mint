import os
import pickle
import sys
import shutil
import subprocess
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import messagebox
from celery import shared_task
from django.conf import settings

@shared_task
def package_tkinter_gui(model_path, scaler_path, model_type, feature_columns):
    model_dir = os.path.dirname(model_path)
    output_file = os.path.join(model_dir, "prediction_app.exe")

    # === Create GUI Script ===
    gui_script = f"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
import tensorflow as tf

# Load model and scalers
model = tf.keras.models.load_model(r"{model_path}", compile=False)
with open(r"{scaler_path}", "rb") as f:
    vectorizers, x_scaler, y_scaler = pickle.load(f)

# GUI setup
root = tk.Tk()
root.title("AI Model GUI")
root.geometry("400x500")

labels = []
entries = []

# Dynamic Input Fields
feature_columns = {feature_columns}

def predict():
    try:
        values = [float(entry.get()) for entry in entries]
        X_input = np.array([values])
        X_scaled = x_scaler.transform(X_input)
        y_pred = model.predict(X_scaled)

        if '{model_type}' == 'regression':
            y_pred = y_scaler.inverse_transform(y_pred)
            messagebox.showinfo("Prediction", f"Predicted Value: {{y_pred[0][0]}}")
        else:
            messagebox.showinfo("Classification", f"Predicted Class: {{np.argmax(y_pred)}}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

for i, col in enumerate(feature_columns):
    lbl = tk.Label(root, text=col, font=("Arial", 12))
    lbl.pack()
    entry = tk.Entry(root, font=("Arial", 12))
    entry.pack()
    labels.append(lbl)
    entries.append(entry)

btn = tk.Button(root, text="{'Classify' if model_type == 'classification' else 'Predict'}", font=("Arial", 14), command=predict)
btn.pack(pady=20)

root.mainloop()
"""

    # Save the script
    gui_script_path = os.path.join(model_dir, "prediction_app.py")
    with open(gui_script_path, "w") as f:
        f.write(gui_script)

    # Convert to Executable
    pyinstaller_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--windowed",
        "--name", "prediction_app",
        "--distpath", model_dir,
        gui_script_path
    ]

    try:
        subprocess.run(pyinstaller_cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error in packaging the app: {str(e)}"

    # Clean up
    shutil.rmtree(os.path.join(model_dir, "build"), ignore_errors=True)
    shutil.rmtree(os.path.join(model_dir, "__pycache__"), ignore_errors=True)
    os.remove(gui_script_path)
    os.remove(os.path.join(model_dir, "prediction_app.spec"))

    return output_file
