import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox

# If running as a frozen app, override stdout and stderr with dummy streams.
if getattr(sys, "frozen", False):
    class DummyFile:
        def write(self, s):
            pass
        def flush(self):
            pass
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()

# Determine application directory.
if getattr(sys, "frozen", False):
    try:
        app_dir = sys._MEIPASS
    except Exception:
        app_dir = os.path.dirname(sys.executable)
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))

def normalize_path(path):
    return os.path.normpath(path)

model_path = normalize_path(os.path.join(app_dir, "trained_model.keras"))
scaler_path = normalize_path(os.path.join(app_dir, "scalers.pkl"))

# Load the trained model.
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    messagebox.showerror("Error", f"Error loading model: {e}")
    sys.exit(1)

# Load the saved metadata and scalers.
try:
    with open(scaler_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        cat_mappings = data.get("cat_mappings", {})
        metadata = data.get("metadata", {})
        model_type = metadata.get("model_type", "")
        feature_columns = metadata.get("feature_columns", [])
        target_column = metadata.get("target_column", "")
        x_scaler = data.get("x_scaler")
        y_scaler = data.get("y_scaler")  # May be None for classification.
    else:
        if len(data) == 3:
            cat_mappings, x_scaler, metadata = data
            model_type = "classification"
            target_column = metadata.get("target_column", "")
            y_scaler = None
        elif len(data) == 4:
            cat_mappings, x_scaler, y_scaler, metadata = data
            model_type = "regression"
            target_column = metadata.get("target_column", "")
        else:
            messagebox.showerror("Error", "Unexpected format in scalers.pkl")
            sys.exit(1)
        feature_columns = metadata.get("feature_columns", [])
except Exception as e:
    messagebox.showerror("Error", f"Failed to load metadata/scalers: {e}")
    sys.exit(1)

if not feature_columns:
    messagebox.showerror("Error", "No feature columns defined in metadata.")
    sys.exit(1)

# Build the GUI application.
root = tk.Tk()
root.title("Desktop Model Predictor")

# Create a dictionary to hold Entry widgets for each feature.
entries = {}
for feature in feature_columns:
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=5)
    label = tk.Label(frame, text=f"{feature}:")
    label.pack(side=tk.LEFT)
    entry = tk.Entry(frame)
    entry.pack(side=tk.RIGHT)
    entries[feature] = entry

def predict():
    try:
        input_values = []
        for feature in feature_columns:
            raw_value = entries[feature].get().strip()
            if feature in cat_mappings:
                mapping = cat_mappings[feature].get("inverse", {})
                code = mapping.get(raw_value.lower())
                if code is None:
                    raise ValueError(f"Invalid input for '{feature}': '{raw_value}'")
                input_values.append(code)
            else:
                try:
                    input_values.append(float(raw_value))
                except ValueError:
                    raise ValueError(f"Expected numerical input for '{feature}', got '{raw_value}'")
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = x_scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        if model_type == "classification":
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            if target_column in cat_mappings:
                predicted_text = cat_mappings[target_column].get("mapping", {}).get(predicted_class, str(predicted_class))
                result = f"Predicted Class: {predicted_text}"
            else:
                result = f"Predicted Class: {predicted_class}"
        else:
            if y_scaler:
                prediction_rescaled = y_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                result = f"Predicted Value: {prediction_rescaled[0]:.4f}"
            else:
                result = f"Predicted Value: {prediction[0][0]:.4f}"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Prediction Error", f"{e}")

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=20)

root.mainloop()
