import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
import logging
import seaborn as sns
# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import shutil
import zipfile
import hashlib
import json
import threading
import re
import glob


from django.conf import settings
from django.http import JsonResponse, FileResponse, HttpResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from accounts.models import Profile
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from web3 import Web3
from pathlib import Path
from datetime import datetime


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from tensorflow.keras import regularizers


from tensorflow.keras.models import  Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, 
    Add, Multiply, Activation, LayerNormalization, Lambda, PReLU,
    GlobalAveragePooling1D, Reshape, Permute
)

import time
import random
from sklearn.model_selection import KFold


# === GLOBAL VARIABLES FOR NAS ===
input_shape = None
output_shape = None
output_activation = None
loss_function = None
metric = None

# Load ABI + Contract address
ABI_PATH = Path("web3app/build/ModelMetadata.json")
ADDRESS_PATH = Path("web3app/contract_address.txt")

# Connect to Ganache
GANACHE_URL = "http://127.0.0.1:7545"
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

def build_model_info(request):
    return render(request,'build_model/build_model.html')

def upload_view(request):
    uploaded_file = None
    preview_data = None

    # Define the upload path within MEDIA_ROOT/uploads
    upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads')
    
    # Ensure the upload directory exists
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    
    # Helper function to safely delete a file
    def safe_delete(file_path):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                messages.error(request, f"Error deleting file: {str(e)}")
    
    # Clear out all files in upload directory to ensure one file at a time workflow
    def clear_upload_dir():
        for file in os.listdir(upload_path):
            safe_delete(os.path.join(upload_path, file))
    
    # Check for an existing uploaded file (assumes one file at a time)
    existing_files = os.listdir(upload_path)
    if existing_files:
        uploaded_file = existing_files[0]

    if request.method == 'POST':
        # Handle reupload: delete the existing file if requested
        if 'reupload' in request.POST and uploaded_file:
            safe_delete(os.path.join(upload_path, uploaded_file))
            messages.success(request, "Previous file deleted. Please upload a new dataset.")
            return redirect('upload')
        
        # Handle web scraping
        if 'scrape_data' in request.POST and 'scrape_url' in request.POST:
            url = request.POST['scrape_url'].strip()
            
            if not url:
                messages.error(request, "Please enter a valid URL to scrape.")
                return redirect('upload')
                
            # Import the WebScraper from your scraper.py
            from .scraper import WebScraper
            
            try:
                # Initialize the scraper
                scraper = WebScraper()
                
                # Scrape the URL
                dataframes, success = scraper.scrape_url(url)
                
                if not success or not dataframes:
                    messages.error(request, "No tables found on the provided URL.")
                    return redirect('upload')
                
                # Clear the upload directory (remove all existing files)
                clear_upload_dir()
                
                # Create a safe filename from the URL
                domain = re.sub(r'^https?://', '', url)
                domain = re.sub(r'[^\w.-]', '_', domain)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scraped_{domain}_{timestamp}.xlsx"
                file_path = os.path.join(upload_path, filename)
                
                # Save to Excel - prioritize saving first dataframe for profiling
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    # Always save the first dataframe as "Data" for consistent profiling access
                    first_df = dataframes[0]
                    first_df.to_excel(writer, sheet_name="Data", index=False)
                    
                    # Save additional tables if available
                    for i, df in enumerate(dataframes[1:], start=1):
                        sheet_name = f"Table_{i}"
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                uploaded_file = filename
                
                # Generate preview
                preview_data = dataframes[0].head().to_html(classes="table table-striped")
                
                messages.success(request, f"Data successfully scraped from {url} and saved.")
                
            except Exception as e:
                messages.error(request, f"Error scraping data: {str(e)}")
                return redirect('upload')
        
        # Handle file upload
        elif 'dataset' in request.FILES:
            dataset = request.FILES['dataset']
            file_extension = os.path.splitext(dataset.name)[1].lower()

            # Validate file type
            if file_extension not in ['.csv', '.xlsx', '.xls']:
                messages.error(request, "Invalid file format! Please upload a CSV or Excel file.")
                return redirect('upload')

            # Clear the upload directory (remove all existing files)
            clear_upload_dir()

            # Save the new file using FileSystemStorage
            fs = FileSystemStorage(location=upload_path)
            filename = fs.save(dataset.name, dataset)
            uploaded_file = filename

            # Construct the full file path for processing
            file_path = os.path.join(upload_path, filename)
            try:
                # Load preview data based on file type
                if file_extension == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                preview_data = df.head().to_html(classes="table table-striped")
            except Exception as e:
                messages.error(request, f"Error processing file: {str(e)}")
                return redirect('upload')

            messages.success(request, f"File '{dataset.name}' uploaded successfully!")

    return render(request, 'build_model/upload.html', {
        'uploaded_file': uploaded_file,
        'preview_data': preview_data
    })

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    return outliers.sum()

def ensure_directory_exists(directory):
    """Ensure a directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def profiling_view(request):
    # Define paths for uploads and plot outputs.
    upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads')
    plot_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
    ensure_directory_exists(plot_dir)
    
    # Check if upload directory exists and contains a file.
    if not os.path.exists(upload_path):
        messages.error(request, "No dataset found. Please upload a file first.")
        return redirect('upload')
    
    existing_files = os.listdir(upload_path)
    if not existing_files:
        messages.error(request, "No dataset found. Please upload a file first.")
        return redirect('upload')
    
    file_path = os.path.join(upload_path, existing_files[0])
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            # For Excel files, explicitly load the first sheet or the "Data" sheet if it exists
            try:
                # Try to load specific "Data" sheet first (used for scraped data)
                df = pd.read_excel(file_path, sheet_name="Data")
            except:
                # If that fails, just load the first sheet
                df = pd.read_excel(file_path)
    except Exception as e:
        messages.error(request, f"Error loading dataset: {str(e)}")
        return redirect('upload')
    
    # Ensure we have valid data to profile
    if df.empty:
        messages.error(request, "The dataset appears to be empty. Please upload a valid file.")
        return redirect('upload')
    
    # Check if we have numeric data to create visualizations
    has_numeric = any(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)))
    if not has_numeric:
        messages.warning(request, "No numeric columns found in the dataset. Some visualizations will not be available.")
    
    num_rows, num_cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    file_size = os.path.getsize(file_path) / 1024**2  # MB
    missing_series = df.isnull().sum()
    missing_values_percent = (missing_series / num_rows * 100).round(2)
    num_duplicates = df.duplicated().sum()
    
    # Collect column statistics.
    column_analysis = []
    for col in df.columns:
        col_data = df[col]
        col_info = {
            "name": col,
            "dtype": col_data.dtype,
            "unique_values": col_data.nunique(),
            "missing_percent": missing_values_percent.get(col, 0)
        }
        
        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(col_data.dtype):
            try:
                col_info.update({
                    "min": round(col_data.min(), 2) if not pd.isna(col_data.min()) else None,
                    "max": round(col_data.max(), 2) if not pd.isna(col_data.max()) else None,
                    "mean": round(col_data.mean(), 2) if not pd.isna(col_data.mean()) else None,
                    "median": round(col_data.median(), 2) if not pd.isna(col_data.median()) else None,
                    "std_dev": round(col_data.std(), 2) if not pd.isna(col_data.std()) else None,
                    "outliers": detect_outliers_iqr(col_data),
                })
            except Exception as e:
                # Handle any calculation errors gracefully
                col_info.update({
                    "error": f"Error calculating statistics: {str(e)}"
                })
        else:
            # Handle categorical/text columns
            try:
                mode_value = col_data.mode()
                value_counts = col_data.value_counts()
                col_info.update({
                    "most_common": mode_value.iloc[0] if not mode_value.empty else None,
                    "top_frequency": value_counts.iloc[0] if not value_counts.empty else None,
                })
            except Exception as e:
                col_info.update({
                    "error": f"Error calculating statistics: {str(e)}"
                })
                
        column_analysis.append(col_info)
    
    # Handle numerical columns for correlation
    num_columns = df.select_dtypes(include=['number'])
    pearson_corr = None
    if not num_columns.empty and num_columns.shape[1] > 1:
        try:
            pearson_corr = num_columns.corr().to_html(classes="table table-striped")
        except Exception as e:
            messages.warning(request, f"Error calculating correlations: {str(e)}")
    
    # Generate missing values heatmap.
    missing_heatmap_path = None
    if missing_series.sum() > 0:
        try:
            missing_heatmap_path = os.path.join('plots', 'missing_heatmap.png')
            full_path = os.path.join(plot_dir, 'missing_heatmap.png')
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
            plt.title("Missing Values Heatmap")
            plt.tight_layout()
            plt.savefig(full_path)
            plt.close('all')
        except Exception as e:
            messages.error(request, f"Error generating missing heatmap: {str(e)}")
    
    # Generate boxplot for outlier detection.
    outlier_plot_path = None
    if not num_columns.empty:
        try:
            # Only use reasonable number of columns (max 10) for the boxplot
            columns_to_plot = num_columns.columns[:10]
            if not columns_to_plot.empty:
                outlier_plot_path = os.path.join('plots', 'outlier_plot.png')
                full_path = os.path.join(plot_dir, 'outlier_plot.png')
                plt.figure(figsize=(12, 6))
                num_columns[columns_to_plot].boxplot(rot=90)
                plt.title("Outlier Detection (Box Plot)")
                plt.tight_layout()
                plt.savefig(full_path)
                plt.close('all')
        except Exception as e:
            messages.error(request, f"Error generating outlier plot: {str(e)}")

    context = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "memory_usage": round(memory_usage, 2),
        "file_size": round(file_size, 2),
        "num_duplicates": num_duplicates,
        "missing_values": missing_values_percent.to_dict(),
        "column_analysis": column_analysis,
        "pearson_corr": pearson_corr,
        "missing_heatmap_path": missing_heatmap_path,
        "outlier_plot_path": outlier_plot_path,
        "MEDIA_URL": settings.MEDIA_URL
    }
    
    return render(request, 'build_model/profiling.html', context)

def preprocessing_view(request):
    """View function for data preprocessing operations."""
    upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads')
    processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    ensure_directory_exists(processed_dir)
    
    # Check if upload directory exists and contains a file
    if not os.path.exists(upload_path):
        messages.error(request, "No dataset found. Please upload a file first.")
        return redirect('upload')
    
    existing_files = os.listdir(upload_path)
    if not existing_files:
        messages.error(request, "No dataset found. Please upload a file first.")
        return redirect('upload')
    
    file_path = os.path.join(upload_path, existing_files[0])
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        messages.error(request, f"Error loading dataset: {str(e)}")
        return redirect('upload')
    
    original_shape = df.shape
    processing_steps = []
    preview_data = df.head().to_html(classes="table table-striped")
    
    # Process the form submission
    if request.method == 'POST':
        processed_df = df.copy()
        
        # 1. Handle missing values
        missing_strategy = request.POST.get('missing_strategy', '')
        missing_columns = request.POST.getlist('missing_columns')
        
        if missing_strategy and missing_columns:
            for column in missing_columns:
                if column in processed_df.columns:
                    if missing_strategy == 'drop_rows':
                        processed_df = processed_df.dropna(subset=[column])
                        processing_steps.append(f"Dropped rows with missing values in '{column}'")
                    
                    elif missing_strategy == 'drop_column':
                        processed_df = processed_df.drop(columns=[column])
                        processing_steps.append(f"Dropped column '{column}' with missing values")
                    
                    elif missing_strategy == 'fill_mean' and pd.api.types.is_numeric_dtype(processed_df[column]):
                        mean_value = processed_df[column].mean()
                        processed_df[column] = processed_df[column].fillna(mean_value)
                        processing_steps.append(f"Filled missing values in '{column}' with mean ({mean_value:.2f})")
                    
                    elif missing_strategy == 'fill_median' and pd.api.types.is_numeric_dtype(processed_df[column]):
                        median_value = processed_df[column].median()
                        processed_df[column] = processed_df[column].fillna(median_value)
                        processing_steps.append(f"Filled missing values in '{column}' with median ({median_value:.2f})")
                    
                    elif missing_strategy == 'fill_mode':
                        mode_value = processed_df[column].mode()[0]
                        processed_df[column] = processed_df[column].fillna(mode_value)
                        processing_steps.append(f"Filled missing values in '{column}' with mode ({mode_value})")
                    
                    elif missing_strategy == 'fill_constant':
                        constant_value = request.POST.get('constant_value', '')
                        if constant_value:
                            # Try to convert to number if the column is numeric
                            if pd.api.types.is_numeric_dtype(processed_df[column]):
                                try:
                                    constant_value = float(constant_value)
                                except ValueError:
                                    pass
                            processed_df[column] = processed_df[column].fillna(constant_value)
                            processing_steps.append(f"Filled missing values in '{column}' with constant value ({constant_value})")
                    
                    elif missing_strategy == 'fill_interpolation':
                        if pd.api.types.is_numeric_dtype(processed_df[column]):
                            # Use pandas interpolation (linear by default) for numeric data
                            original_missing = processed_df[column].isna().sum()
                            processed_df[column] = processed_df[column].interpolate(method='linear').fillna(
                                method='bfill').fillna(method='ffill')
                            # Count remaining missing after interpolation
                            remaining_missing = processed_df[column].isna().sum()
                            filled_count = original_missing - remaining_missing
                            processing_steps.append(f"Filled {filled_count} missing values in '{column}' using linear interpolation")
                        else:
                            # For text data, use forward fill and backward fill
                            original_missing = processed_df[column].isna().sum()
                            processed_df[column] = processed_df[column].fillna(method='ffill').fillna(method='bfill')
                            # Count remaining missing after interpolation
                            remaining_missing = processed_df[column].isna().sum()
                            filled_count = original_missing - remaining_missing
                            processing_steps.append(f"Filled {filled_count} missing text values in '{column}' using forward/backward fill")
        
        # 2. Handle duplicates
        if 'remove_duplicates' in request.POST:
            initial_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            removed_rows = initial_rows - len(processed_df)
            if removed_rows > 0:
                processing_steps.append(f"Removed {removed_rows} duplicate rows")
        
        # 3. Handle outliers
        outlier_strategy = request.POST.get('outlier_strategy', '')
        outlier_columns = request.POST.getlist('outlier_columns')
        
        if outlier_strategy and outlier_columns:
            for column in outlier_columns:
                if column in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[column]):
                    Q1 = processed_df[column].quantile(0.25)
                    Q3 = processed_df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    if outlier_strategy == 'remove_outliers':
                        initial_rows = len(processed_df)
                        processed_df = processed_df[(processed_df[column] >= lower_bound) & 
                                                  (processed_df[column] <= upper_bound)]
                        removed_rows = initial_rows - len(processed_df)
                        if removed_rows > 0:
                            processing_steps.append(f"Removed {removed_rows} outlier rows from '{column}'")
                    
                    elif outlier_strategy == 'cap_outliers':
                        # Count outliers before capping
                        outliers_count = ((processed_df[column] < lower_bound) | 
                                         (processed_df[column] > upper_bound)).sum()
                        
                        # Cap outliers at boundaries
                        processed_df[column] = processed_df[column].clip(lower=lower_bound, upper=upper_bound)
                        
                        if outliers_count > 0:
                            processing_steps.append(f"Capped {outliers_count} outliers in '{column}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Save the processed dataframe
        if processing_steps:
            output_filename = f"processed_{os.path.splitext(existing_files[0])[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = os.path.join(processed_dir, output_filename)
            processed_df.to_csv(output_path, index=False)
            messages.success(request, f"Data preprocessing completed with {len(processing_steps)} operations. Saved as {output_filename}")
            
            # Update preview with processed data
            preview_data = processed_df.head().to_html(classes="table table-striped")
        else:
            messages.info(request, "No preprocessing operations were selected or applied.")
    
    # Prepare context for the template
    context = {
        "original_shape": original_shape,
        "preview_data": preview_data,
        "processing_steps": processing_steps if request.method == 'POST' and processing_steps else [],
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(exclude=['number']).columns),
        "missing_columns": [col for col in df.columns if df[col].isnull().any()],
        "columns_with_outliers": [col for col in df.select_dtypes(include=['number']).columns 
                                 if detect_outliers_iqr(df[col]) > 0],
    }
    
    return render(request, 'build_model/preprocessing.html', context)

# === TEXT COLUMN PROCESSING ===
def process_text_columns_preserve(df):
    """
    Convert object-type columns to categorical codes while preserving mapping.
    Returns:
      df: processed DataFrame.
      cat_mappings: dictionary with key = column name, value = {
          "mapping": {code: original_value},
          "inverse": {original_value (lowercase): code}
      }
    """
    cat_mappings = {}
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in text_columns:
        cat = df[col].astype('category')
        code_to_val = dict(enumerate(cat.cat.categories))
        inverse_mapping = {str(v).strip().lower(): k for k, v in code_to_val.items()}
        cat_mappings[col] = {"mapping": code_to_val, "inverse": inverse_mapping}
        df[col] = cat.cat.codes
    return df, cat_mappings

# === AUTOMATED NEURAL ARCHITECTURE SEARCH (NAS) ===
import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
import logging
import seaborn as sns
# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import shutil
import zipfile
import hashlib
import json
import threading
import re
import glob


from django.conf import settings
from django.http import JsonResponse, FileResponse, HttpResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from accounts.models import Profile
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from web3 import Web3
from pathlib import Path
from datetime import datetime


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from tensorflow.keras import regularizers


from tensorflow.keras.models import  Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, 
    Add, Multiply, Activation, LayerNormalization, Lambda, PReLU,
    GlobalAveragePooling1D, Reshape, Permute
)

import time
import random
from sklearn.model_selection import KFold


# === GLOBAL VARIABLES FOR NAS ===
input_shape = None
output_shape = None
output_activation = None
loss_function = None
metric = None



def build_model(hp=None, X_train=None, y_train=None, X_val=None, y_val=None,
               input_shape_param=None, output_shape_param=None, 
               output_activation_param=None, loss_function_param=None, 
               metric_param=None, problem_type=None, search_time_minutes=10, 
               search_iterations=10, verbose=1):

    # Use global variables if parameters are not provided
    global input_shape, output_shape, output_activation, loss_function, metric
    
    # Assign local variables based on parameters or globals
    actual_input_shape = input_shape_param if input_shape_param is not None else input_shape
    actual_output_shape = output_shape_param if output_shape_param is not None else output_shape
    actual_output_activation = output_activation_param if output_activation_param is not None else output_activation
    actual_loss_function = loss_function_param if loss_function_param is not None else loss_function
    actual_metric = metric_param if metric_param is not None else metric
    
    # Determine problem type if not explicitly provided
    if problem_type is None:
        if actual_output_activation in ['softmax', 'sigmoid']:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
            
    # Print configuration summary
    if verbose > 0: 
        print(f"Model Configuration:")
        print(f"- Problem Type: {problem_type}")
        print(f"- Input Shape: {actual_input_shape}")
        print(f"- Output Shape: {actual_output_shape}")
        print(f"- Output Activation: {actual_output_activation}")
        print(f"- Loss Function: {actual_loss_function}")
        print(f"- Metric: {actual_metric}")
    
    #--------------------------------
    # Architecture Components Library
    #--------------------------------
    
    def _create_advanced_dense_block(x, units, activation='relu', use_bn=True, 
                                    dropout_rate=0.2, l2_reg=1e-4, name_prefix=""):
        """Create an advanced dense block with normalization, activations and regularization"""
        # Main dense layer
        dense = Dense(units, 
                     activation=None, 
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name=f"{name_prefix}dense_{units}")(x)
        
        # Normalization (before activation is often better)
        if use_bn:
            dense = BatchNormalization(name=f"{name_prefix}bn")(dense)
        
        # Activation function
        if activation == 'prelu':
            dense = PReLU(name=f"{name_prefix}prelu")(dense)
        elif activation in ['gelu', 'swish', 'mish']:
            if activation == 'gelu':
                dense = tf.keras.activations.gelu(dense)
            elif activation == 'swish':
                dense = tf.keras.activations.swish(dense)
            else:  # mish activation
                dense = Lambda(lambda x: x * tf.math.tanh(tf.math.softplus(x)), 
                              name=f"{name_prefix}mish")(dense)
        else:
            dense = Activation(activation, name=f"{name_prefix}act")(dense)
            
        # Dropout for regularization
        if dropout_rate > 0:
            dense = Dropout(dropout_rate, name=f"{name_prefix}dropout")(dense)
            
        return dense
    
    def _create_residual_block(x, units, activation='relu', use_bn=True, 
                              dropout_rate=0.2, l2_reg=1e-4, name_prefix=""):
        """Create a residual block with skip connection"""
        # Store input for the skip connection
        input_tensor = x
        
        # First dense block
        x = _create_advanced_dense_block(
            x, units, activation, use_bn, dropout_rate, l2_reg,
            name_prefix=f"{name_prefix}res1_"
        )
        
        # Second dense block
        x = _create_advanced_dense_block(
            x, units, activation, use_bn, dropout_rate, l2_reg,
            name_prefix=f"{name_prefix}res2_"
        )
        
        # Skip connection (with projection if needed)
        if input_tensor.shape[-1] != units:
            input_tensor = Dense(units, activation='linear', 
                                kernel_regularizer=regularizers.l2(l2_reg),
                                name=f"{name_prefix}proj")(input_tensor)
        
        # Add skip connection
        x = Add(name=f"{name_prefix}add")([x, input_tensor])
        
        return x
        
    def _create_wide_deep_block(x, units_wide, units_deep, activation='relu', 
                               use_bn=True, dropout_rate=0.2, l2_reg=1e-4,
                               name_prefix=""):
        """Create a Wide & Deep learning block"""
        # Wide path (minimal transformation)
        wide_path = Dense(units_wide, activation='linear',
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f"{name_prefix}wide")(x)
        
        # Deep path (multiple nonlinear transformations)
        deep_path = _create_advanced_dense_block(
            x, units_deep, activation, use_bn, dropout_rate, l2_reg,
            name_prefix=f"{name_prefix}deep1_"
        )
        
        deep_path = _create_advanced_dense_block(
            deep_path, units_deep, activation, use_bn, dropout_rate, l2_reg,
            name_prefix=f"{name_prefix}deep2_"
        )
        
        # Combine paths
        combined = Concatenate(name=f"{name_prefix}concat")([wide_path, deep_path])
        
        # Final integration layer
        output = Dense(units_deep, activation=activation,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=f"{name_prefix}integrate")(combined)
        
        return output
    
    def _create_self_attention_block(x, units, name_prefix=""):
        """Create a self-attention mechanism for tabular data"""
        # Get input dimension
        input_dim = x.shape[-1]
        
        # Reshape for attention mechanism
        attention_input = Reshape((input_dim, 1), name=f"{name_prefix}reshape_in")(x)
        
        # Self-attention mechanism
        # Query, Key, Value projections
        query = Dense(units // 4, activation='linear', name=f"{name_prefix}query")(attention_input)
        key = Dense(units // 4, activation='linear', name=f"{name_prefix}key")(attention_input)
        value = Dense(units, activation='linear', name=f"{name_prefix}value")(attention_input)
        
        # Scaled dot-product attention
        scores = Lambda(lambda x: tf.matmul(x[0], tf.transpose(x[1], [0, 2, 1])) / 
                                 tf.sqrt(tf.cast(units // 4, tf.float32)),
                       name=f"{name_prefix}scores")([query, key])
        
        attention_weights = Activation('softmax', name=f"{name_prefix}attention_weights")(scores)
        attention_output = Lambda(lambda x: tf.matmul(x[0], x[1]), 
                                 name=f"{name_prefix}apply_attention")([attention_weights, value])
        
        # Global pooling to get back to flat representation
        attention_output = GlobalAveragePooling1D(name=f"{name_prefix}pool")(attention_output)
        
        # Skip connection
        output = Add(name=f"{name_prefix}skip")([x, attention_output])
        
        return output
    
    def _create_feature_interaction_block(x, units, activation='relu', 
                                         l2_reg=1e-4, name_prefix=""):
        """Create a block that models feature interactions (inspired by TabNet)"""
        # Feature transformation
        transformed = Dense(units, activation=activation, 
                           kernel_regularizer=regularizers.l2(l2_reg),
                           name=f"{name_prefix}transform")(x)
        
        # Feature gating - learn which features to focus on
        gate = Dense(units, activation='sigmoid', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name=f"{name_prefix}gate")(x)
        
        # Apply gating mechanism
        gated_features = Multiply(name=f"{name_prefix}gated")([transformed, gate])
        
        # Feature interaction - learn patterns across features
        output = Dense(units, activation=activation,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=f"{name_prefix}interaction")(gated_features)
        
        return output
    
    #--------------------------------
    # Architecture Definition
    #--------------------------------
    
    # For Keras Tuner mode
    if hp is not None:
        # Define core hyperparameters
        architecture_type = hp.Choice('architecture_type', [
            'standard', 'residual', 'wide_deep', 'attention', 'feature_interaction', 'hybrid'
        ])
        
        num_layers = hp.Int('num_layers', min_value=2, max_value=8)
        initial_units = hp.Int('initial_units', min_value=64, max_value=512, step=64)
        reduction_factor = hp.Choice('reduction_factor', [1.0, 0.75, 0.5])
        activation_fn = hp.Choice('activation', [
            'relu', 'elu', 'selu', 'tanh', 'gelu', 'swish', 'prelu'
        ])
        
        # Regularization
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_reg = hp.Choice('l2_reg', [0.0, 1e-6, 1e-5, 1e-4, 1e-3])
        use_batch_norm = hp.Boolean('use_batch_norm', default=True)
        
        # Optimizer settings
        learning_rate = hp.Choice('learning_rate', [1e-4, 3e-4, 1e-3, 3e-3])
        optimizer_choice = hp.Choice('optimizer', ['adam', 'radam', 'sgd', 'rmsprop'])
        
        # Input layer
        inputs = Input(shape=(actual_input_shape,))
        x = inputs
        
        # Architecture-specific structure
        if architecture_type == 'standard':
            # Standard feedforward architecture
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_advanced_dense_block(
                    x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                    name_prefix=f"layer{i}_"
                )
                
        elif architecture_type == 'residual':
            # ResNet-style architecture
            for i in range(num_layers // 2):  # Each residual block has 2 dense layers
                units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_residual_block(
                    x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                    name_prefix=f"res{i}_"
                )
                
        elif architecture_type == 'wide_deep':
            # Wide & Deep architecture
            for i in range(num_layers // 2):  # Wide & Deep blocks are more complex
                wide_units = max(32, int(initial_units * 0.5 * (reduction_factor ** i)))
                deep_units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_wide_deep_block(
                    x, wide_units, deep_units, activation_fn, use_batch_norm, 
                    dropout_rate, l2_reg, name_prefix=f"wd{i}_"
                )
                
        elif architecture_type == 'attention':
            # Self-attention based architecture
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                if i % 2 == 0:  # Alternate between dense and attention
                    x = _create_advanced_dense_block(
                        x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                        name_prefix=f"dense{i}_"
                    )
                else:
                    x = _create_self_attention_block(
                        x, units, name_prefix=f"att{i}_"
                    )
                    
        elif architecture_type == 'feature_interaction':
            # Feature interaction architecture (TabNet-inspired)
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_feature_interaction_block(
                    x, units, activation_fn, l2_reg, name_prefix=f"fi{i}_"
                )
                
        else:  # hybrid architecture
            # Hybrid approach combining multiple techniques
            blocks = [
                'dense', 'residual', 'wide_deep', 'attention', 'feature_interaction'
            ]
            
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                block_type = blocks[i % len(blocks)]
                
                if block_type == 'dense':
                    x = _create_advanced_dense_block(
                        x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                        name_prefix=f"h_dense{i}_"
                    )
                elif block_type == 'residual':
                    x = _create_residual_block(
                        x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                        name_prefix=f"h_res{i}_"
                    )
                elif block_type == 'wide_deep':
                    x = _create_wide_deep_block(
                        x, units // 2, units, activation_fn, use_batch_norm, 
                        dropout_rate, l2_reg, name_prefix=f"h_wd{i}_"
                    )
                elif block_type == 'attention':
                    x = _create_self_attention_block(
                        x, units, name_prefix=f"h_att{i}_"
                    )
                else:  # feature_interaction
                    x = _create_feature_interaction_block(
                        x, units, activation_fn, l2_reg, name_prefix=f"h_fi{i}_"
                    )
        
        # Final integration layer
        x = Dense(max(32, initial_units // 4), activation=activation_fn,
                 kernel_regularizer=regularizers.l2(l2_reg),
                 name="integration")(x)
        
        if dropout_rate > 0:
            x = Dropout(dropout_rate / 2, name="final_dropout")(x)
            
        # Output layer
        outputs = Dense(actual_output_shape, activation=actual_output_activation,
                       name="output")(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Configure optimizer
        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'radam':
            # RAdam (Rectified Adam)
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-4
            )
        elif optimizer_choice == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        else:  # rmsprop
            optimizer = RMSprop(learning_rate=learning_rate)
            
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=actual_loss_function,
            metrics=[actual_metric]
        )
        
        return model
    
    # For standalone mode (NAS without Keras Tuner)
    # Only proceed if training data is provided
    if X_train is None or y_train is None:
        print("Warning: No training data provided for standalone mode. Returning basic model.")
        return _create_basic_model(
            actual_input_shape, actual_output_shape, actual_output_activation,
            actual_loss_function, actual_metric
        )
    
    # Initialize search variables
    best_model = None
    best_val_score = float('inf') if 'loss' in actual_metric.lower() else -float('inf')
    start_time = time.time()
    max_time = start_time + (search_time_minutes * 60)
    
    # Configuration variables for architecture search
    search_space = {
        'architecture_types': ['standard', 'residual', 'wide_deep', 'attention', 'feature_interaction', 'hybrid'],
        'num_layers': list(range(2, 9)),
        'initial_units': [64, 128, 256, 384, 512],
        'reduction_factors': [1.0, 0.75, 0.5],
        'activations': ['relu', 'elu', 'selu', 'gelu', 'swish', 'prelu'],
        'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'l2_regs': [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        'batch_norms': [True, False],
        'learning_rates': [1e-4, 3e-4, 1e-3, 3e-3],
        'optimizers': ['adam', 'radam', 'sgd', 'rmsprop']
    }
    
    # Get feature dimensionality for intelligent search space pruning
    feature_dim = X_train.shape[1]
    
    # Apply intelligent search space constraints based on data characteristics
    # For small feature dims, reduce complexity
    if feature_dim < 10:
        search_space['initial_units'] = [64, 128, 256]
        search_space['num_layers'] = list(range(2, 5))
    # For large feature dims, ensure sufficient capacity
    elif feature_dim > 100:
        search_space['initial_units'] = [256, 384, 512]
        search_space['num_layers'] = list(range(3, 9))
    
    # Apply intelligent constraints based on dataset size
    dataset_size = len(X_train)
    if dataset_size < 1000:
        # For small datasets, focus on simpler models with stronger regularization
        search_space['architecture_types'] = ['standard', 'residual', 'feature_interaction']
        search_space['dropout_rates'] = [0.3, 0.4, 0.5]
        search_space['l2_regs'] = [1e-5, 1e-4, 1e-3]
    elif dataset_size > 100000:
        # For large datasets, allow more complex models
        search_space['dropout_rates'] = [0.1, 0.2, 0.3]
        
    # Check available time and adjust search iterations if needed
    available_time_per_model = search_time_minutes * 60 / search_iterations
    if available_time_per_model < 30:  # If less than 30 seconds per model
        if verbose > 0:
            print(f"Warning: Time budget allows only {available_time_per_model:.1f} seconds per model.")
            print(f"Reducing search iterations from {search_iterations} to {max(3, search_iterations // 2)}")
        search_iterations = max(3, search_iterations // 2)
    
    # Create function to sample configuration intelligently
    def _sample_config():
        # Start with random selection
        config = {
            'architecture_type': random.choice(search_space['architecture_types']),
            'num_layers': random.choice(search_space['num_layers']),
            'initial_units': random.choice(search_space['initial_units']),
            'reduction_factor': random.choice(search_space['reduction_factors']),
            'activation': random.choice(search_space['activations']),
            'dropout_rate': random.choice(search_space['dropout_rates']),
            'l2_reg': random.choice(search_space['l2_regs']),
            'use_batch_norm': random.choice(search_space['batch_norms']),
            'learning_rate': random.choice(search_space['learning_rates']),
            'optimizer': random.choice(search_space['optimizers'])
        }
        
        # Apply intelligent constraints
        if config['architecture_type'] == 'attention' and feature_dim < 10:
            # Self-attention needs sufficient features to be effective
            config['architecture_type'] = 'standard'
            
        if config['architecture_type'] in ['wide_deep', 'hybrid'] and config['num_layers'] < 3:
            # These architectures need more layers to be effective
            config['num_layers'] += 2
            
        # For small datasets, ensure strong regularization
        if dataset_size < 1000:
            config['dropout_rate'] = max(config['dropout_rate'], 0.3)
            config['l2_reg'] = max(config['l2_reg'], 1e-5)
        
        # Enforce minimum diversity between iterations
        if search_counter > 1:
            # Every 3rd iteration, force a different architecture type
            if search_counter % 3 == 0:
                previous_architectures = [prev_config.get('architecture_type') for prev_config in previous_configs]
                available_types = [t for t in search_space['architecture_types'] if t not in previous_architectures[-2:]]
                if available_types:
                    config['architecture_type'] = random.choice(available_types)
                    
        return config
    
    # Utility function to create model based on configuration
    def _create_model_from_config(config):
        inputs = Input(shape=(actual_input_shape,))
        x = inputs
        
        architecture_type = config['architecture_type']
        num_layers = config['num_layers']
        initial_units = config['initial_units']
        reduction_factor = config['reduction_factor']
        activation_fn = config['activation']
        dropout_rate = config['dropout_rate']
        l2_reg = config['l2_reg']
        use_batch_norm = config['use_batch_norm']
        
        # Building blocks based on architecture type
        if architecture_type == 'standard':
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_advanced_dense_block(
                    x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                    name_prefix=f"layer{i}_"
                )
                
        elif architecture_type == 'residual':
            for i in range(num_layers // 2 + num_layers % 2):  # Ensure at least one block
                units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_residual_block(
                    x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                    name_prefix=f"res{i}_"
                )
                
        elif architecture_type == 'wide_deep':
            for i in range(max(1, num_layers // 2)):  # Ensure at least one block
                wide_units = max(32, int(initial_units * 0.5 * (reduction_factor ** i)))
                deep_units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_wide_deep_block(
                    x, wide_units, deep_units, activation_fn, use_batch_norm, 
                    dropout_rate, l2_reg, name_prefix=f"wd{i}_"
                )
                
        elif architecture_type == 'attention':
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                if i % 2 == 0:  # Alternate between dense and attention
                    x = _create_advanced_dense_block(
                        x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                        name_prefix=f"dense{i}_"
                    )
                else:
                    x = _create_self_attention_block(
                        x, units, name_prefix=f"att{i}_"
                    )
                    
        elif architecture_type == 'feature_interaction':
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                x = _create_feature_interaction_block(
                    x, units, activation_fn, l2_reg, name_prefix=f"fi{i}_"
                )
                
        else:  # hybrid
            blocks = ['dense', 'residual', 'attention', 'feature_interaction']
            
            for i in range(num_layers):
                units = max(32, int(initial_units * (reduction_factor ** i)))
                block_type = blocks[i % len(blocks)]
                
                if block_type == 'dense':
                    x = _create_advanced_dense_block(
                        x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                        name_prefix=f"h_dense{i}_"
                    )
                elif block_type == 'residual':
                    x = _create_residual_block(
                        x, units, activation_fn, use_batch_norm, dropout_rate, l2_reg,
                        name_prefix=f"h_res{i}_"
                    )
                elif block_type == 'attention':
                    x = _create_self_attention_block(
                        x, units, name_prefix=f"h_att{i}_"
                    )
                else:  # feature_interaction
                    x = _create_feature_interaction_block(
                        x, units, activation_fn, l2_reg, name_prefix=f"h_fi{i}_"
                    )
        
        # Final integration layer
        x = Dense(max(32, initial_units // 4), activation=activation_fn,
                 kernel_regularizer=regularizers.l2(l2_reg),
                 name="integration")(x)
        
        if dropout_rate > 0:
            x = Dropout(dropout_rate / 2, name="final_dropout")(x)
            
        # Output layer
        outputs = Dense(actual_output_shape, activation=actual_output_activation,
                       name="output")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Configure optimizer
        if config['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=config['learning_rate'])
        elif config['optimizer'] == 'radam':
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=config['learning_rate'],
                weight_decay=1e-4
            )
        elif config['optimizer'] == 'sgd':
            optimizer = SGD(learning_rate=config['learning_rate'], momentum=0.9)
        else:  # rmsprop
            optimizer = RMSprop(learning_rate=config['learning_rate'])
            
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=actual_loss_function,
            metrics=[actual_metric]
        )
        
        return model
    
    # Function to quickly evaluate a model
    # Function to quickly evaluate a model
    def _quick_evaluate_model(model, epochs=20, batch_size=None):
        # Determine appropriate batch size based on dataset size
        if batch_size is None:
            if dataset_size < 1000:
                batch_size = 32
            elif dataset_size < 10000:
                batch_size = 64
            else:
                batch_size = 128
                
        # Use early stopping for efficiency
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        try:
            # Train with reduced verbosity
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val) if X_val is not None else None,
                callbacks=[early_stopping],
                verbose=0 if verbose < 2 else 1
            )
            
            # Evaluate model
            if X_val is not None:
                val_loss, val_metric = model.evaluate(X_val, y_val, verbose=0)
                
                # Check for NaN or inf values that indicate training issues
                if np.isnan(val_loss) or np.isinf(val_loss):
                    # Return a very bad score to ensure this model isn't selected
                    print("Warning: Training produced NaN/inf values")
                    return float('-inf') if 'loss' not in actual_metric.lower() else float('inf')
                    
                return val_metric if 'loss' not in actual_metric.lower() else -val_loss
            else:
                # If no validation data, perform k-fold cross-validation
                k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
                cv_scores = []
                
                for train_idx, val_idx in k_fold.split(X_train):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    # Quick training on fold
                    model.fit(
                        X_fold_train, y_fold_train,
                        epochs=min(10, epochs),  # Reduced epochs for CV
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    # Evaluate on fold
                    fold_loss, fold_metric = model.evaluate(X_fold_val, y_fold_val, verbose=0)
                    
                    # Check for NaN values
                    if np.isnan(fold_loss) or np.isinf(fold_loss):
                        continue  # Skip this fold
                        
                    cv_scores.append(fold_metric if 'loss' not in actual_metric.lower() else -fold_loss)
                
                # Return mean or a bad score if all folds failed
                return np.mean(cv_scores) if cv_scores else (float('-inf') if 'loss' not in actual_metric.lower() else float('inf'))
        
        except Exception as e:
            print(f"Error during model evaluation: {str(e)}")
            return float('-inf') if 'loss' not in actual_metric.lower() else float('inf')
    
    # Begin the architecture search process
    search_counter = 0
    previous_configs = []  # Track previous configurations
    best_config = None     # Track configuration of best model

    if verbose > 0:
        print(f"Starting neural architecture search for {problem_type} problem...")
        print(f"Time budget: {search_time_minutes} minutes")
        print(f"Max iterations: {search_iterations}")
        print(f"Dataset size: {dataset_size} samples with {feature_dim} features")

    # Main search loop
    while search_counter < search_iterations and time.time() < max_time:
        search_counter += 1
        
        if search_counter == 1:
            print(f"Will train at least {min(4, search_iterations)} different model architectures")
        elif search_counter <= 4:
            print(f"Now training model architecture {search_counter}/{min(4, search_iterations)}...")
        
        if verbose > 0:
            print(f"\nArchitecture search iteration {search_counter}/{search_iterations}")
            print(f"Time remaining: {(max_time - time.time()) / 60:.1f} minutes")
        
        # Sample configuration intelligently
        config = _sample_config()
        previous_configs.append(config)  # Track this configuration
        
        if verbose > 0:
            print("Selected configuration:")
            for key, value in config.items():
                print(f"- {key}: {value}")
        
        # Create model from configuration
        try:
            model = _create_model_from_config(config)
            
            # Debug information
            if verbose > 1:
                model.summary()
            
            # Quick evaluation
            search_start = time.time()
            eval_score = _quick_evaluate_model(model, epochs=20)
            search_time = time.time() - search_start
            
            # Track best model
            # Track best model
            is_better = False
            if 'loss' in actual_metric.lower():
                is_better = eval_score < best_val_score
            else:
                is_better = eval_score > best_val_score
                
            if is_better:
                if verbose > 0:
                    print(f"New best model found! Score: {eval_score:.4f} (previous best: {best_val_score:.4f})")
                best_val_score = eval_score
                
                # Important: Save both the model itself and its weights
                best_model = model
                best_model.save_weights("/tmp/best_model_weights.h5")
                
                # Also save the config that produced this model
                best_config = config
                
                if verbose > 0:
                    print("Best model architecture so far: " + best_config['architecture_type'])
            
            if verbose > 0:
                print(f"Evaluation score: {eval_score:.4f}, Time taken: {search_time:.1f}s")
                
        except Exception as e:
            if verbose > 0:
                print(f"Error evaluating architecture: {str(e)}")
            continue
            
        # Check if time budget is about to be exceeded
        if time.time() > max_time - 60:  # Leave 1 minute for final training
            if verbose > 0:
                print("Time budget nearly exceeded, stopping search")
            break
    
    # If no model found, create a default model
    if best_model is None:
        if verbose > 0:
            print("No successful models found during search. Creating default model.")
        
        # Create a simple reliable model
        inputs = Input(shape=(actual_input_shape,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(actual_output_shape, activation=actual_output_activation)(x)
        
        best_model = Model(inputs=inputs, outputs=outputs)
        best_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=actual_loss_function,
            metrics=[actual_metric]
        )
    else:
        # Restore best model
        if os.path.exists("/tmp/best_model_weights.h5"):
            best_model.load_weights("/tmp/best_model_weights.h5")
    
    # Final training with early stopping
    if verbose > 0:
        print("\nPerforming final training on best model...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if X_val is not None else 'loss',
        patience=10,
        restore_best_weights=True,
        verbose=1 if verbose > 0 else 0
    )
    
    best_model.fit(
        X_train, y_train,
        epochs=100,  # More epochs for final training
        batch_size=64,
        validation_data=(X_val, y_val) if X_val is not None else None,
        callbacks=[early_stopping],
        verbose=1 if verbose > 0 else 0
    )
    
    # Clean up temporary files
    if os.path.exists("/tmp/best_model_weights.h5"):
        os.remove("/tmp/best_model_weights.h5")
    
    return best_model

def _create_basic_model(input_shape, output_shape, output_activation, loss_function, metric):
    """Create a basic reliable model when no architecture search is performed"""
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_shape, activation=output_activation)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_function,
        metrics=[metric]
    )
    
    return model

@login_required
def model_view(request):
    upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads')
    processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    
    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # First try to use the most recent processed file if available
    processed_files = sorted(glob.glob(os.path.join(processed_dir, "processed_*.csv")), 
                            key=os.path.getmtime, reverse=True)
    
    if processed_files:
        # Use the most recently processed file
        file_path = processed_files[0]
        print(f"Using processed dataset: {os.path.basename(file_path)}")
    elif os.listdir(upload_path):
        # Fall back to the uploaded file if no processed files exist
        file_path = os.path.join(upload_path, os.listdir(upload_path)[0])
        print(f"Using original uploaded dataset: {os.path.basename(file_path)}")
    else:
        return HttpResponse("Error: No dataset found. Please upload a file first.", status=400)

    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    except Exception as e:
        return HttpResponse(f"Error loading dataset: {str(e)}", status=400)

    original_rows = df.shape[0]

    # Use the revised text processing that preserves one column per original field
    df, cat_mappings = process_text_columns_preserve(df)

    # Minimal cleaning - we assume most cleaning was done in preprocessing_view
    duplicates_removed = df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df.interpolate(inplace=True)

    cleaning_summary = [
        f"Total Rows: {original_rows}",
        f"Duplicates Removed: {duplicates_removed}",
        f"Missing Values Interpolated: {missing_values}"
    ]

    if request.method == "POST":
        try:
            model_type = request.POST.get("model_type")
            target_column = request.POST.get("target_column")
            feature_columns = request.POST.getlist("feature_columns")
            model_title = request.POST.get("model_title")

            search_time = int(request.POST.get("search_time", "5"))  # Default 5 minutes
            search_iterations = int(request.POST.get("search_iterations", "5"))

            # Validate target and feature selection.
            if target_column not in df.columns:
                return HttpResponse("Error: Invalid target column.", status=400)
            if not set(feature_columns).issubset(df.columns):
                return HttpResponse("Error: Invalid feature selection.", status=400)

            X = df[feature_columns].values
            y = df[target_column].values

            global input_shape, output_shape, output_activation, loss_function, metric

            if model_type == "classification":
                output_shape = len(np.unique(y))
                output_activation = 'softmax'
                loss_function = "sparse_categorical_crossentropy"
                metric = "accuracy"
            else:
                output_shape = 1
                output_activation = 'linear'
                loss_function = "mse"
                metric = "mae"

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            x_scaler = StandardScaler()
            X_train = x_scaler.fit_transform(X_train)
            X_val = x_scaler.transform(X_val)

            if model_type == "regression":
                y_scaler = StandardScaler()
                y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

            input_shape = X_train.shape[1]

            # Hyperparameter Tuning using Keras Tuner.
            # Use build_model in standalone mode to try multiple architectures
            print(f"Starting architecture search with {search_iterations} iterations over {search_time} minutes...")
            best_model = build_model(
                X_train=X_train, 
                y_train=y_train, 
                X_val=X_val, 
                y_val=y_val,
                input_shape_param=input_shape, 
                output_shape_param=output_shape, 
                output_activation_param=output_activation, 
                loss_function_param=loss_function, 
                metric_param=metric, 
                problem_type=model_type,
                search_time_minutes=search_time,  
                search_iterations=search_iterations,
                verbose=1
            )

            # Evaluate model quality
            val_loss = best_model.evaluate(X_val, y_val, verbose=0)[0]
            if model_type == "regression" and (val_loss > 1000 or np.isnan(val_loss)):
                print(f"Warning: Model has high validation loss ({val_loss}). Creating simpler model...")
                # Create a simpler model as fallback
                inputs = Input(shape=(input_shape,))
                x = Dense(64, activation='relu')(inputs)
                x = BatchNormalization()(x)
                x = Dropout(0.2)(x)
                x = Dense(32, activation='relu')(x)
                x = BatchNormalization()(x)
                outputs = Dense(output_shape, activation=output_activation)(x)
                
                simpler_model = Model(inputs=inputs, outputs=outputs)
                simpler_model.compile(
                    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
                    loss=loss_function,
                    metrics=[metric]
                )
                
                # Train with more patience
                early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
                simpler_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                epochs=150, batch_size=32, callbacks=[early_stop], verbose=1)
                
                # Check if simpler model is better
                simpler_val_loss = simpler_model.evaluate(X_val, y_val, verbose=0)[0]
                if simpler_val_loss < val_loss:
                    best_model = simpler_model
                    print(f"Using simpler model. Validation loss improved: {val_loss} -> {simpler_val_loss}")

            model_path = os.path.join(model_dir, "trained_model.keras")
            best_model.save(model_path)

            scaler_path = os.path.join(model_dir, "scalers.pkl")
            metadata = {
                "model_type": model_type,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "processed_file": os.path.basename(file_path) if processed_files else None
            }
            
            with open(scaler_path, "wb") as f:
                if model_type == "regression":
                    pickle.dump((cat_mappings, x_scaler, y_scaler, metadata), f)
                else:
                    pickle.dump((cat_mappings, x_scaler, metadata), f)

            print("Saved scalers.pkl successfully!")
            
            # Store model hash and title in projects dict
            user_profile = Profile.objects.get(user=request.user)
            with open(model_path, "rb") as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()

            user_profile.projects[model_title] = {
                "hash": model_hash,
                "uploaded": False
            }
            user_profile.save()

            y_pred = best_model.predict(X_val)
            if model_type == "classification":
                y_pred_classes = np.argmax(y_pred, axis=1)
                decoded_y_pred = decode_predictions(y_pred_classes, cat_mappings, target_column)
                metrics_report = {
                    "Accuracy": round(accuracy_score(y_val, y_pred_classes), 4),
                    "Precision": round(precision_score(y_val, y_pred_classes, average='weighted'), 4),
                    "Recall": round(recall_score(y_val, y_pred_classes, average='weighted'), 4),
                    "F1-score": round(f1_score(y_val, y_pred_classes, average='weighted'), 4),
                    "Predictions": decoded_y_pred
                }
            else:
                y_pred = y_scaler.inverse_transform(y_pred).flatten()
                metrics_report = {
                    "MSE": round(mean_squared_error(y_val, y_pred), 4),
                    "MAE": round(mean_absolute_error(y_val, y_pred), 4),
                    "R² Score": round(r2_score(y_val, y_pred), 4)
                }
                
            return JsonResponse({
                "message": "Model training complete!",
                "metrics": metrics_report,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "cleaning_summary": cleaning_summary,
                "model_hash": model_hash,
                "title": model_title,
                "username": request.user.username,
                "account_id": user_profile.ganache_address,
                "timestamp": timezone.now().isoformat()
            })

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}", status=500)

    # Pass the final DataFrame's columns (after processing) to the template
    return render(request, "build_model/model.html", {"columns": list(df.columns), "cleaning_summary": cleaning_summary})

# Configure logging for errors
logging.basicConfig(filename='errors.log', level=logging.ERROR)


def decode_predictions(preds, cat_mappings, target_column):
    """ Map encoded predictions back to original labels. """
    if target_column in cat_mappings:
        mapping = cat_mappings[target_column].get("mapping", {})
        return [mapping.get(pred, pred) for pred in preds]
    return preds


def validate_input(value, feature, cat_mappings):
    """ Validate and convert input values """
    if value is None or value.strip() == "":
        raise ValueError(f"Missing value for {feature}")

    if feature in cat_mappings:
        inverse_mapping = cat_mappings[feature].get("inverse", {})
        key = value.strip().lower()

        if key not in inverse_mapping:
            raise ValueError(f"Invalid categorical value for {feature}: {value}")
        
        return inverse_mapping[key]

    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Invalid numerical value for {feature}: {value}")


from django.shortcuts import render
from django.http import JsonResponse
import os
import tensorflow as tf
import pickle
import numpy as np
from django.conf import settings

def predict_view(request):
    model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    model_path = os.path.join(model_dir, "trained_model.keras")
    scaler_path = os.path.join(model_dir, "scalers.pkl")

    # Check if model and scaler exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return render(request, "build_model/predict.html", {
            "feature_columns": [],
            "error_message": "Model not found. Please train the model first."
        })

    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return render(request, "build_model/predict.html", {"error_message": f"Error loading model: {str(e)}"})

    # Load scalers, mappings, and metadata dynamically
    try:
        with open(scaler_path, "rb") as f:
            unpacked = pickle.load(f)

        # Dynamically unpack based on content length
        if len(unpacked) == 3:  # Classification
            cat_mappings, x_scaler, metadata = unpacked
            model_type = "classification"
        elif len(unpacked) == 4:  # Regression
            cat_mappings, x_scaler, y_scaler, metadata = unpacked
            model_type = "regression"
        else:
            return render(request, "build_model/predict.html", {"error_message": "Unexpected format in scalers.pkl"})

    except Exception as e:
        return render(request, "build_model/predict.html", {"error_message": f"Error loading scalers: {str(e)}"})

    feature_columns = metadata.get("feature_columns", [])
    target_column = metadata.get("target_column", None)

    prediction_result = None
    error_message = None

    if request.method == "POST":
        try:
            # Extract input features
            input_data = []
            for feature in feature_columns:
                value = request.POST.get(feature)
                if value is None or value.strip() == "":
                    error_message = f"Missing value for {feature}"
                    break

                # Handle categorical and numerical features
                if feature in cat_mappings:
                    inverse_mapping = cat_mappings[feature].get("inverse", {})
                    key = value.strip().lower()
                    if key in inverse_mapping:
                        numeric_value = inverse_mapping[key]
                    else:
                        error_message = f"Invalid categorical value for {feature}: {value}"
                        break
                else:
                    try:
                        numeric_value = float(value)
                    except ValueError:
                        error_message = f"Invalid numerical value for {feature}: {value}"
                        break
                
                input_data.append(numeric_value)

            # Check the input feature count
            if len(input_data) != len(feature_columns):
                error_message = "Input feature count does not match expected."
                return render(request, "build_model/predict.html", {
                    "feature_columns": feature_columns,
                    "error_message": error_message
                })

            # Prepare input data for prediction
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = x_scaler.transform(input_array)
            prediction = model.predict(input_scaled)

            # Handle classification and regression separately
            if model_type == "classification":
                predicted_code = int(np.argmax(prediction, axis=1)[0])
                predicted_label = cat_mappings[target_column]["mapping"].get(predicted_code, predicted_code)
                prediction_result = {
                    "model_type": "Classification",
                    "prediction": predicted_label
                }
            else:  # Regression
                predicted_value = y_scaler.inverse_transform(prediction).flatten()[0]
                prediction_result = {
                    "model_type": "Regression",
                    "prediction": round(float(predicted_value), 4)
                }

        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"

    return render(request, "build_model/predict.html", {
        "feature_columns": feature_columns,
        "prediction_result": prediction_result,
        "error_message": error_message
    })



# Global variables to track progress and process output
_current_progress = {"percent": 0, "message": "Initializing..."}
_build_log = ["Starting build process..."]
_build_complete = False
_download_url = ""
_build_error = None

def download_view(request):
    global _current_progress, _build_log, _build_complete, _download_url, _build_error
    
    # Define the directory and file paths.
    model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    scaler_path = os.path.join(model_dir, 'scalers.pkl')
    model_path = os.path.join(model_dir, 'trained_model.keras')

    # Progress check endpoint - return current progress
    if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest' and request.POST.get('action') == 'check_progress':
        return JsonResponse({
            "progress": _current_progress,
            "log": "\n".join(_build_log[-20:]),  # Return last 20 log lines
            "complete": _build_complete,
            "download_url": _download_url if _build_complete else "",
            "error": str(_build_error) if _build_error else None
        })

    # Verify model and scaler existence.
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return JsonResponse({"error": "Model or scaler not found"}, status=400)

    # Load the scaler and metadata information.
    try:
        with open(scaler_path, 'rb') as f:
            saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            cat_mappings = saved_data.get("cat_mappings", {})
            metadata = saved_data.get("metadata", {})
            model_type = metadata.get("model_type", "")
            feature_columns = metadata.get("feature_columns", [])
            target_column = metadata.get("target_column", "")
        else:
            if len(saved_data) == 3:  # Classification case.
                cat_mappings, x_scaler, metadata = saved_data
                model_type = "classification"
                target_column = metadata.get("target_column", "")
            elif len(saved_data) == 4:  # Regression case.
                cat_mappings, x_scaler, y_scaler, metadata = saved_data
                model_type = "regression"
                target_column = metadata.get("target_column", "")
            else:
                return JsonResponse({"error": "Unexpected format in scalers.pkl"}, status=500)
            feature_columns = metadata.get("feature_columns", [])
    except Exception as e:
        return JsonResponse({"error": f"Failed to load scalers: {str(e)}"}, status=500)

    # For GET requests, render the download page.
    if request.method == "GET":
        # Reset global state on page load
        _current_progress = {"percent": 0, "message": "Initializing..."}
        _build_log = ["Waiting for build to start..."]
        _build_complete = False
        _download_url = ""
        _build_error = None
        
        return render(request, 'build_model/download.html', {
            'model_type': model_type,
            'feature_columns': feature_columns
        })

    # POST: Generate the GUI script, bundle it, and offer for download.
    if request.method == "POST":
        # Reset build state
        _current_progress = {"percent": 0, "message": "Initializing..."}
        _build_log = ["Starting build process..."]
        _build_complete = False
        _download_url = ""
        _build_error = None
        
        bundle_param = request.POST.get('bundle', 'true').lower()
        bundle = bundle_param == 'true'
        
        # Start the build process in a separate thread
        build_thread = threading.Thread(
            target=build_application,
            args=(bundle, model_dir, model_path, scaler_path, feature_columns, model_type, target_column)
        )
        build_thread.daemon = True
        build_thread.start()
        
        # Return immediately with an acknowledgment
        return JsonResponse({
            "status": "started",
            "message": "Build process started in background"
        })

    return JsonResponse({"error": "Unsupported request method"}, status=405)

def update_progress(percent, message):
    """Update the current progress"""
    global _current_progress
    _current_progress = {"percent": percent, "message": message}
    _build_log.append(f"{percent}% - {message}")
    print(f"Progress updated: {percent}% - {message}")  # Debug output

def log_message(message):
    """Add a message to the build log"""
    _build_log.append(message)
    print(f"Log: {message}")  # Debug output

def build_application(bundle, model_dir, model_path, scaler_path, feature_columns, model_type, target_column):
    """Background thread function to build the application"""
    global _build_complete, _download_url, _build_error
    
    try:
        # Generate the GUI script
        update_progress(5, "Generating GUI script")
        gui_script = generate_gui_script(feature_columns, model_type, target_column)
        
        # Write the generated GUI script to a file
        gui_script_path = os.path.join(model_dir, "prediction_app.py")
        try:
            with open(gui_script_path, "w", encoding="utf-8") as f:
                f.write(gui_script)
            update_progress(15, "Saved GUI script to file")
        except Exception as e:
            _build_error = f"Failed to write GUI script: {str(e)}"
            return
        
        # Create a staging directory for the build
        update_progress(20, "Creating staging directory")
        staging_dir = os.path.join(model_dir, "staging")
        os.makedirs(staging_dir, exist_ok=True)
        
        if bundle:
            try:
                # Copy files to staging directory
                update_progress(25, "Copying model file")
                shutil.copy2(model_path, os.path.join(staging_dir, "trained_model.keras"))
                
                update_progress(30, "Copying scaler file")
                shutil.copy2(scaler_path, os.path.join(staging_dir, "scalers.pkl"))
                
                update_progress(35, "Copying GUI script")
                shutil.copy2(gui_script_path, os.path.join(staging_dir, "prediction_app.py"))
                
                # Set up PyInstaller command
                pyinstaller_cmd = [
                    sys.executable, "-m", "PyInstaller",
                    "--onefile", "--windowed",
                    "--add-data", f"{os.path.join(staging_dir, 'trained_model.keras')};.",
                    "--add-data", f"{os.path.join(staging_dir, 'scalers.pkl')};.",
                    "--name", "prediction_app",
                    "--distpath", model_dir,
                    os.path.join(staging_dir, "prediction_app.py")
                ]
                
                # Launch PyInstaller
                update_progress(40, "Starting PyInstaller")
                log_message("Executing PyInstaller command...")
                
                # Create a subprocess to run PyInstaller
                process = subprocess.Popen(
                    pyinstaller_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                # Read and process output line by line
                update_progress(45, "Building executable")
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        log_message(line.strip())
                        # Update progress based on PyInstaller output
                        if "Analyzing" in line:
                            update_progress(50, "Analyzing dependencies")
                        elif "Processing" in line:
                            update_progress(60, "Processing modules")
                        elif "Building EXE" in line:
                            update_progress(75, "Building executable")
                        elif "Building COLLECT" in line:
                            update_progress(85, "Collecting files")
                        elif "Writing" in line and ".exe" in line:
                            update_progress(95, "Writing executable")
                
                # Check if process completed successfully
                return_code = process.poll()
                if return_code != 0:
                    _build_error = f"PyInstaller failed with return code: {return_code}"
                    update_progress(-1, "Build failed")
                    return
                
                # Verify the executable was created
                exe_path = os.path.join(model_dir, "prediction_app.exe")
                if os.path.exists(exe_path):
                    update_progress(100, "Build completed successfully")
                    _download_url = f"/media/models/prediction_app.exe"
                    _build_complete = True
                    log_message("Executable created successfully")
                else:
                    _build_error = "Executable was not generated"
                    update_progress(-1, "Build failed")
                    log_message("ERROR: Executable file was not created")
            
            except Exception as e:
                _build_error = f"Bundling failed: {str(e)}"
                update_progress(-1, f"Build failed: {str(e)}")
                log_message(f"ERROR: {str(e)}")
        
        else:
            # Create a ZIP file with the Python script and model files
            try:
                update_progress(40, "Creating ZIP archive")
                zip_path = os.path.join(model_dir, "prediction_app.zip")
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    log_message("Adding Python script to ZIP")
                    zipf.write(gui_script_path, arcname="prediction_app.py")
                    update_progress(60, "Added Python script to ZIP")
                    
                    log_message("Adding model file to ZIP")
                    zipf.write(model_path, arcname="trained_model.keras")
                    update_progress(80, "Added model file to ZIP")
                    
                    log_message("Adding scaler file to ZIP")
                    zipf.write(scaler_path, arcname="scalers.pkl")
                    update_progress(95, "Added scaler file to ZIP")
                
                update_progress(100, "ZIP file created successfully")
                _download_url = f"/media/models/prediction_app.zip"
                _build_complete = True
                log_message("ZIP archive created successfully")
            
            except Exception as e:
                _build_error = f"Zipping failed: {str(e)}"
                update_progress(-1, f"Build failed: {str(e)}")
                log_message(f"ERROR: {str(e)}")
    
    except Exception as e:
        _build_error = f"Build process failed: {str(e)}"
        update_progress(-1, f"Build failed: {str(e)}")
        log_message(f"ERROR: {str(e)}")

# Function to generate the GUI script without importing tkinter
def generate_gui_script(feature_columns, model_type, target_column):
    """Generate the GUI script as a string without importing tkinter"""
    # Your existing GUI script generation code here
    gui_script = f'''import os
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
    messagebox.showerror("Error", f"Error loading model: {{e}}")
    sys.exit(1)

# Load the saved metadata and scalers.
try:
    with open(scaler_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        cat_mappings = data.get("cat_mappings", {{}})
        metadata = data.get("metadata", {{}})
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
    messagebox.showerror("Error", f"Failed to load metadata/scalers: {{e}}")
    sys.exit(1)

if not feature_columns:
    messagebox.showerror("Error", "No feature columns defined in metadata.")
    sys.exit(1)

# Build the GUI application.
root = tk.Tk()
root.title("Desktop Model Predictor")

# Create a dictionary to hold Entry widgets for each feature.
entries = {{}}
for feature in feature_columns:
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=5)
    label = tk.Label(frame, text=f"{{feature}}:")
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
                mapping = cat_mappings[feature].get("inverse", {{}})
                code = mapping.get(raw_value.lower())
                if code is None:
                    raise ValueError(f"Invalid input for '{{feature}}': '{{raw_value}}'")
                input_values.append(code)
            else:
                try:
                    input_values.append(float(raw_value))
                except ValueError:
                    raise ValueError(f"Expected numerical input for '{{feature}}', got '{{raw_value}}'")
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = x_scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        if model_type == "classification":
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            if target_column in cat_mappings:
                predicted_text = cat_mappings[target_column].get("mapping", {{}}).get(predicted_class, str(predicted_class))
                result = f"Predicted Class: {{predicted_text}}"
            else:
                result = f"Predicted Class: {{predicted_class}}"
        else:
            if y_scaler:
                prediction_rescaled = y_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                result = f"Predicted Value: {{prediction_rescaled[0]:.4f}}"
            else:
                result = f"Predicted Value: {{prediction[0][0]:.4f}}"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Prediction Error", f"{{e}}")

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=20)

root.mainloop()
'''
    return gui_script


with open(ABI_PATH) as f:
    contract_data = json.load(f)
    abi = contract_data["abi"]

with open(ADDRESS_PATH) as f:
    contract_address = f.read().strip()


@csrf_exempt
def upload_to_blockchain(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required."}, status=400)

    try:
        data = json.loads(request.body)
        hash_id = data["hash"]
        title = data["title"]
        username = data["username"]
        account_id = data["account_id"]

        # Get user private key from DB (example below; update for your actual model)
        from accounts.models import Profile
        profile = Profile.objects.get(user__username=username)
        private_key = profile.ganache_private_key
        sender_address = profile.ganache_address

        # Load contract
        contract = w3.eth.contract(address=contract_address, abi=abi)

        # Build transaction
        nonce = w3.eth.get_transaction_count(sender_address)
        tx = contract.functions.storeModel(
            hash_id, title, username, account_id
        ).build_transaction({
            'from': sender_address,
            'nonce': nonce,
            'gas': 2000000,
            'gasPrice': w3.to_wei('50', 'gwei')
        })

        # Sign + Send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Update the user's profile
        projects = profile.projects
        if title in projects:
            projects[title]["uploaded"] = True
            profile.projects = projects
            profile.save()


        return JsonResponse({"tx_hash": tx_hash.hex()})

    except Exception as e:
        print("Upload error:", e)
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def dashboard_view(request):
    profile = Profile.objects.get(user=request.user)
    projects = profile.projects or {}

    project_data = []
    for title, meta in projects.items():
        project_data.append({
            "title": title,
            "hash": meta.get("hash", ""),
            "uploaded": meta.get("uploaded", False),
            "username": request.user.username,
            "timestamp": meta.get("timestamp", "N/A")
        })

    return render(request, "build_model/dashboard.html", {"projects": project_data})



# Load ABI and contract address
with open("web3app/build/ModelMetadata.json") as f:
    metadata = json.load(f)
    abi = metadata["abi"]

with open("web3app/contract_address.txt") as f:
    contract_address = f.read().strip()

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
contract = w3.eth.contract(address=contract_address, abi=abi)

@csrf_exempt
def fetch_model_metadata(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=400)

    try:
        data = json.loads(request.body)
        hash_id = data.get("hash")

        result = contract.functions.getModel(hash_id).call()

        return JsonResponse({
            "title": result[0],
            "username": result[1],
            "account": result[2],
            "timestamp": result[3]
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

