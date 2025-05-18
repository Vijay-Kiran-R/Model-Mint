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