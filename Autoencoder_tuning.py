import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time

start_time = time.time()
# Change the working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def load_and_preprocess_data():
    selected_columns = ['URLLength', 'DomainLength', 'URLSimilarityIndex', 'CharContinuationRate', 
                       'TLDLegitimateProb', 'TLDLength', 'NoOfSubDomain', 'NoOfObfuscatedChar', 
                       'LetterRatioInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 
                       'NoOfOtherSpecialCharsInURL', 'IsHTTPS', 'LineOfCode', 'LargestLineLength', 
                       'HasTitle', 'DomainTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 
                       'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 
                       'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 
                       'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto', 
                       'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 
                       'NoOfEmptyRef', 'NoOfExternalRef', 'label']
    
    df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
    df = df.loc[:, selected_columns]
    
    features = df.iloc[:, :-1]
    label = df.iloc[:, -1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    return train_test_split(X_scaled, label, test_size=0.2, random_state=42)

def create_autoencoder(input_dim, architecture, activation='relu', dropout_rate=0.0, learning_rate=0.001):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = input_layer
    
    # Add encoding layers
    for units in architecture:
        encoded = Dense(units, activation=activation)(encoded)
        if dropout_rate > 0:
            encoded = Dropout(dropout_rate)(encoded)
    
    # Bottleneck layer is the last layer in architecture
    bottleneck_dim = architecture[-1]
    
    # Decoder (reverse architecture excluding bottleneck)
    decoded = encoded
    for units in reversed(architecture[:-1]):
        decoded = Dense(units, activation=activation)(decoded)
        if dropout_rate > 0:
            decoded = Dropout(dropout_rate)(decoded)
    
    # Output layer
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Create and compile model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    # Create encoder model
    encoder = Model(input_layer, encoded)
    
    return autoencoder, encoder

def print_params(params):
    """Helper function to format parameter printing"""
    return (f"Architecture: {params['architecture']}, "
            f"Activation: {params['activation']}, "
            f"Dropout: {params['dropout_rate']}, "
            f"LR: {params['learning_rate']}, "
            f"Epochs: {params['epochs']}, "
            f"Batch Size: {params['batch_size']}")

def evaluate_model(X_train, X_test, y_train, y_test, params, combination_num, total_combinations):
    input_dim = X_train.shape[1]
    
    print(f"\nCombination {combination_num}/{total_combinations}")
    print("Parameters:", print_params(params))
    
    # Create and train autoencoder
    autoencoder, encoder = create_autoencoder(
        input_dim=input_dim,
        architecture=params['architecture'],
        activation=params['activation'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate']
    )
    
    # Train autoencoder
    history = autoencoder.fit(
        X_train, X_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        shuffle=True,
        validation_split=0.2,
        verbose=0
    )
    
    # Generate encoded features
    encoded_features_train = encoder.predict(X_train)
    encoded_features_test = encoder.predict(X_test)
    
    # Train logistic regression
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(encoded_features_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(encoded_features_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print immediate results
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 80)
    
    return {
        'accuracy': accuracy,
        'history': history.history,
        'predictions': y_pred,
        'params': params
    }

def plot_results(results):
    # Sort results by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Create the accuracy histogram
    plt.figure(figsize=(12, 6))
    accuracies = [r['accuracy'] for r in sorted_results]
    
    # Calculate the minimum accuracy to adjust y-axis
    min_accuracy = min(accuracies)
    # Set the y-axis to start slightly below the minimum accuracy
    y_min = min_accuracy - 0.001
    
    plt.bar(range(len(accuracies)), accuracies)
    plt.title('Model Accuracies Across Different Configurations')
    plt.xlabel('Configuration Index')
    plt.ylabel('Accuracy')
    
    # Set y-axis limits to zoom in on the differences
    plt.ylim(y_min, 1.0)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(accuracies):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create the training history plot for the best model
    plt.figure(figsize=(12, 6))
    best_history = sorted_results[0]['history']
    
    # Plot training and validation loss
    plt.plot(best_history['loss'], label='Train Loss', linewidth=2, marker='o')
    plt.plot(best_history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
    
    plt.title(f'Training History for Best Model (Accuracy: {sorted_results[0]["accuracy"]:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print best configurations
    print("\nTop 5 Configurations:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"\n{i+1}. Accuracy: {result['accuracy']:.4f}")
        print("Parameters:", print_params(result['params']))

# Optional: Add a helper function to create a more detailed accuracy plot
def plot_detailed_accuracy_comparison(results, top_n=10):
    """
    Creates a detailed comparison of the top N configurations
    with error bars and configuration details
    """
    plt.figure(figsize=(15, 8))
    
    # Sort results by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:top_n]
    
    accuracies = [r['accuracy'] for r in sorted_results]
    positions = range(len(accuracies))
    
    # Create bars
    plt.bar(positions, accuracies)
    
    # Calculate the minimum accuracy to adjust y-axis
    min_accuracy = min(accuracies) - 0.001
    plt.ylim(min_accuracy, 1.0)
    
    # Customize the plot
    plt.title(f'Top {top_n} Model Configurations - Accuracy Comparison')
    plt.xlabel('Configuration Index')
    plt.ylabel('Accuracy')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels and configuration details
    for i, result in enumerate(sorted_results):
        accuracy = result['accuracy']
        params = result['params']
        
        # Add accuracy value on top of bar
        plt.text(i, accuracy, f'{accuracy:.4f}', 
                ha='center', va='bottom', rotation=90)
        
        # Add architecture info below x-axis
        arch_text = f"[{','.join(map(str, params['architecture']))}]"
        plt.text(i, min_accuracy - 0.0005, arch_text, 
                ha='center', va='top', rotation=90, 
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig('detailed_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Define parameter grid
    param_grid = {
        # 'architecture': [
        #     [32, 16, 8],
        #     [64, 32, 16],
        #     [128, 64, 32],
        #     [64, 32, 16, 8],
        #     [128, 64, 32, 16]
        # ],
        # 'activation': ['relu', 'tanh', 'elu'],
        # 'dropout_rate': [0.0, 0.2, 0.4],
        # 'learning_rate': [0.001, 0.01],
        # 'epochs': [50, 100],
        # 'batch_size': [32, 64]
        'architecture': [
            [128, 64, 3],
        ],
        'activation': ['relu'],
        'dropout_rate': [0.0],
        'learning_rate': [0.001],
        'epochs': [100],
        'batch_size': [32]
    }
    
    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    total_combinations = len(param_combinations)
    print(f"\nTotal number of combinations to test: {total_combinations}")
    print("=" * 80)
    
    # Evaluate all combinations
    results = []
    for i, params in enumerate(param_combinations, 1):
        result = evaluate_model(X_train, X_test, y_train, y_test, params, i, total_combinations)
        results.append(result)
    
    print("\nEvaluation complete!")
    
    # Plot and display results
    plot_results(results)

if __name__ == "__main__":
    main()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")