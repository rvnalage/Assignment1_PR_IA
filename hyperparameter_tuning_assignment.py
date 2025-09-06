"""
Hyperparameter Tuning for Multi-Class Classifier
This script demonstrates how to find optimal hyperparameters for better accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import itertools
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class HyperparameterTuner:
    """
    Class for hyperparameter tuning of the multi-class classifier
    """
    
    def __init__(self):
        self.results = []
        self.best_params = None
        self.best_accuracy = 0
        
    def load_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset for hyperparameter tuning...")
        
        # Load MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Reshape and normalize
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
        
        # Apply StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, hidden_layers, dropout_rate, learning_rate):
        """Build model with given hyperparameters"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], 
                              activation='relu', 
                              input_shape=(784,)))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(10, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate(self, model, X_train, y_train, X_val, y_val, epochs=20):
        """Train and evaluate a model"""
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get best validation accuracy
        best_val_acc = max(history.history['val_accuracy'])
        
        return best_val_acc, history
    
    def grid_search(self, X_train, y_train, X_val, y_val):
        """Perform grid search for hyperparameter tuning"""
        print("Starting Grid Search for Hyperparameter Tuning...")
        
        # Define hyperparameter grid
        param_grid = {
            'hidden_layers': [
                [256, 128],
                [512, 256],
                [512, 256, 128],
                [512, 256, 128, 64],
                [1024, 512, 256]
            ],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0001, 0.00001]
        }
        
        # Generate all combinations
        param_combinations = list(itertools.product(
            param_grid['hidden_layers'],
            param_grid['dropout_rate'],
            param_grid['learning_rate']
        ))
        
        print(f"Total combinations to test: {len(param_combinations)}")
        
        for i, (hidden_layers, dropout_rate, learning_rate) in enumerate(param_combinations):
            print(f"\nTesting combination {i+1}/{len(param_combinations)}")
            print(f"Hidden layers: {hidden_layers}")
            print(f"Dropout rate: {dropout_rate}")
            print(f"Learning rate: {learning_rate}")
            
            # Build and train model
            model = self.build_model(hidden_layers, dropout_rate, learning_rate)
            val_accuracy, history = self.train_and_evaluate(model, X_train, y_train, X_val, y_val)
            
            # Store results
            result = {
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'val_accuracy': val_accuracy,
                'num_layers': len(hidden_layers),
                'total_params': model.count_params()
            }
            
            self.results.append(result)
            
            # Update best parameters
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_params = result.copy()
                print(f"New best accuracy: {val_accuracy:.4f}")
            
            print(f"Validation accuracy: {val_accuracy:.4f}")
        
        print(f"\nGrid search completed!")
        print(f"Best accuracy: {self.best_accuracy:.4f}")
        print(f"Best parameters: {self.best_params}")
    
    def plot_results(self):
        """Plot hyperparameter tuning results"""
        if not self.results:
            print("No results to plot!")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy vs Dropout Rate
        sns.boxplot(data=df, x='dropout_rate', y='val_accuracy', ax=axes[0,0])
        axes[0,0].set_title('Accuracy vs Dropout Rate')
        axes[0,0].set_xlabel('Dropout Rate')
        axes[0,0].set_ylabel('Validation Accuracy')
        
        # Plot 2: Accuracy vs Learning Rate
        sns.boxplot(data=df, x='learning_rate', y='val_accuracy', ax=axes[0,1])
        axes[0,1].set_title('Accuracy vs Learning Rate')
        axes[0,1].set_xlabel('Learning Rate')
        axes[0,1].set_ylabel('Validation Accuracy')
        
        # Plot 3: Accuracy vs Number of Layers
        sns.boxplot(data=df, x='num_layers', y='val_accuracy', ax=axes[1,0])
        axes[1,0].set_title('Accuracy vs Number of Hidden Layers')
        axes[1,0].set_xlabel('Number of Hidden Layers')
        axes[1,0].set_ylabel('Validation Accuracy')
        
        # Plot 4: Accuracy vs Total Parameters
        axes[1,1].scatter(df['total_params'], df['val_accuracy'], alpha=0.6)
        axes[1,1].set_title('Accuracy vs Model Complexity')
        axes[1,1].set_xlabel('Total Parameters')
        axes[1,1].set_ylabel('Validation Accuracy')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\nHyperparameter Tuning Summary:")
        print("=" * 50)
        print(f"Best validation accuracy: {self.best_accuracy:.4f}")
        print(f"Best hidden layers: {self.best_params['hidden_layers']}")
        print(f"Best dropout rate: {self.best_params['dropout_rate']}")
        print(f"Best learning rate: {self.best_params['learning_rate']}")
        
        # Top 5 configurations
        df_sorted = df.sort_values('val_accuracy', ascending=False)
        print("\nTop 5 Configurations:")
        print(df_sorted[['hidden_layers', 'dropout_rate', 'learning_rate', 'val_accuracy']].head())
    
    def save_results(self, filename='hyperparameter_results.csv'):
        """Save results to CSV file"""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    """Main function for hyperparameter tuning"""
    print("=" * 60)
    print("Hyperparameter Tuning for Multi-Class Classifier")
    print("=" * 60)
    
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Load data
    X_train, X_test, y_train, y_test = tuner.load_data()
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    # Perform grid search
    tuner.grid_search(X_train, y_train, X_val, y_val)
    
    # Plot results
    tuner.plot_results()
    
    # Save results
    tuner.save_results()
    
    print("\n" + "=" * 60)
    print("Hyperparameter tuning completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
