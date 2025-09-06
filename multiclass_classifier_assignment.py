"""
Assignment No-1: Multi-Class Classifier using Deep Multilayer Perceptron
Aim: Develop multi class classifier using deep multilayer perceptron (tensorflow/pytorch) 
for any suitable dataset. Fine the parameters for better accuracy. Analyse the model accuracy 
and generate classification report. Plot accuracy and loss graph.

Author: Deep Learning Assignment
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MultiClassClassifier:
    """
    Multi-Class Classifier using Deep Multilayer Perceptron
    """
    
    def __init__(self, input_shape, num_classes):
        """
        Initialize the classifier
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load and preprocess MNIST dataset
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Reshape data for MLP (flatten images)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Convert to float32 and normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Apply StandardScaler (Z-score normalization)
        print("Applying StandardScaler (Z-score normalization)...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        print("Data preprocessing completed!")
        print(f"Training data shape after preprocessing: {X_train.shape}")
        print(f"Testing data shape after preprocessing: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, hidden_layers=[512, 256, 128], dropout_rate=0.3):
        """
        Build the deep multilayer perceptron model
        
        Args:
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
        """
        print("Building Deep Multilayer Perceptron...")
        
        # Create sequential model
        self.model = keras.Sequential()
        
        # Input layer
        self.model.add(layers.Dense(hidden_layers[0], 
                                   activation='relu', 
                                   input_shape=self.input_shape))
        self.model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers[1:], 1):
            self.model.add(layers.Dense(units, 
                                       activation='relu'))
            self.model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        self.model.add(layers.Dense(self.num_classes, 
                                   activation='softmax'))
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=128, validation_split=0.2):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
        """
        print("Training the model...")
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Learning rate reduction
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test, y_test: Test data
        """
        print("Evaluating model performance...")
        
        # Predict on test data
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return accuracy, y_pred, cm
    
    def plot_training_history(self):
        """
        Plot training history (accuracy and loss)
        """
        if self.history is None:
            print("No training history available!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, filepath='multiclass_classifier_model.h5'):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def main():
    """
    Main function to run the complete pipeline
    """
    print("=" * 60)
    print("Assignment No-1: Multi-Class Classifier using Deep MLP")
    print("=" * 60)
    
    # Initialize parameters
    input_shape = (784,)  # MNIST images are 28x28 = 784 pixels
    num_classes = 10      # Digits 0-9
    
    # Create classifier instance
    classifier = MultiClassClassifier(input_shape, num_classes)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = classifier.load_data()
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    # Build model with optimized parameters
    classifier.build_model(hidden_layers=[512, 256, 128, 64], dropout_rate=0.3)
    
    # Train model
    classifier.train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)
    
    # Evaluate model
    accuracy, y_pred, cm = classifier.evaluate_model(X_test, y_test)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(cm)
    
    # Save model
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("Assignment completed successfully!")
    print("=" * 60)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("Files generated:")
    print("- training_history.png: Training accuracy and loss plots")
    print("- confusion_matrix.png: Confusion matrix visualization")
    print("- multiclass_classifier_model.h5: Saved model")
    print("=" * 60)

if __name__ == "__main__":
    main()
