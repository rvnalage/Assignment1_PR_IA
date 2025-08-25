"""
Demo script for Assignment No-1: Multi-Class Classifier
This script demonstrates the complete pipeline and shows key results.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiclass_classifier import MultiClassClassifier

def main():
    """
    Demo function to showcase the multi-class classifier
    """
    print("=" * 60)
    print("DEMO: Assignment No-1 - Multi-Class Classifier")
    print("=" * 60)
    
    # Initialize parameters
    input_shape = (784,)  # MNIST images are 28x28 = 784 pixels
    num_classes = 10      # Digits 0-9
    
    print("1. Creating Multi-Class Classifier...")
    classifier = MultiClassClassifier(input_shape, num_classes)
    
    print("\n2. Loading and preprocessing MNIST dataset...")
    X_train, X_test, y_train, y_test = classifier.load_data()
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, 
        stratify=np.argmax(y_train, axis=1)
    )
    
    print("\n3. Building Deep Multilayer Perceptron...")
    classifier.build_model(hidden_layers=[512, 256, 128, 64], dropout_rate=0.3)
    
    print("\n4. Training the model (this may take a few minutes)...")
    classifier.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=128)
    
    print("\n5. Evaluating model performance...")
    accuracy, y_pred, cm = classifier.evaluate_model(X_test, y_test)
    
    print("\n6. Generating visualizations...")
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(cm)
    
    print("\n7. Saving the model...")
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("\nKey Features Demonstrated:")
    print("✅ Data preprocessing with StandardScaler (Z-score normalization)")
    print("✅ Deep multilayer perceptron architecture")
    print("✅ Model training with regularization")
    print("✅ Performance evaluation and classification report")
    print("✅ Visualization of training history and confusion matrix")
    print("✅ Model saving and deployment")
    print("=" * 60)

if __name__ == "__main__":
    main()
