# Assignment No-1: Multi-Class Classifier using Deep Multilayer Perceptron

## Aim
Develop multi-class classifier using deep multilayer perceptron (TensorFlow) for MNIST handwritten digits dataset. Fine-tune parameters for better accuracy, analyze model accuracy, generate classification report, and plot accuracy and loss graphs.

## Objectives
- Learn Deep Neural Network modeling
- Learn to develop and deploy models
- Implement data preprocessing techniques
- Optimize model hyperparameters
- Evaluate model performance comprehensively

## Theory

### Standardization (Z-score Normalization)
This is one of the most used types of scaling in data preprocessing. It redistributes the data so that:
- Mean (μ) = 0
- Standard Deviation (σ) = 1

**Formula:**
```
z = (x - μ) / σ
```

Where:
- x = original value
- μ = mean of the feature
- σ = standard deviation of the feature

### Deep Multilayer Perceptron (MLP)
A feedforward artificial neural network that maps input data to appropriate outputs. It consists of:
- Input layer
- Multiple hidden layers with activation functions
- Output layer with softmax activation for multi-class classification

## Dataset
**MNIST Handwritten Digits Dataset**
- 60,000 training images
- 10,000 test images
- 28x28 pixel grayscale images
- 10 classes (digits 0-9)

## Implementation Details

### Data Preprocessing
1. **Flattening**: Convert 28x28 images to 784-dimensional vectors
2. **Normalization**: Scale pixel values to [0, 1] range
3. **Standardization**: Apply Z-score normalization using StandardScaler
4. **One-hot Encoding**: Convert labels to categorical format

### Model Architecture
- **Input Layer**: 784 neurons (flattened image)
- **Hidden Layers**: [512, 256, 128, 64] neurons with ReLU activation
- **Dropout**: 0.3 dropout rate for regularization
- **Output Layer**: 10 neurons with softmax activation

### Training Strategy
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 20%

### Regularization Techniques
- Dropout layers to prevent overfitting
- Early stopping to halt training when validation loss stops improving
- Learning rate reduction when validation loss plateaus

## Files Structure
```
DeepLearning_Assignment1/
├── multiclass_classifier.py    # Main implementation
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── training_history.png        # Generated plots
├── confusion_matrix.png        # Generated plots
└── multiclass_classifier_model.h5  # Saved model
```

## Installation and Setup

1. **Install Python dependencies:**
```bash
pip install -r requirements_assignment.txt
```

2. **Run the classifier:**
```bash
python multiclass_classifier_assignment.py
```

## Expected Output

The program will:
1. Load and preprocess the MNIST dataset
2. Build and train the deep MLP model
3. Display model architecture and training progress
4. Generate classification report with precision, recall, and F1-score
5. Create visualization plots:
   - Training accuracy and loss curves
   - Confusion matrix heatmap
6. Save the trained model

## Model Performance
Expected accuracy: **~98%** on test set

## Key Features Implemented

### ✅ Data Preprocessing
- StandardScaler (Z-score normalization)
- Data normalization and reshaping
- One-hot encoding for labels

### ✅ Deep Neural Network
- Multi-layer perceptron architecture
- ReLU activation functions
- Dropout regularization
- Softmax output layer

### ✅ Model Training
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting
- Validation monitoring

### ✅ Model Evaluation
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Accuracy metrics

### ✅ Visualization
- Training accuracy and loss plots
- Confusion matrix heatmap
- Model performance analysis

### ✅ Model Deployment
- Model saving functionality
- Reproducible results with seed setting

## Technical Details

### Hyperparameter Optimization
- **Hidden Layers**: [512, 256, 128, 64] - Balanced complexity and performance
- **Dropout Rate**: 0.3 - Prevents overfitting while maintaining learning capacity
- **Batch Size**: 128 - Good balance between memory usage and training stability
- **Learning Rate**: Adaptive (Adam optimizer with ReduceLROnPlateau)

### Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Callbacks Used
- **EarlyStopping**: Monitors validation loss, stops training when no improvement
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus

## Results Analysis

The model achieves high accuracy through:
1. **Proper data preprocessing** with standardization
2. **Deep architecture** with multiple hidden layers
3. **Regularization techniques** to prevent overfitting
4. **Optimized hyperparameters** for the MNIST dataset

## Future Improvements

1. **Data Augmentation**: Add rotation, scaling, and noise to training data
2. **Architecture Tuning**: Experiment with different layer sizes and activation functions
3. **Advanced Regularization**: Try L1/L2 regularization and batch normalization
4. **Ensemble Methods**: Combine multiple models for better performance

## Troubleshooting

### Common Issues:
1. **Memory Error**: Reduce batch size or model complexity
2. **Slow Training**: Use GPU acceleration if available
3. **Overfitting**: Increase dropout rate or reduce model complexity
4. **Underfitting**: Increase model capacity or reduce regularization

### GPU Support:
To enable GPU acceleration, ensure you have:
- CUDA-compatible GPU
- cuDNN library
- TensorFlow-GPU installed

## Conclusion

This implementation successfully demonstrates:
- Deep learning model development with TensorFlow
- Comprehensive data preprocessing including standardization
- Model training with regularization techniques
- Performance evaluation and visualization
- Model deployment and saving

The achieved accuracy of ~98% demonstrates the effectiveness of the deep MLP architecture for multi-class classification tasks.
