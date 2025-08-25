"""
Setup script for Assignment No-1: Multi-Class Classifier
This script helps with installation and project setup.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing packages. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    return True

def check_tensorflow():
    """Check TensorFlow installation"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} is installed")
        return True
    except ImportError:
        print("❌ TensorFlow is not installed")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
        return True
    except:
        print("⚠️  Could not check GPU status")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'plots', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Setup for Assignment No-1: Multi-Class Classifier")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check TensorFlow
    if not check_tensorflow():
        return
    
    # Check GPU
    check_gpu()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("You can now run:")
    print("  python multiclass_classifier.py    # Main implementation")
    print("  python demo.py                     # Quick demo")
    print("  python hyperparameter_tuning.py    # Parameter optimization")
    print("=" * 60)

if __name__ == "__main__":
    main()
