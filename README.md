# Fake-Smile-Detection-Model
Fake Smile Detection Model using CNNs in Python

Original images are in fake_smiles and true_smiles folders, and the extracted features are then stored in respective folders (for example, true_eyes folder contains images of extracted eyes from images in true_smiles folder)

All the codes (from data preparation, feature extraction to model building) in Python are written in single "fake_smile_detection_overall.py" file, which makes use of numpy, matplotlib, tensorflow, scikit-learn and opencv libraries of Python

Three features are extracted from original images: faces, eyes and mouths, and three convolutional neural network models are created: LeNet, AlexNet, and ResNet. Cross-validations are used to choose the optimal feature, optimal model with tuned associated parameters 
