# kannadaML
This code classifies Kannada MNIST dataset using PCA for dimensionality reduction and various ML models (Random Forest, Decision Trees, Naive Bayes, K-NN, SVM), evaluating with metrics, confusion matrices, and ROC-AUC curves for different component sizes.
Overview
This repository contains code for performing classification on the Kannada MNIST dataset using various machine learning models. The models utilized include Random Forest, Decision Trees, Naive Bayes, K-NN, and SVM. PCA (Principal Component Analysis) is employed for dimensionality reduction to enhance model performance.
Dataset
The Kannada MNIST dataset consists of handwritten digits similar to the traditional MNIST dataset but with characters from the Kannada script. The dataset is split into training and testing sets.
Dependencies
Python 3
Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn
Setup
Clone the repository:
git clone https://github.com/username/repository.git
Install dependencies:
pip install -r requirements.txt
Download the Kannada MNIST dataset and place it in the appropriate directory.
Usage
Run the main.py script to execute the classification process.
python main.py
The script will iterate over different component sizes for PCA, train various models, and evaluate their performance using classification reports, confusion matrices, and ROC-AUC curves.
Results
The results of each model's performance, including accuracy, precision, recall, and ROC-AUC scores, will be displayed for different PCA component sizes.
