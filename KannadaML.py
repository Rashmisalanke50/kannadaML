#Libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load the data
data_train = np.load(r"C:\Users\rashm\OneDrive\Desktop\streamlit\Data\PROJECT DATA\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_train.npz")['arr_0']
data_test = np.load(r"C:\Users\rashm\OneDrive\Desktop\streamlit\Data\PROJECT DATA\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_test.npz")['arr_0']
label_train = np.load(r"C:\Users\rashm\OneDrive\Desktop\streamlit\Data\PROJECT DATA\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_train.npz")['arr_0']
label_test = np.load(r"C:\Users\rashm\OneDrive\Desktop\streamlit\Data\PROJECT DATA\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_test.npz")['arr_0']

# Reshape 3D data to 2D
data_train = data_train.reshape(data_train.shape[0], -1)
data_test = data_test.reshape(data_test.shape[0], -1)

# Step 2: Perform PCA to 10 components
pca = PCA(n_components=10)
data_train_pca = pca.fit_transform(data_train)
data_test_pca = pca.transform(data_test)

# Step 5: Repeat for different component sizes
component_sizes = [10, 15, 20, 25, 30]

#Random Forest Classifier
for size in component_sizes:
    print(f"Component Size: {size}")
    pca = PCA(n_components=size)
    data_train_pca = pca.fit_transform(data_train)
    data_test_pca = pca.transform(data_test)

    # Step 3: Apply Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(data_train_pca, label_train)
    predictions = rfc.predict(data_test_pca)

    # Step 4: Compute metrics
    print("Classification Report:")
    print(classification_report(label_test, predictions))
    
    # Confusion Matrix
    cm = confusion_matrix(label_test, predictions)
    cm_mat = pd.DataFrame(data=cm, columns=[f"P{i}" for i in range(10)], index=[f"A{i}" for i in range(10)])
    sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
    plt.title(f"Confusion Matrix - Component Size: {size}")
    plt.show()

    # ROC - AUC curve
    probs = rfc.predict_proba(data_test_pca)

    # ROC - AUC curve for each class
    n_classes = len(np.unique(label_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each class')
    plt.legend(loc="lower right")
    plt.show()

# Decision Trees classifier
for size in component_sizes:
    print(f"Component Size: {size}")
    pca = PCA(n_components=size)
    data_train_pca = pca.fit_transform(data_train)
    data_test_pca = pca.transform(data_test)

    # Decision Trees classifier
    dtc = DecisionTreeClassifier()
    dtc.fit(data_train_pca, label_train)
    predictions = dtc.predict(data_test_pca)

    # Compute metrics
    print("Classification Report:")
    print(classification_report(label_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(label_test, predictions)
    cm_mat = pd.DataFrame(data=cm, columns=[f"P{i}" for i in range(10)], index=[f"A{i}" for i in range(10)])
    sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
    plt.title(f"Confusion Matrix - Component Size: {size}")
    plt.show()

    # ROC - AUC curve
    probs = dtc.predict_proba(data_test_pca)

    # ROC - AUC curve for each class
    n_classes = len(np.unique(label_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each class')
    plt.legend(loc="lower right")
    plt.show()

#Naive Bayes Model
for size in component_sizes:
    print(f"Component Size: {size}")
    pca = PCA(n_components=size)
    data_train_pca = pca.fit_transform(data_train)
    data_test_pca = pca.transform(data_test)

    # Naive Bayes Model
    nb = GaussianNB()
    nb.fit(data_train_pca, label_train)
    predictions = nb.predict(data_test_pca)

    # Compute metrics
    print("Classification Report:")
    print(classification_report(label_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(label_test, predictions)
    cm_mat = pd.DataFrame(data=cm, columns=[f"P{i}" for i in range(10)], index=[f"A{i}" for i in range(10)])
    sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
    plt.title(f"Confusion Matrix - Component Size: {size}")
    plt.show()

    # ROC - AUC curve
    probs = nb.predict_proba(data_test_pca)

    # ROC - AUC curve for each class
    n_classes = len(np.unique(label_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each class')
    plt.legend(loc="lower right")
    plt.show()
    
# K-NN Classifier
for size in component_sizes:
    print(f"Component Size: {size}")
    pca = PCA(n_components=size)
    data_train_pca = pca.fit_transform(data_train)
    data_test_pca = pca.transform(data_test)

    # K-NN Classifier with a specific algorithm
    knn = KNeighborsClassifier(algorithm='ball_tree')
    knn.fit(data_train_pca, label_train)
    predictions = knn.predict(data_test_pca)

    # Compute metrics
    print("Classification Report:")
    print(classification_report(label_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(label_test, predictions)
    cm_mat = pd.DataFrame(data=cm, columns=[f"P{i}" for i in range(10)], index=[f"A{i}" for i in range(10)])
    sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
    plt.title(f"Confusion Matrix - Component Size: {size}")
    plt.show()

    # ROC - AUC curve
    probs = knn.predict_proba(data_test_pca)

    # ROC - AUC curve for each class
    n_classes = len(np.unique(label_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each class')
    plt.legend(loc="lower right")
    plt.show()
    
# SVM Classifier
for size in component_sizes:
    print(f"Component Size: {size}")
    pca = PCA(n_components=size)
    data_train_pca = pca.fit_transform(data_train)
    data_test_pca = pca.transform(data_test)

    # SVM Classifier
    svm = SVC(probability=True)
    svm.fit(data_train_pca, label_train)
    predictions = svm.predict(data_test_pca)

    # Compute metrics
    print("Classification Report:")
    print(classification_report(label_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(label_test, predictions)
    cm_mat = pd.DataFrame(data=cm, columns=[f"P{i}" for i in range(10)], index=[f"A{i}" for i in range(10)])
    sns.heatmap(cm_mat, annot=True, cmap='YlGnBu')
    plt.title(f"Confusion Matrix - Component Size: {size}")
    plt.show()

    # ROC - AUC curve
    probs = svm.predict_proba(data_test_pca)

    # ROC - AUC curve for each class
    n_classes = len(np.unique(label_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each class')
    plt.legend(loc="lower right")
    plt.show()    