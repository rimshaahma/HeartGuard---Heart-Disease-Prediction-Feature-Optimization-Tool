

# **HeartGuard - Heart Disease Prediction & Feature Optimization Tool**

## **Project Overview**

**HeartGuard** is an interactive Python-based application designed to predict heart disease and optimize feature selection for machine learning classification tasks. The tool uses **Recursive Feature Elimination (RFE)** to optimize feature selection and compares model performance with and without selected features. The entire process is visualized through a **Graphical User Interface (GUI)**, making it user-friendly and approachable for anyone interested in feature optimization and machine learning.

The dataset used in this project is the **Heart Disease dataset** from Kaggle, which contains multiple patient features to predict heart disease.

---

## **Problem Statement**

The aim of this project is to predict whether a patient is likely to have heart disease based on various attributes like age, blood pressure, cholesterol level, and more. Alongside prediction, we also aim to demonstrate the use of **feature selection** techniques like **Recursive Feature Elimination (RFE)** and compare the performance of models trained on the original and optimized features.

---

## **Key Concepts and Definitions**

### **1. Recursive Feature Elimination (RFE)**

- **Definition**: Recursive Feature Elimination (RFE) is a feature selection technique that recursively removes the least important features and builds the model until the desired number of features remains. The goal is to identify the most influential features.
  
- **Purpose**: 
  - To eliminate irrelevant or redundant features.
  - To reduce model complexity, which helps improve both performance and interpretability.

- **How it works**: 
  - RFE starts by training a model using all features.
  - Then, it ranks the importance of each feature.
  - The least important features are removed, and the process repeats until the optimal number of features is selected.

---

### **2. Feature Importance (from Random Forest)**

- **Definition**: Feature importance is a technique used by tree-based algorithms like Random Forest to estimate the significance of each feature in making predictions. It tells us which features have the highest contribution to the accuracy of the model.
  
- **Purpose**: 
  - To identify the most influential features.
  - To improve the model by focusing on the important features and discarding less relevant ones.

- **How it works**: 
  - Random Forest uses a method called "Gini impurity" to measure the importance of each feature.
  - Features that, when split, reduce the error most significantly are considered more important.

---

### **3. Model Performance Comparison**

In this project, we compare two models:
1. **Model trained with all features**: A model trained on the entire dataset without feature selection.
2. **Model trained with selected features**: A model trained on only the most important features selected using RFE.

We evaluate the models based on standard classification metrics:
- **Accuracy**: The proportion of correct predictions out of total predictions.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to the actual positives.
- **F1-score**: A weighted average of Precision and Recall.

---

## **Steps and Procedures**

### **Step 1: Loading the Dataset**

We load the **Heart Disease dataset** from Kaggle. This dataset includes various medical features (like age, blood pressure, cholesterol levels) and a target variable (whether the person has heart disease).

```python
import pandas as pd
data = pd.read_csv('heart_disease.csv')
```

### **Step 2: Data Preprocessing**

- **Handling missing values**: We replace any missing values with the mean of the column. This helps to ensure that the model is trained on a complete dataset.
- **Feature scaling**: We scale the features to standardize them. StandardScaler is used here to scale each feature to have a mean of 0 and a standard deviation of 1.

```python
# Handle missing values by filling with the column's mean value
data.fillna(data.mean(), inplace=True)

# Feature scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = data.drop('target', axis=1)  # Feature set
y = data['target']  # Target variable
X_scaled = scaler.fit_transform(X)
```

### **Step 3: Feature Selection using RFE**

We use **RFE (Recursive Feature Elimination)** to select the top 5 most relevant features. Here, we use a **RandomForestClassifier** to rank feature importance.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()  # We use Random Forest for feature importance
selector = RFE(model, n_features_to_select=5)  # Select 5 important features
selector = selector.fit(X_scaled, y)

# Get the selected features
selected_features = X.columns[selector.support_]
print("Selected features:", selected_features)
```

### **Step 4: Model Training**

- **Model 1 (All Features)**: We train a model using all available features.
- **Model 2 (Selected Features)**: We train a model using only the features selected by RFE.

```python
# Train model with all features
model_all = RandomForestClassifier()
model_all.fit(X_scaled, y)

# Train model with selected features
X_selected = X_scaled[:, selector.support_]
model_selected = RandomForestClassifier()
model_selected.fit(X_selected, y)
```

### **Step 5: Model Performance Comparison**

We compare the performance of both models by evaluating metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

```python
from sklearn.metrics import accuracy_score, classification_report

# Predictions for both models
y_pred_all = model_all.predict(X_scaled)
y_pred_selected = model_selected.predict(X_selected)

# Performance metrics
accuracy_all = accuracy_score(y, y_pred_all)
accuracy_selected = accuracy_score(y, y_pred_selected)

# Print results
print("Accuracy with all features:", accuracy_all)
print("Accuracy with selected features:", accuracy_selected)
print("Classification Report (All Features):", classification_report(y, y_pred_all))
print("Classification Report (Selected Features):", classification_report(y, y_pred_selected))
```

### **Step 6: Visualization of Model Performance (Optional)**

We can visualize the comparison of accuracy for both models.

```python
import matplotlib.pyplot as plt

# Plot accuracy comparison
plt.bar(['All Features', 'Selected Features'], [accuracy_all, accuracy_selected], color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()
```

### **Step 7: GUI Implementation**

The GUI allows users to:
1. Upload the dataset.
2. Perform feature selection using RFE.
3. Train models and compare performance.
4. View the results in a user-friendly manner.

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from your_code import load_data, preprocess_data, perform_rfe, train_model, compare_performance

class HeartDiseaseModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Prediction & Feature Optimization Tool")
        
        self.filepath = None
        
        # Button to upload CSV file
        self.upload_btn = tk.Button(root, text="Upload CSV", command=self.upload_csv, width=20, bg="blue", fg="white")
        self.upload_btn.pack(pady=20)
        
        # Button to start model training and optimization
        self.train_btn = tk.Button(root, text="Start Training & Optimization", command=self.train_model, width=20, bg="green", fg="white")
        self.train_btn.pack(pady=20)

        # Label to display results
        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)
    
    def upload_csv(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            messagebox.showinfo("Success", "CSV file uploaded successfully!")
        else:
            messagebox.showwarning("Error", "No file selected!")

    def train_model(self):
        if not self.filepath:
            messagebox.showwarning("Error", "Please upload a CSV file first!")
            return
        
        # Load and preprocess the data
        X, y = load_data(self.filepath)
        X_scaled, y = preprocess_data(X, y)
        
        # Apply RFE and train models
        selected_features, accuracy_all, accuracy_selected = perform_rfe(X_scaled, y)
        
        # Display results
        result_text = f"Selected Features: {', '.join(selected_features)}\nAccuracy with all features: {accuracy_all:.4f}\nAccuracy with selected features: {accuracy_selected:.4f}"
        self.result_label.config(text=result_text)

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseModelApp(root)
    root.mainloop()
```

---

## **Conclusion**

In this project, we successfully:
- **Predicted heart disease risk** using machine learning models.
- **Optimized feature selection** using **RFE** to identify the most important features for prediction.
- **Compared model performance** with and without selected features, showing how feature optimization can improve model performance.

The GUI provides an easy-to-use interface to interact with the tool, upload datasets, and view results.

---

## **Requirements**

To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
