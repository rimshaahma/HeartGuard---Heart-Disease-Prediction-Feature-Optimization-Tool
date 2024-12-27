import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class HeartDiseaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Prediction & Feature Optimization Tool")
        
        self.filepath = None
        self.data = None
        self.X = None
        self.y = None
        self.model_selected = None
        self.model_all = None
        
        # Upload CSV Button
        self.upload_btn = tk.Button(root, text="Upload Heart Disease CSV", command=self.upload_csv, width=30, bg="blue", fg="white")
        self.upload_btn.pack(pady=20)
        
        # Start feature selection and model training
        self.process_btn = tk.Button(root, text="Process & Train Model", command=self.process_data, width=30, bg="green", fg="white")
        self.process_btn.pack(pady=20)
        
        # Display Result Text
        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)
        
    def upload_csv(self):
        # Let the user choose the dataset file
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            self.data = pd.read_csv(self.filepath)
            messagebox.showinfo("Success", "Dataset uploaded successfully!")
        else:
            messagebox.showwarning("Error", "No file selected!")

    def preprocess_data(self):
        # Step 1: Handle missing values (optional, can be skipped if dataset is clean)
        self.data.fillna(self.data.mean(), inplace=True)
        
        # Step 2: Split data into features and target variable
        X = self.data.drop('target', axis=1)  # Replace 'target' with the actual target column name in the dataset
        y = self.data['target']  # Adjust accordingly
        
        # Step 3: Normalize the data using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y
    
    def process_data(self):
        if not self.filepath:
            messagebox.showwarning("Error", "Please upload a dataset first!")
            return

        X, y = self.preprocess_data()
        
        # Step 4: Apply RFE for feature selection
        model = RandomForestClassifier(random_state=42)
        selector = RFE(model, n_features_to_select=5)  # Select top 5 features
        selector = selector.fit(X, y)
        
        # Get selected features
        selected_features = [feature for feature, support in zip(self.data.columns, selector.support_) if support]
        self.result_label.config(text=f"Selected Features: {', '.join(selected_features)}")

        # Step 5: Train Model with Selected Features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model with selected features
        model_selected = RandomForestClassifier(random_state=42)
        model_selected.fit(X_train[:, selector.support_], y_train)
        
        # Predict on test data with selected features
        y_pred_selected = model_selected.predict(X_test[:, selector.support_])
        
        # Step 6: Evaluate Model with Selected Features
        accuracy_selected = accuracy_score(y_test, y_pred_selected)
        report_selected = classification_report(y_test, y_pred_selected)
        
        # Display results
        result_text = f"Accuracy with Selected Features: {accuracy_selected:.4f}\n\n{report_selected}"
        self.result_label.config(text=result_text)
        
        # Step 7: Visualize the Confusion Matrix for the Selected Features Model
        cm_selected = confusion_matrix(y_test, y_pred_selected)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_selected, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Selected Features)')
        plt.show()
        
        # Step 8: Model with All Features
        model_all = RandomForestClassifier(random_state=42)
        model_all.fit(X_train, y_train)
        
        # Predict on test data with all features
        y_pred_all = model_all.predict(X_test)
        
        # Evaluate Model with All Features
        accuracy_all = accuracy_score(y_test, y_pred_all)
        report_all = classification_report(y_test, y_pred_all)
        
        # Step 9: Display Results for All Features
        result_text_all = f"Accuracy with All Features: {accuracy_all:.4f}\n\n{report_all}"
        self.result_label.config(text=result_text_all)
        
        # Step 10: Visualize the Confusion Matrix for All Features Model
        cm_all = confusion_matrix(y_test, y_pred_all)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (All Features)')
        plt.show()
        
        # Step 11: Feature Importance (for Best Model)
        importances = model_all.feature_importances_
        feature_importance = pd.DataFrame({'Feature': self.data.columns[:-1], 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        # Visualize Feature Importance
        feature_importance.plot(kind='barh', x='Feature', y='Importance', legend=False, figsize=(10, 6))
        plt.title('Feature Importance from Random Forest Model')
        plt.show()

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseGUI(root)
    root.mainloop()
