import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Collection & Preprocessing
class DataHandler:
    def __init__(self):
        self.student_data = None

    def load_student_data(self, file_path="student_data.csv"):
        """Load and preprocess student performance data"""
        self.student_data = pd.read_csv(file_path)
        self.student_data.dropna(inplace=True)  # Remove missing values
        return self.student_data

    def preprocess_data(self):
        """Feature Scaling and Splitting Data"""
        X = self.student_data[['hours_studied', 'previous_scores', 'attendance']]
        y = self.student_data['exam_result'].map({'Pass': 1, 'Fail': 0})  # Convert to binary

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

# Step 2: Supervised Learning (Classification)
class ExamPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def predict(self, study_data):
        return self.model.predict([study_data])[0]

# Step 3: Unsupervised Learning (Student Clustering)
class StudentClusterer:
    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def train(self, X):
        self.kmeans.fit(X)
        return self.kmeans.labels_

    def visualize_clusters(self, X):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=self.kmeans.labels_, palette='viridis')
        plt.xlabel("Hours Studied")
        plt.ylabel("Previous Scores")
        plt.title("Student Clusters")
        plt.show()

# Step 4: Load Dataset and Train Models
data_handler = DataHandler()
student_data = data_handler.load_student_data()
print("Loaded Student Data:")
print(student_data.head())

X_train, X_test, y_train, y_test = data_handler.preprocess_data()

exam_predictor = ExamPredictor()
exam_predictor.train(X_train, y_train)
exam_predictor.evaluate(X_test, y_test)

student_clusterer = StudentClusterer()
labels = student_clusterer.train(X_train)
student_clusterer.visualize_clusters(X_train)
