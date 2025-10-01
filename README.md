# cybersecurity-mini-project-threat-detection
using random forest ML.
To get the phising get, approach karggle: https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector
download the file, and upload to drive. In google colab, addd the file from the uploaded drive

#step 1- import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#step 2- loading the data into the variable df
df = pd.read_csv("phishing.csv")
print("Data shape:", df.shape)
print(df.head())

#Step 3: Prepare features and label
X = df.drop(columns=['class'], errors='ignore')
y = df['class'].map({1: 1, -1: 0})  #labelling - phishing=1, legitimate=0

#Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#step5: Initialize and train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#step6- Evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
print("Confusion Matrix:\n", confusion_matrix_result)

#step7- plotting the result in a graph
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Step 8: Manual Prediction
sample_input = X_test.iloc[1:2]
manual_pred = model.predict(sample_input)[0]
print("\n Manual prediction:", " Phishing" if manual_pred else " Legitimate")

