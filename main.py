import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_students = 200
data = {
    'Attendance': np.random.randint(0, 100, num_students), # Percentage
    'Homework_Scores': np.random.randint(0, 100, num_students),
    'Test_Scores': np.random.randint(0, 100, num_students),
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], size=num_students, p=[0.2, 0.25, 0.25, 0.15, 0.15])
}
df = pd.DataFrame(data)
# Create binary 'At-Risk' variable (simplified for demonstration)
df['At-Risk'] = np.where(df['Grade'].isin(['D', 'F']), 1, 0)
# --- 2. Data Cleaning and Preprocessing ---
# (In a real-world scenario, this would involve handling missing values, outliers etc.)
# For this example, data is already clean
# --- 3. Feature Engineering (Optional) ---
# Could add features like average score, etc.
df['Average_Score'] = (df['Homework_Scores'] + df['Test_Scores']) / 2
# --- 4. Model Training ---
X = df[['Attendance', 'Homework_Scores', 'Test_Scores', 'Average_Score']]
y = df['At-Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# --- 6. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not at Risk', 'At Risk'], 
            yticklabels=['Not at Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png")
plt.figure(figsize=(8,6))
sns.countplot(x='At-Risk', data=df)
plt.title('Distribution of At-Risk Students')
plt.xlabel('At-Risk (0=No, 1=Yes)')
plt.ylabel('Count')
plt.savefig('at_risk_distribution.png')
print("Plot saved to at_risk_distribution.png")