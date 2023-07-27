# Step zero: install the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1a. First, add a class column that is a zero for fake_news, and one for true
fake_news = pd.read_csv("Fake.csv")
fake_news["class"] = 0

real_news = pd.read_csv("True.csv")
real_news["class"] = 1

# Step 1b. Next, combine the two dataframes
df = pd.concat([fake_news, real_news])

# Step 2: Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['class']

# Step 3: Train a Machine Learning Classifier (Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Step 5: Get Predicted Probabilities
# Probability for the positive class (real news)
y_prob = classifier.predict_proba(X_test)[:, 1]

# Plot Histogram of Predicted Probabilities
plt.figure(figsize=(8, 6))
plt.hist(y_prob, bins=20, color='blue', alpha=0.7)
plt.xlabel('Predicted Probability (Real News)')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')
plt.show()

# Visualize the Confusion Matrix as a Heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
