

# Step 1: Load and Preprocess the Data
# Replace 'fake_news.csv' and 'real_news.csv' with the appropriate paths to your dataset files
fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')

# Add a label '0' for fake news and '1' for real news
fake_news_df['label'] = 0
real_news_df['label'] = 1

# Combine fake and real news data into a single DataFrame
all_news_df = pd.concat([fake_news_df, real_news_df])

# Step 2: Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(all_news_df['text'])
y = all_news_df['label']

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
