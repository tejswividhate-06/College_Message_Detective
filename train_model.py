import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load the dataset
data = pd.read_csv("college_messages.csv")

# 2. Separate messages and labels
X = data["message"]
y = data["label"]

# 3. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Convert text into numerical features using Bag of Words
vectorizer = CountVectorizer()

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# 5. Create and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# 6. Make predictions on the test data
predictions = model.predict(X_test_bow)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("=" * 55)
print("       COLLEGE MESSAGE DETECTIVE - MODEL TRAINING")
print("=" * 55)

print(f"\nDataset size: {len(data)} messages")
print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# 8. Display classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 9. Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 10. Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# 11. Save the trained vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\n" + "=" * 55)
print("Training completed successfully!")
print("Created: model.pkl")
print("Created: vectorizer.pkl")
print("=" * 55)
