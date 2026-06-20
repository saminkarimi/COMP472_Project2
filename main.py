import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['lable', 'message']]

X = df['message']
y = df['lable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

probabilities = model.predict_proba(X_test_tfidf)
print("First prediction:", y_pred[0])

confidence = max(probabilities[0]) * 100
print("Confidence:", confidence, "%")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
spam_count = (df['lable'] == 'spam').sum()
ham_count = (df['lable'] == 'ham').sum()

plt.bar(['Spam', 'Ham'], [spam_count, ham_count])

plt.title('Spam vs Ham Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')

plt.show()

user_message = input("Enter a message: ")

user_vector = vectorizer.transform([user_message])

prediction = model.predict(user_vector)

probability = model.predict_proba(user_vector)

confidence = max(probability[0]) * 100

if prediction[0] == "spam":
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")

print("Confidence:", round(confidence, 2), "%")