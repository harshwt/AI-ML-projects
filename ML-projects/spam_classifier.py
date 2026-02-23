# spamming classifier project

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#nltk.download('stopwords')
#nltk.download('wordnet')

#converting encoding
message = pd.read_csv("spam.csv", encoding="latin-1")

#priting format of the data
print(message.head())
print(message.info())
print(message.size)

#basic cleaning technique using pandas

message.dropna(inplace=True)
message.drop_duplicates(inplace=True)
message = message.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

#renaming rows and cols
message.columns = ['label', 'text']
message.label = message['label'].map({'ham': 0, 'spam': 1})

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

import re
#cleaning data using re library
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words =  text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


message['clean_text'] = message['text'].apply(clean_text)

# Features and Labels
X = message['clean_text']
y = message['label']

# Convert text to numerical features (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("training and test data shape: \n")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("Predicted:", y_pred)
print("Actual:", y_test.values)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
