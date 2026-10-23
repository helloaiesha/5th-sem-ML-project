import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("it_complaints.csv")

data.columns = data.columns.str.strip()

X = data["Complaint"].astype(str)
y = data["Category"].astype(str)

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X).toarray()

model = GaussianNB()

model.fit(X_tfidf, y)

while True:
    complaint = input("\nEnter your IT Complaint (type exit to stop): ")

    if complaint.lower() == "exit":
        break

    complaint_tfidf = vectorizer.transform([complaint]).toarray()

    prediction = model.predict(complaint_tfidf)

    print("Predicted Category:", prediction[0])
    