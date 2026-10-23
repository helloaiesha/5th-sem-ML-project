import tkinter as tk
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

data = pd.read_csv("it_complaints.csv")

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(data["Complaint"], data["Category"])


def classify():
    complaint = box.get()

    if complaint:
        category = model.predict([complaint])[0]

        info = {
            "Network Issue": ("High", "Network Support"),
            "Login/Password Issue": ("Medium", "IT Security"),
            "Software Issue": ("Medium", "Software Support"),
            "Hardware Issue": ("High", "Hardware Support"),
            "Printer Issue": ("Medium", "Printer Support"),
            "Email Issue": ("Medium", "Email Support")
        }

        priority, department = info.get(
            category, ("Medium", "IT Support")
        )

        ticket = "IT-" + str(random.randint(1000, 9999))

        output.config(
            text=f"Category: {category}\n"
                 f"Priority: {priority}\n"
                 f"Department: {department}\n"
                 f"Ticket ID: {ticket}\n"
                 f"Status: OPEN"
        )


app = tk.Tk()
app.title("Smart IT Helpdesk")
app.geometry("600x450")
app.configure(bg="lightblue")

tk.Label(
    app,
    text="SMART IT HELPDESK",
    font=("Arial", 22, "bold"),
    bg="lightblue",
    fg="darkblue"
).pack(pady=25)

tk.Label(
    app,
    text="Enter IT Complaint",
    font=("Arial", 12, "bold"),
    bg="lightblue"
).pack()

box = tk.Entry(app, width=60)
box.pack(pady=15)

tk.Button(
    app,
    text="CLASSIFY COMPLAINT",
    command=classify,
    bg="darkblue",
    fg="white"
).pack(pady=10)

output = tk.Label(
    app,
    text="Category: -",
    font=("Arial", 13, "bold"),
    bg="white",
    fg="darkblue",
    width=45,
    height=8,
    justify="left"
)
output.pack(pady=25)

app.mainloop()