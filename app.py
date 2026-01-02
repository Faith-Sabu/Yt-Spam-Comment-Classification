from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load vectorizer and model
vectorizer = joblib.load("vct.pkl")
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        comment = request.form["comment"]
        if comment.strip():
            vect_comment = vectorizer.transform([comment])
            prediction = model.predict(vect_comment)[0]
            result = "⚠️ SPAM" if prediction == 1 else "✅ NOT SPAM"
        else:
            result = "Please enter a comment!"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
