from flask import Flask, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import flask


with open(f'reviews.pkl', 'rb') as f:
    model = pickle.load(f)
app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
    if flask.request.method == 'GET':
        return render_template('index.html')
    elif flask.request.method == 'POST':
        review = flask.request.form['review'].strip()
        with open(f'data.pkl','rb') as f:
            data = pickle.load(f)
        vec = TfidfVectorizer()
        vec.fit(data)
        inp = vec.transform([review])
        predictions = model.predict(inp)
        return render_template('index.html',result=predictions[0])

if __name__ == "__main__":
    app.run(debug=True)

