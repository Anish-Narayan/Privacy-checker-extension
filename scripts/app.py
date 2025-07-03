import joblib
import re
from flask import Flask, request

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("/home/anish/Privacy-checker-extension/models/tracker_detection_model.pkl")
vectorizer = joblib.load("/home/anish/Privacy-checker-extension/models/vectorizer.pkl")

def is_valid_url(url):
    url_regex = re.compile(
        r'^(https?|ftp):\/\/[^\s/$.?#].[^\s]*$', re.IGNORECASE)
    return re.match(url_regex, url) is not None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            if not is_valid_url(url):
                result = "Invalid URL"
            else:
                # Transform the URL using the saved vectorizer
                url_features = vectorizer.transform([url])
                
                # Predict
                prediction = model.predict(url_features)[0]
                result = "Tracker Present" if prediction == 1 else "No Trackers Found"
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Privacy Checker</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; margin: 50px; }}
            input {{ padding: 10px; width: 300px; }}
            button {{ padding: 10px 20px; cursor: pointer; }}
            .result {{ margin-top: 20px; font-size: 1.2em; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>Check if a URL has a Tracker</h2>
        <form method="POST">
            <input type="text" name="url" placeholder="Enter URL" required>
            <button type="submit">Check</button>
        </form>
        
        {f'<div class="result">Result: {result}</div>' if result else ''}
    </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True)