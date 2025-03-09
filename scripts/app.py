from flask import Flask, request, render_template_string
import threading
import joblib
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import socket

app = Flask(__name__)

model = joblib.load("/home/anish/Privacy checker extension/models/tracker_detection_model.pkl")
vectorizer = joblib.load("/home/anish/Privacy checker extension/models/vectorizer.pkl")

proxy_port = 8080
proxy_status = "Not Running"

# Shared list to store intercepted requests
intercepted_requests = []

URL_REGEX = re.compile(
    r'^(https?:\/\/)?'
    r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,})'
    r'(:\d+)?'
    r'(\/.*)?$'
)


def is_valid_url(url):
    return bool(URL_REGEX.match(url))


def preprocess_url(url):
    return re.sub(r"https?://(www\.)?", "", url)


class PrivacyProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global intercepted_requests
        intercepted_requests.append(self.path)  # Log each requested path
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Intercepted Request to: " + self.path.encode())

    def log_message(self, format, *args):
        return


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    pass


def find_available_port(start_port=8080, max_port=8100):
    global proxy_port
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                proxy_port = port
                return port
    raise RuntimeError("No available ports found.")


def run_proxy():
    global proxy_status
    try:
        port = find_available_port()
        server = ThreadedHTTPServer(('127.0.0.1', port), PrivacyProxyHandler)
        proxy_status = f"Running on port {port}"
        server.serve_forever()
    except Exception as e:
        proxy_status = f"Failed to start proxy: {str(e)}"


proxy_thread = threading.Thread(target=run_proxy, daemon=True)
proxy_thread.start()

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>Privacy Checker - URL Tracker Detector & Proxy Log</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; }
        h1 { color: #333; }
        input, button { padding: 10px; width: 400px; margin: 5px 0; }
        .result, .error, .request-log { margin-top: 20px; font-size: 1.1em; }
        .proxy-status { margin-top: 30px; padding: 10px; background-color: #f0f0f0; border-left: 5px solid #007bff; }
        .tracker-list { margin-top: 15px; color: darkred; }
        .error { color: red; }
        .request-log { max-height: 200px; overflow-y: auto; background: #f9f9f9; padding: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Privacy Checker - URL Tracker Detector</h1>

    <form method="POST">
        <label for="url">Enter URL:</label><br>
        <input type="text" id="url" name="url" required><br>
        <button type="submit">Check URL</button>
    </form>

    {% if error %}
        <div class="error">
            <strong>Error:</strong> {{ error }}
        </div>
    {% endif %}

    {% if result %}
        <div class="result">
            <strong>Result:</strong> {{ result }}
            {% if trackers %}
                <div class="tracker-list">
                    <strong>Detected Trackers:</strong>
                    <ul>
                        {% for tracker in trackers %}
                            <li>{{ tracker }}</li>
                        {% endfor %}
                    </ul>
                </div>
            {% endif %}
        </div>
    {% endif %}

    {% if requests %}
        <div class="request-log">
            <strong>Intercepted Requests:</strong>
            <ul>
                {% for req in requests %}
                    <li>{{ req }}</li>
                {% endfor %}
            </ul>
        </div>
    {% endif %}

    <div class="proxy-status">
        <h2>Proxy Server Status</h2>
        <p>{{ proxy_status }}</p>
        <p>Set your browser proxy to <code>127.0.0.1:{{ proxy_port }}</code> to intercept requests.</p>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    global intercepted_requests
    result = None
    trackers = []
    error = None
    requests_list = intercepted_requests.copy()  # Fetch intercepted requests

    if request.method == "POST":
        intercepted_requests.clear()  # Clear log on new check
        url = request.form.get("url")

        if not is_valid_url(url):
            error = "Invalid URL entered. Please enter a valid URL starting with http:// or https://"
        else:
            processed_url = preprocess_url(url)
            url_vector = vectorizer.transform([processed_url])
            prediction = model.predict(url_vector)[0]

            result = "⚠️ Tracker Present" if prediction == 1 else "✅ No Trackers Detected"

            if prediction == 1:
                trackers = extract_possible_trackers(processed_url)

            # Display requests logged during this check (optional)
            requests_list = intercepted_requests.copy()

    return render_template_string(HTML_TEMPLATE, 
        result=result, trackers=trackers, error=error, 
        proxy_status=proxy_status, proxy_port=proxy_port, requests=requests_list)


def extract_possible_trackers(url):
    tracker_patterns = [
        "google-analytics", "doubleclick", "facebook", "adservice", 
        "tracking", "beacon", "pixel", "ads", "track"
    ]
    found_trackers = [pattern for pattern in tracker_patterns if pattern in url.lower()]
    return found_trackers


if __name__ == "__main__":
    app.run(debug=True)
