import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, urlunparse
import httpx
import joblib
import re
import threading
import queue

# Load model and vectorizer
model = joblib.load("../models/tracker_detection_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

# Known trackers list
KNOWN_TRACKERS = ["doubleclick.net", "google-analytics.com", "facebook.com/tr", 
                  "adservice.google.com", "googletagmanager.com", "utm_", "fbclid"]

# Preprocessing (from predict_api)
def preprocess_url(url):
    return re.sub(r"https?://(www\.)?", "", url)

def find_known_trackers(url):
    return [t for t in KNOWN_TRACKERS if t in url]

def detect_with_model(url):
    processed = preprocess_url(url)
    vector = vectorizer.transform([processed])
    return model.predict(vector)[0] == 1

# Queue to send data back to UI
log_queue = queue.Queue()

class PrivacyProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.handle_request('GET')

    def do_POST(self):
        self.handle_request('POST')

    def handle_request(self, method):
        parsed_url = urlparse(self.path)
        target_url = urlunparse(parsed_url)

        log_entry = {
            "url": target_url,
            "known_trackers": find_known_trackers(target_url),
            "ml_tracker": detect_with_model(target_url)
        }

        # Add log to queue for UI consumption
        log_queue.put(log_entry)

        # Forward request to actual server
        self.forward_request(method, target_url)

    def forward_request(self, method, target_url):
        with httpx.Client(follow_redirects=True) as client:
            headers = {k: v for k, v in self.headers.items()}
            try:
                if method == 'GET':
                    response = client.get(target_url, headers=headers)
                elif method == 'POST':
                    length = int(self.headers.get('Content-Length', 0))
                    data = self.rfile.read(length) if length > 0 else None
                    response = client.post(target_url, headers=headers, content=data)

                self.send_response(response.status_code)
                for k, v in response.headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(response.content)

            except httpx.RequestError as e:
                self.send_error(502, f"Proxy error: {e}")

    def log_message(self, format, *args):
        return  # Suppress console logs

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def run_proxy():
    server = ThreadedHTTPServer(('127.0.0.1', 8081), PrivacyProxyHandler)
    server.serve_forever()

def start_proxy_in_thread():
    t = threading.Thread(target=run_proxy, daemon=True)
    t.start()
