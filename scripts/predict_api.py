import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, urlunparse
import httpx
import joblib
import re
import threading

# Load your existing model and vectorizer from predict_api.py
model = joblib.load("/home/anish/Privacy checker extension/models/tracker_detection_model.pkl")
vectorizer = joblib.load("/home/anish/Privacy checker extension/models/vectorizer.pkl")

# Known trackers (can be extended)
KNOWN_TRACKERS = [
    "doubleclick.net", "google-analytics.com", "facebook.com/tr", 
    "adservice.google.com", "googletagmanager.com", "utm_", "fbclid"
]

# Preprocessing from predict_api.py (same logic)
def preprocess_url(url):
    return re.sub(r"https?://(www\.)?", "", url)

def is_valid_url(url):
    """Basic URL validation."""
    url_regex = re.compile(
        r'^(https?://)?'
        r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        r'(:\d+)?'
        r'(/[^\s]*)?$'
    )
    return re.match(url_regex, url)

def find_known_trackers(url):
    """Find known trackers directly from the URL."""
    return [tracker for tracker in KNOWN_TRACKERS if tracker in url]

def detect_with_model(url):
    """Use your ML model to predict if a URL is a tracker."""
    processed_url = preprocess_url(url)
    url_vector = vectorizer.transform([processed_url])
    prediction = model.predict(url_vector)[0]
    return prediction == 1  # True if tracker detected, False otherwise

class PrivacyProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._handle_request('GET')

    def do_POST(self):
        self._handle_request('POST')

    def _handle_request(self, method):
        parsed_url = urlparse(self.path)
        target_url = urlunparse(parsed_url)

        if not is_valid_url(target_url):
            self.send_error(400, "Invalid URL")
            return

        print(f"\n🔎 Intercepted request to: {target_url}")

        # Step 1: Check for known trackers
        known_trackers = find_known_trackers(target_url)

        # Step 2: Use ML Model to predict if it's a tracker
        is_tracker = detect_with_model(target_url)

        if is_tracker or known_trackers:
            print(f"⚠️ Tracker Alert for URL: {target_url}")
            if known_trackers:
                print(f"   - Known trackers detected: {', '.join(known_trackers)}")
            if is_tracker:
                print(f"   - ML Model Prediction: Tracker Detected")

        else:
            print("✅ No trackers detected by model or known list.")

        # Forward the request to actual target
        self.forward_request(method, target_url)

    def forward_request(self, method, target_url):
        """Forward request to target server and relay response back to client."""
        with httpx.Client(follow_redirects=True) as client:
            headers = {key: value for key, value in self.headers.items()}

            try:
                if method == 'GET':
                    response = client.get(target_url, headers=headers)
                elif method == 'POST':
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length) if content_length > 0 else None
                    response = client.post(target_url, headers=headers, content=post_data)

                # Relay response back to client
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.content)

            except httpx.RequestError as e:
                self.send_error(502, f"Proxy Error: {e}")

    def log_message(self, format, *args):
        """Suppress default HTTP server logs for cleaner output."""
        return

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def run_proxy(host='127.0.0.1', port=8080):
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, PrivacyProxyHandler)
    print(f"🔐 Privacy Proxy running on {host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_proxy()
