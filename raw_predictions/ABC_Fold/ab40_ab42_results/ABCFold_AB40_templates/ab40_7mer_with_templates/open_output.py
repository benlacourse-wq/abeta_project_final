
import http.server
import socketserver
import webbrowser
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8000

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control",
                        "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

try:
    with socketserver.TCPServer(("", PORT),
                                NoCacheHTTPRequestHandler) as httpd:
        print(
            f"Serving at port 8000: http://localhost:8000/index.html"
            )
        print("Press Ctrl+C to stop the server")
        webbrowser.open(f"http://localhost:8000/index.html")
        httpd.serve_forever()
except KeyboardInterrupt:
    print("Server stopped")
    httpd.server_close()
    sys.exit(0)
