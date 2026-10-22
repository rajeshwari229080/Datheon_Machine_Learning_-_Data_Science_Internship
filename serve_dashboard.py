"""Zero-dependency local HTTP server to launch the Central Laboratory Dashboard."""

import argparse
import http.server
import os
import socket
import socketserver
import sys
import webbrowser

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local dashboard preview
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    parser = argparse.ArgumentParser(description="Serve the Datheon ML Internship Laboratory Dashboard.")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port number (default: {PORT})")
    parser.add_argument("--open", action="store_true", help="Automatically open browser on start")
    args = parser.parse_args()

    port = args.port
    while is_port_in_use(port):
        print(f"[Notice] Port {port} is occupied, trying port {port + 1}...")
        port += 1

    handler = CustomHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/index.html"
        print("\n" + "=" * 65)
        print("  DATHEON MACHINE LEARNING & DATA SCIENCE INTERNSHIP SUITE")
        print("  Central Laboratory Dashboard Server Online")
        print("=" * 65)
        print(f"  Access URL:  {url}")
        print(f"  Local Path:  {DIRECTORY}")
        print("  Press CTRL+C to terminate the server.\n" + "=" * 65 + "\n")

        if args.open:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Shutting down] Server stopped gracefully.")
            httpd.server_close()


if __name__ == "__main__":
    main()
