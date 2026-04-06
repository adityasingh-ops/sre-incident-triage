#!/usr/bin/env python3
"""
Simple HTTP server to serve the SRE Triage HTML UI
"""
import http.server
import socketserver
import os
import sys

PORT = 7860
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Serve index.html when accessing root
        if self.path == '/':
            self.path = '/sre_triage.html'
        return super().do_GET()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 70)
        print(f"🎮 SRE Incident Triage UI Server")
        print("=" * 70)
        print(f"📡 Serving on: http://localhost:{PORT}")
        print(f"📁 Directory: {DIRECTORY}")
        print(f"🌐 Open your browser to: http://localhost:{PORT}")
        print("=" * 70)
        print("Press Ctrl+C to stop the server")
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            sys.exit(0)
