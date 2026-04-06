#!/usr/bin/env python3
"""
Launch script for SRE Incident Triage
Starts both the FastAPI backend and the HTML UI server
"""
import subprocess
import sys
import time
import requests
import signal
import os

def check_port(port):
    """Check if a port is already in use"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

def main():
    print("=" * 70)
    print("🚀 Starting SRE Incident Triage Environment")
    print("=" * 70)
    
    # Check if ports are available
    if check_port(8000):
        print("⚠️  Port 8000 is already in use (backend may already be running)")
    
    if check_port(7860):
        print("⚠️  Port 7860 is already in use (UI may already be running)")
        print("\n💡 If services are already running, open http://localhost:7860")
        return
    
    processes = []
    
    try:
        # Start FastAPI backend
        print("\n📡 Starting FastAPI backend on port 8000...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.main:app", 
             "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(backend)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to initialize...")
        for i in range(10):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    print("✅ Backend is ready!")
                    break
            except:
                pass
        else:
            print("⚠️  Backend may not be ready, but continuing...")
        
        # Start UI server
        print("\n🎮 Starting UI server on port 7860...")
        ui_server = subprocess.Popen(
            [sys.executable, "serve_ui.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(ui_server)
        
        time.sleep(1)
        
        print("\n" + "=" * 70)
        print("✨ SRE Incident Triage is ready!")
        print("=" * 70)
        print(f"🌐 UI:         http://localhost:7860")
        print(f"📡 Backend:    http://localhost:8000")
        print(f"📚 API Docs:   http://localhost:8000/docs")
        print("=" * 70)
        print("\n👉 Press Ctrl+C to stop all services\n")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if backend.poll() is not None:
                print("❌ Backend process died unexpectedly")
                break
            if ui_server.poll() is not None:
                print("❌ UI server process died unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean up processes
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
