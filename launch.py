"""
Unified launcher - Runs both FastAPI backend and Gradio UI
This is the main entry point for HuggingFace Spaces
"""
import os
import threading
import time
import uvicorn
from app import build_interface

def run_backend():
    """Run FastAPI backend in a separate thread"""
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        log_level="info"
    )

def run_frontend():
    """Run Gradio UI (runs in main thread)"""
    # Give backend time to start
    time.sleep(3)
    
    demo = build_interface()
    demo.queue()  # Enable queueing for better performance
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Gradio on different port
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    print("🚀 Starting SRE Incident Triage Environment...")
    print("=" * 60)
    print("📡 Backend API: http://localhost:7860")
    print("📊 API Docs: http://localhost:7860/docs")
    print("🎮 Gradio UI: http://localhost:7861")
    print("=" * 60)
    
    # Start backend in background thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Run frontend in main thread
    run_frontend()
