import subprocess
import sys
import time
import os

def run_backend():
    """Start the Flask API server."""
    print("\n" + "="*60)
    print("STARTING FLASK BACKEND API")
    print("="*60 + "\n")
    
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    subprocess.Popen([sys.executable, "app.py"], cwd=backend_path)
    time.sleep(3)  # Wait for API to start
    print("✓ Backend API started on http://localhost:5000")

def run_streamlit():
    """Start the Streamlit dashboard."""
    print("\n" + "="*60)
    print("STARTING STREAMLIT DASHBOARD")
    print("="*60 + "\n")
    
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app_streamlit.py"], cwd=backend_path)
    print("✓ Dashboard started on http://localhost:8501")

def main():
    """Run the complete application."""
    print("\n" + "="*60)
    print("PRODUCTIONIZED DASHBOARD")
    print("="*60)
    
    try:
        # Start backend API
        run_backend()
        
        # Start Streamlit frontend
        run_streamlit()
        
        print("\n" + "="*60)
        print("✓ APPLICATION RUNNING")
        print("="*60)
        print("\nAccess your dashboard at: http://localhost:8501")
        print("API running at: http://localhost:5000")
        print("\nPress Ctrl+C to stop all services\n")
        
        # Keep the process running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down application...")
        sys.exit(0)

if __name__ == "__main__":
    main()
