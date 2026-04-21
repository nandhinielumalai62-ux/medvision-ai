import os
import subprocess
import time
import urllib.request
import signal
import sys

def is_server_running(port):
    try:
        # Check Streamlit health endpoint
        urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1)
        return True
    except:
        return False

def get_python_exe():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(base_dir, 'venv_tf', 'Scripts', 'pythonw.exe')):
        return os.path.join(base_dir, 'venv_tf', 'Scripts', 'pythonw.exe')
    if os.path.exists(os.path.join(base_dir, 'venv', 'Scripts', 'pythonw.exe')):
        return os.path.join(base_dir, 'venv', 'Scripts', 'pythonw.exe')
    
    # Fallback to regular python if pythonw is missing
    if os.path.exists(os.path.join(base_dir, 'venv_tf', 'Scripts', 'python.exe')):
        return os.path.join(base_dir, 'venv_tf', 'Scripts', 'python.exe')
    if os.path.exists(os.path.join(base_dir, 'venv', 'Scripts', 'python.exe')):
        return os.path.join(base_dir, 'venv', 'Scripts', 'python.exe')
        
    return sys.executable

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    port = 8501
    
    # Start streamlit if not already running
    process = None
    if not is_server_running(port):
        python_exe = get_python_exe()
        # Ensure we use python from the same dir to load 'streamlit' module properly
        streamlit_cmd = [python_exe, '-m', 'streamlit', 'run', 'src/app.py', '--server.headless=true', '--server.port', str(port)]
        
        # Hide the command window flag
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        process = subprocess.Popen(
            streamlit_cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            startupinfo=startupinfo
        )
        
        # Wait up to 120 seconds for Streamlit to start up (TensorFlow takes time)
        start_time = time.time()
        while time.time() - start_time < 120:
            if is_server_running(port):
                break
            time.sleep(2)
            
    if is_server_running(port):
        # Launch modern browser in App Mode (borderless, minimalist UI, looks like native app)
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        
        if os.path.exists(edge_path):
            subprocess.Popen([edge_path, f'--app=http://localhost:{port}'])
        elif os.path.exists(chrome_path):
            subprocess.Popen([chrome_path, f'--app=http://localhost:{port}'])
        else:
            import webbrowser
            webbrowser.open(f'http://localhost:{port}')
    else:
        # If it failed to start, show an error alert
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, f"Error: The MedVision AI Core failed to start within 120 seconds or encountered an error.", "MedVision AI - Startup Failed", 16)
        if process:
            process.terminate()

    # Note: We do not wait for the browser process or terminate the Server here.
    # Chromium browsers immediately fork and exit the initial command process.
    # The server is left running in the background. If the user launches the app again,
    # the script will detect the existing server and just open a new browser window.

if __name__ == '__main__':
    main()
