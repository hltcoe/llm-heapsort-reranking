import time
import requests
import sys
from datetime import datetime

def check_server(url="http://0.0.0.0:8000/v1/models"):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def main(url):
    start_time = time.time()
    timeout = 1800  # seconds
    retry_interval = 30  # seconds
    
    print(f"{datetime.now()} Waiting for vLLM OpenAI server to start...")
    
    while time.time() - start_time < timeout:
        if check_server(url):
            print(f"{datetime.now()} Server is up!")
            sys.exit(0)
        
        time_left = timeout - (time.time() - start_time)
        print(f"{datetime.now()} Server not ready, waiting... ({time_left:.1f} seconds remaining)")
        time.sleep(retry_interval)
    
    print(f"*** *** {datetime.now()} Timeout reached. Server did not start within {timeout} seconds. *** ***")
    sys.exit(1)

if __name__ == "__main__":
    url = sys.argv[1]
    main(url)
