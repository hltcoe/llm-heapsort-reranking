import subprocess
import sys
import time

import requests


def wait_for_server(api_url):
    health_url = api_url + "/models"
    max_retries = 20
    retry_delay = 30

    for _ in range(max_retries):
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                print("vLLM server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(retry_delay)

    print("Failed to connect to vLLM server after max retries")
    return False


def launch_vllm(model: str, gpus: int, maxlen: int = 20480, port: int = 25251):
    cmd = f"python -m vllm.entrypoints.openai.api_server --model {model} --gpu-memory-utilization 0.95 --max-model-len {maxlen} --tensor-parallel-size {gpus} --port {port} --distributed-executor-backend=mp"
    server_process = subprocess.Popen(
        cmd.split(),
    )

    api_url = f"http://0.0.0.0:{port}/v1"
    if not wait_for_server(api_url):
        sys.exit(1)

    return api_url, server_process
