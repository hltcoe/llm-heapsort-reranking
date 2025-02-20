#!/bin/bash

set -e

MODEL=meta-llama/Llama-3.1-8B-Instruct
MAXLEN=20480
NUM_GPUS=2
#MODEL=meta-llama/Llama-2-7b-chat-hf
#MAXLEN=4096
#NUM_GPUS=1
export API_PORT=8128

echo launching vLLM...
set -x
python -m vllm.entrypoints.openai.api_server --model $MODEL --gpu-memory-utilization 0.95 --tensor-parallel-size $NUM_GPUS --distributed-executor-backend=mp --port $API_PORT --max-model-len $MAXLEN &
API_SERVER_PID=$!

# TODO add trap line here

python -m llm_heapsort_reranking.wait_for_vllm http://0.0.0.0:${API_PORT}/v1/models

set +e

#python -m llm_heapsort_reranking.run $*
python heapsort-llama2.py $*

kill -TERM $API_SERVER_PID
wait $API_SERVER_PID
