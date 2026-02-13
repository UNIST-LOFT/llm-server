import argparse
import logging
import os
import subprocess

MODELS = {
    'llama-4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'llama-3-2': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen-3-next': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'qwen-3': 'Qwen/Qwen3-1.7B',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Server")
    parser.add_argument('model', type=str, choices=MODELS.keys(),
                        help='The model to serve.')
    parser.add_argument('--gpu-id', nargs='+', type=int, default=None,
                        help='List of GPU IDs to use for model loading.')
    parser.add_argument('--gpu-mem-utilization', type=float, default=0.85,
                        help='GPU memory utilization for model loading. Default: 0.85')
    parser.add_argument('-p', '--port', type=int, default=5000,
                        help='Port to run the server on.')
    parser.add_argument('--log-debug', action='store_true',
                        help='Enable debug level logging.')
    parser.add_argument('--max-tokens', type=int, default=8192,
                        help='Max length of tokens for model (prompt + response). Defualt: 8192')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG if args.log_debug else logging.INFO)

    if args.gpu_id is not None:
        logger.info(f"Using GPUs: {args.gpu_id}")
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_id))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # Ensure consistent GPU ordering
    if args.log_debug:
        os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

    dtype = 'bfloat16'

    cmd = [
        'vllm', 'serve', MODELS[args.model],
        '--tensor-parallel-size', str(len(args.gpu_id)) if args.gpu_id is not None else '1',
        '--dtype', dtype,
        '--max-model-len', str(args.max_tokens),
        '--gpu-memory-utilization', str(args.gpu_mem_utilization),
        '--host', '127.0.0.1',
        '--port', str(args.port),
        '--max-num-seqs', '64'
    ]

    logger.info(f"Starting LLM server with command: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"LLM server exited with code {res.returncode}")
    else:
        logger.info("LLM server exited successfully")