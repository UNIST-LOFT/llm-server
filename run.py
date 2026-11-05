import argparse
import logging
import os
import subprocess

MODELS = {
    'llama-4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'llama-3-2': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen-3-next': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'qwen-3': 'Qwen/Qwen3-1.7B',
    'qwen-3-coder-next': 'Qwen/Qwen3-Coder-Next',
    'deepseek-v4-flash': 'deepseek-ai/DeepSeek-V4-Flash',
    'qwen-3-coder': 'nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4', # quantized
    'glm-4.6': 'bullpoint/GLM-4.6-AWQ', # quantized
    'minimax-m3.1-flash': 'MiniMaxAI/MiniMax-M3.1-Flash'
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
    parser.add_argument('--max-tokens', type=int, default=131072,
                        help='Max length of tokens for model (prompt + response). Default: 131072')
    parser.add_argument('--enable-tool', action='store_true',
                        help='Enable the use of external tools. Recommended when using LLMs in agentic.')
    parser.add_argument('-i', '--model-path', type=str, default=None,
                        help='Path to the local model checkpoint. Overrides the default model path if provided.')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG if args.log_debug else logging.INFO)

    if args.gpu_id is not None:
        logger.info(f"Using GPUs: {args.gpu_id}")
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_id))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # Ensure consistent GPU ordering
    if args.log_debug:
        os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
    if args.model == 'qwen-3-coder':
        os.environ['VLLM_USE_DEEP_GEMM'] = '1'

    if args.model in ('qwen-3-coder-next', 'deepseek-v4-flash'):
        dtype = 'auto'
    else:
        dtype = 'bfloat16'

    cmd = ['vllm', 'serve']
    if args.model_path is not None:
        cmd += [args.model_path, '--served-model-name', MODELS[args.model]]
    else:
        cmd += [MODELS[args.model]]

    cmd += [
        '--tensor-parallel-size', str(len(args.gpu_id)) if args.gpu_id is not None else '1',
        '--dtype', dtype,
        '--max-model-len', str(args.max_tokens),
        '--gpu-memory-utilization', str(args.gpu_mem_utilization),
        '--host', '0.0.0.0',
        '--port', str(args.port),
        '--max-num-seqs', '64',
        '--trust-remote-code',
        '--enable-prefix-caching',
        '--enable-expert-parallel',
        '--enable-chunked-prefill',
    ]
    if args.model == 'deepseek-v4-flash':
        cmd += ['--kv-cache-dtype', 'fp8']
    elif args.model == 'qwen-3-coder':
        cmd += ['--quantization', 'fp8']
    if args.enable_tool:
        cmd += ['--enable-auto-tool-choice']
        if 'qwen' in args.model:
            cmd += ['--tool-call-parser', 'qwen3_coder']
        if 'glm' in args.model:
            cmd += ['--tool-call-parser', 'glm45', '--reasoning-parser', 'glm45']
        if 'minimax' in args.model:
            cmd += ['--reasoning-parser', 'minimax_m3',
                    '--tool-call-parser', 'minimax_m3']

    logger.info(f"Starting LLM server with command: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"LLM server exited with code {res.returncode}")
    else:
        logger.info("LLM server exited successfully")