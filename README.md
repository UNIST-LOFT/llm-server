# LLM-Server
A vLLM wrapper to use local LLMs easier.

## Supported Models

* qwen-3: Qwen/Qwen3-1.7B
* qwen-3-next: Qwen/Qwen3-Next-80B-A3B-Instruct
* llama-4-scout: meta-llama/Llama-4-Scout-17B-16E-Instruct
* llama-3-2: meta-llama/Llama-3.2-1B-Instruct

### How to add more models
Add a new model in `MODELS`.
A key is an alias for options in command, and a value is a model name in Hugging Face.

## Pre-requisites

* Python 3.10+
* Pytorch with CUDA support (if using GPU)

Anaconda or Miniconda is strongly recommended.

* Note: Qwen recommends Python 3.11+ for best performance.

Install dependencies:

```bash
pip install -r requirements.txt
```

:warning: Note: PyTorch is not included in requirements.txt. Please install PyTorch separately according to your system and CUDA version. See https://pytorch.org/get-started/locally/ to install locally.

### Pre-download models
We highly recommend to pre-download the model before running llm-server.

To download a model in CLI:
```sh
huggingface-cli download <model>
```

## How to run the server

Run the following command to start the server:

```bash
python ./run.py <options> <model_name>
```
Replace `<model_name>` with the desired model from the list below (e.g., `qwen-3-next`).

Options:
* `--gpu-id`: Space seperated list of GPU ids to use. Starts from 0. For example, `--gpu-id 0 1 2` to use first three GPUs. (default: all available GPUs)
* `--gpu-mem-utilization <float>`: GPU memory utilization used in vLLM. For example, `0.8` will use 80% of the GPU memory.
(default: 0.85)
* `-p`, `--port`: Port to run the server on (default: 5000)
* `--max-tokens`: Maximum tokens for model (prompt + response). Default: 8192
* `--log-debug`: Enable debug logging in vLLM

## Get a response (inference)
LLM-Server and vLLM uses OpenAI-style API.

For example, in Python:
```python
import openai
client = openai.OpenAI(base_url='http://localhost:5000/v1', api_key='') # Init OpenAI API client
```
where `base_url` is the url to LLM-Server.
We disable `api_key` to access local server now.

To inference,
```python
response = client.chat.completions.create(
  model='<model_name>',
  messages=[
    {'role': 'system', 'content': SYSTEM_MESSAGE},
    {'role': 'user', 'content': prompt},
  ],
  temperature=<temperature>,
  max_tokens=<max_tokens>,
)

print(response.choices[0].message.content) # Print response
```
where `<model_name>` is proper model name in Hugging Face, `<temperature>` is temperature and `<max_tokens>` is maximum tokens to inference (i.e., length of system msg and prompt).

Note that `<max_tokens>` does not include the response.
Thus, it should be smaller than `--max-tokens` used to run LLM-Server.