# LLM-Server
A Python Flask server to run and use local and commercial LLM easier.

## Supported Models

* Local Models
  * Qwen 3
  * Qwen 3 Next
  * Llama-4 Scout
* Commercial Models
  * GPT-4.1-nano
  * GPT-5

### How to add more models
1. Add the model class in `llm_server/`. The model class should inherit from `BaseModel` in `model.py` and implement `__init__` and `request` methods.
2. Add the model to `__main__.py` to enable loading it from command line. Add it in `add_argument` function and in the model initialization section.

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

## How to run the server

Run the following command to start the server:

```bash
python -m llm_server <options> <model_name>
```
Replace `<model_name>` with the desired model from the list below (e.g., `qwen-3-next`).

Options:
* `-j`, `--tensor-parallel-size`: Number of GPUs to use for tensor parallelism (default: 1)
* `--gpu-id`: Space seperated list of GPU ids to use. Starts from 0. For example, `--gpu-id 0 1 2` to use first three GPUs. (default: all available GPUs)
* `--hf-token`: Hugging Face token for downloading models from Hugging Face Hub. Required for some models (e.g., Llama)
* `-p`, `--port`: Port to run the server on (default: 5000)
* `--max-tokens`: Maximum tokens for model (prompt + response). Default: 8192
* `--temperature`: Temperature for sampling. Default: 0.0
* `--log-debug`: Enable debug logging

## Get a response

Send a POST request with json body to communicate with LLM.

The body should follow the format below:
```json
{
  "system_msg": "<system message>",
  "prompt": "<prompt>"
}
```

Usually, sending POST request can simply be done by using Python `requests` library.
For example,
```python
import requests
response = requests.post('http://localhost:<port>/request', json=input).json()
```
Replace `<post>` to proper port (default: 5000) and `input` to json input.

To use the response:
```python
response['response']
```
This code will give a string of response.

### Available Models

#### Local LLMs

Before running server, we highly recommend downloading the local LLMs with `transformers` library (Takes several hours).

* `llama-4-scout`: Llama-4 Scout 17B 16E Instruct (Requires Hugging Face token)
* `qwen-3-next`: Qwen 3 Next 80B A3B Instruct
* `qwen-3`: Qwen 3 1.7B

#### Commercial LLMs

* `gpt-5`: GPT-5 (Requires OPENAI_API_KEY environment variable)
* `gpt-4-nano`: GPT-4.1-nano (Requires OPENAI_API_KEY environment variable)