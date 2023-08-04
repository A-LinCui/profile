"""
First-token latency test for ChatGLM-6B.

Usage:
    python latency.py --input-length <INPUT_LENGTH>
"""

# pylint: disable=consider-using-f-string,invalid-name

import argparse
import random
import time

import torch
from transformers import AutoTokenizer, AutoModel


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "chatglm-6b")
    parser.add_argument("--input-length", type = int, required = True,
            help = "length of the first input token")
    parser.add_argument("--parallel", action = "store_true", help = "whether use two gpus")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    name = "THUDM/chatglm-6b"
    revision = "v1.1.0"

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code = True, revision = revision)

    if args.parallel:
        from utils import load_model_on_gpus
        model = load_model_on_gpus(name, num_gpus = 2, revision = revision)
    else:
        model = AutoModel.from_pretrained(name, trust_remote_code = True, revision = revision)
        model = model.half().to(device)

    input_ids = torch.Tensor([
        [random.randint(1, 5000) for i in range(args.input_length - 2)] + [130001, 130004]
    ]).to(device).int()

    print("Input ids: {}".format(input_ids))

    gen_kwargs = {"max_length": 1, "num_beams": 1, "do_sample": True, "top_p": 0.7,
                  "temperature": 0.95, "logits_processor": None}

    torch.cuda.synchronize()
    start_time = time.time()
    outputs = model.generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()
    latency = time.time() - start_time
    print("Output ids: {}".format(outputs))

    print("Token length: {}; Inference latency: {:.4f}s".format(args.input_length, latency))
