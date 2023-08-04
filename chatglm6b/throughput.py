"""
Throughput rate test for ChatGLM-6B.
"""

# pylint: disable=consider-using-f-string,no-member,invalid-name,missing-function-docstring

import os
import time
import platform
import signal

import torch
from transformers import AutoTokenizer, AutoModel


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame): # pylint: disable=unused-argument,redefined-outer-name
    global stop_stream # pylint: disable=global-statement
    stop_stream = True


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

name = "THUDM/chatglm-6b"
revision = "v1.1.0"

os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stop_stream = False

tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code = True, revision = revision)
model = AutoModel.from_pretrained(name, trust_remote_code = True, revision = revision)
model = model.half().to(device)


def main():
    history = []
    global stop_stream # pylint: disable=global-statement

    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break

        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue

        count = 0
        start_time = time.time()
        for response, history in model.stream_chat(tokenizer, query, history = history):
            if stop_stream:
                stop_stream = False
                break

            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(build_prompt(history), flush = True)
                signal.signal(signal.SIGINT, signal_handler)

        infer_time = time.time() - start_time
        os.system(clear_command)
        print("QPS is ", len(response) / infer_time) # pylint: disable=undefined-loop-variable
        print(build_prompt(history), flush = True)


if __name__ == "__main__":
    main()
