import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

try:
    client = OpenAI(
        api_key="",  # Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"{e}")