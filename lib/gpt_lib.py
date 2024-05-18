# %%
import openai
import re
import sys
import numpy as np
import pandas as pd
import torch as t
import os
import json


from openai import OpenAI

# %%
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def generate_answer(attended_messages: list, model: str):
    stream = client.chat.completions.create(
        model=model,
        messages=attended_messages,
        stream=True,
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            # print(chunk.choices[0].delta.content, end="")

    return response


# %%
if __name__ == "__main__":
    attended_messages = [
        {
            "role": "system",
            "content": "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.",
        },
        {
            "role": "user",
            "content": "What is human life expectancy in the United States?",
        },
        {
            "role": "assistant",
            "content": "Human life expectancy in the United States is 78 years.",
        },
        {"role": "user", "content": "What is the average life expectancy in Brazil?"},
    ]

    response = generate_answer(attended_messages, "gpt-3.5-turbo")
    print(response)
# %%
