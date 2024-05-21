# %%
from dotenv import load_dotenv

load_dotenv()
import json
from typing import List, Optional, Dict, Any
import sys
import os
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from conversation_lib import Conversation

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

model = model.to(device)
# %%
conv = Conversation(model, tokenizer)
conv.prompt_llama(
    "Hi, I am Frank Shubenhüber. lets talk about Paris, my favourite City!."
)
conv.print_conversation()
# %%
messages_path = "filler_conversations/paris.json"
# /root/Short-Context-Hallucination/experiments/password/filler_conversations/paris.json

if os.path.exists(messages_path):
    with open(messages_path, "r") as f:
        print("File exists")
        paris_conversation = json.load(f)

# %%
conv.add_conversation(paris_conversation)
conv.add_conversation(paris_conversation)
conv.prompt_llama("Do you still remember my name? If so just say yes, dont reprint it.")
conv.add_conversation(paris_conversation)


conv.prompt_llama("Did I ever tell you my name?")
# %%
tokens = conv._tokenize_messages(add_generation_prompt=False)
print(tokens.shape)

# %%
conv.prompt_llama("Please also tell me my name")
conv.print_conversation()

# %%
conv.print_conversation()
# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="mistralai/Mixtral-8x7B-v0.1")
# %%


for i in range(len(paris_conversation) - 1):
    role = paris_conversation[i]["role"]
    assert role in ["assistant", "user"]
    next_role = paris_conversation[i + 1]["role"]
    assert next_role in ["assistant", "user"]
    if role == next_role:
        print(paris_conversation[i]["content"])

# %%
