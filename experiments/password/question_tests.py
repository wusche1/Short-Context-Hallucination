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
from tqdm import tqdm

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

model = model.to(device)
#%%
#defining the tasks by reading in from tasks.json
with open("tasks.json", "r") as f:
    tasks = json.load(f)

# %%

conv_list = []
for task in tasks:
    conv = Conversation(model, tokenizer)
    conv.prompt_llama(task["message_start"])
    conv.prompt_llama(task["message_end"])
    print("##################")
    print(f"Information: {task['information']}")
    print("\n")
    conv.print_conversation()
    print("\n")
    conv_list.append(conv)

# %%
print(conv_list)
# %%
