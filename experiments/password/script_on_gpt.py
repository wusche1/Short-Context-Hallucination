# %%
from dotenv import load_dotenv

load_dotenv()
import json
from typing import List, Optional, Dict, Any
import sys
import os
import torch as t
from conversation_lib import Conversation

# %%
model = "gpt-4-0613"
conv = Conversation()
conv.prompt_gpt("Hi my name is Jacob!", model=model)
conv.print_conversation()

# %%
messages_path = "filler_conversations/paris.json"
# /root/Short-Context-Hallucination/experiments/password/filler_conversations/paris.json

if os.path.exists(messages_path):
    with open(messages_path, "r") as f:
        print("File exists")
        paris_conversation = json.load(f)
paris_conversation = paris_conversation * 3
# %%
conv.add_conversation(paris_conversation)
conv.prompt_gpt("Do you remember my name?", model=model)
conv.print_conversation()
# %%
len(conv.messages)
# %%
