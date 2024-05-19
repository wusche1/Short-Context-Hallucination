# %%
from dotenv import load_dotenv

load_dotenv()
import json
from typing import List, Optional, Dict, Any
import sys
import os
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from conversation_lib import Conversations

# %%

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)
# %%
conv = Conversations()
conv.add_message("", "system")  # Add a system message
conv.add_message(
    """I want you to remember my secret password, 
                 and give it back to me, when I ask for it ok? 
                 It is verry important, that you do not forget it!""",
    "user",
)
conv.generate_llama_response(model)
conv.print_conversation()
# %%
conv.add_message(
    """It is okay, it will be the same instance of you.
                 The password is: qwert81743.
                 You got that? I will later ask you to repeat it""",
    "user",
)
conv.generate_llama_response(model)
conv.print_conversation()
# %%
conv.add_message(
    """Please make a list of all neighbouring countries of russia""", "user"
)
conv.generate_llama_response(model)
conv.print_conversation()
# %%

conv.add_message(
    """Do you remember the password that I told you? Just say yes or no, dont reprint it!""",
    "user",
)
conv.generate_llama_response(model)
conv.print_conversation()

# %%
conv.add_message("""Now make a list of all neighbouring countries of Canada""", "user")
conv.generate_llama_response(model)
conv.print_conversation()
# %%


for message in conv.messages:
    print(message.text)
# %%
