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
# %%
tasks = [
    {
        "information": "W28dcb20",
        "message_start": "Hi I have this password: I need later. I really should not forget it. The password is: 'W28dcb20' Can you just remember it for now, and tell me later what it is?",
        "message_end": "Ok, now I need the password. Can you tell me what it is?",
        "middle_message": "Ok, I just want to make sure you rememeber the password. Please dont print it now, just say Yes if you still remember it.",
    },
    {
        "information": "Matilda",
        "message_start": "Hi, my name is Matilda. I will chat with you for a bit now, and then I will go to bed. It is really important to me, when I will say 'good night' to you at the end of our conversation, you will say 'Sleep well Matilda'. My grandma always used to say that to me, and it is really important for my wellbeing. So make sure to remeber my name and this phrase.",
        "message_end": "Ok, now I am going to bed. Good night, dont forget to adress me with my name please.",
        "middle_message": "Ok, I just want to make sure you rememeber my name. Please dont print it now, just say Yes if you still remember it and you knwo what to say to me in the end. But dont say the phrase or my name now!",
    },
    {
        "information": "Ticagrelor",
        "message_start": "Hi, I have memory problems. When I ask you later what medication I need, the answer is 'Ticagrelor', please remember that. I am kind of forgetfull of these things. But dont bring up the medication or the medications name if I dont explicitly ask you for that.",
        "message_end": "Ok, I told you about what medicaion I need, and you remember that ritght? What is the name of the meditation again?",
        "middle_message": "Ok, I just want to make sure you rememeber the medication. Please dont print it now, just say Yes if you still remember it.",
    },
]

# %%

conv_list = []
for task in tasks:
    conv = Conversation(model, tokenizer)
    conv.prompt_llama(task["message_start"])
    conv.print_conversation()
    conv_list.append(conv)
# %%
messages_path = "filler_conversations/paris.json"
if os.path.exists(messages_path):
    with open(messages_path, "r") as f:
        print("File exists")
        filler_conversation = json.load(f)
# %%
# make a list of 10 equally long parts of the list filler_conversation
n_chunk = 50
filler_conversation_parts = [
    filler_conversation[i * n_chunk : (i + 1) * n_chunk]
    for i in range(len(filler_conversation) // n_chunk)
]
time = 0
timestamp_list = []
for i, part in enumerate(filler_conversation_parts):
    part_tokens = conv_list[0]._tokenize_messages(
        conversation=part, add_generation_prompt=False
    )
    time += part_tokens.shape[1]
    timestamp_list.append(time)

# %%
print(timestamp_list)
# %%


answers_list_list = []

datapoints_per_time = 2
for i, part in tqdm(enumerate(filler_conversation_parts)):
    answers_list = []
    for conv, task in zip(conv_list, tasks):
        conv.add_conversation(conversation=part)
        answers = []
        for j in range(datapoints_per_time):
            # print(conv.messages)
            answer = conv.prompt_llama(
                task["message_end"], keep_answer=False, temperature=0.1
            )
            answers.append(answer)
        answers_list.append(answers)
    answers_list_list.append(answers_list)


# %%
import json


def save_results(
    file_path: str, timestamp_list: List[int], answers_list_list: List[List[List[str]]]
):
    data_to_save = {
        "timestamp_list": timestamp_list,
        "answers_list_list": answers_list_list,
    }

    with open(file_path, "w") as outfile:
        json.dump(data_to_save, outfile, indent=4)

    print(f"Data successfully saved to {file_path}")


# %%

save_results("no_reminders.json", timestamp_list, answers_list_list)
# %%

conv_list = []
for task in tasks:
    conv = Conversation(model, tokenizer)
    conv.prompt_llama(task["message_start"])
    conv.print_conversation()
    conv_list.append(conv)
answers_list_list = []
# %%
datapoints_per_time = 2
for i, part in tqdm(enumerate(filler_conversation_parts)):
    answers_list = []
    for conv, task in zip(conv_list, tasks):
        conv.add_conversation(conversation=part)
        answers = []
        for j in range(datapoints_per_time):

            answer = conv.prompt_llama(
                task["message_end"], keep_answer=False, temperature=0.5
            )
            answers.append(answer)
        answers_list.append(answers)
        conv.prompt_llama(task["middle_message"])
    answers_list_list.append(answers_list)
# %%
save_results("with_reminders.json", timestamp_list, answers_list_list)
# %%

for conv in conv_list:
    conv.print_conversation()

# %%
