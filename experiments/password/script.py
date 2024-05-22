# %%
import json
import os
from typing import List, Optional, Dict, Any
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from conversation_lib import Conversation
from tqdm import tqdm


# %%
def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None


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


def create_timestamps(conv_list, filler_conversation, n_chunk):
    filler_conversation_parts = [
        filler_conversation[i * n_chunk : (i + 1) * n_chunk]
        for i in range(len(filler_conversation) // n_chunk)
    ]
    time = 0
    timestamp_list = [time]
    for i, part in enumerate(filler_conversation_parts):
        part_tokens = conv_list[0]._tokenize_messages(
            conversation=part, add_generation_prompt=False
        )
        time += part_tokens.shape[1]
        timestamp_list.append(time)
    return timestamp_list, filler_conversation_parts


def generate_datapoints(
    conv_list, tasks, filler_conversation_parts, datapoints_per_time, with_reminders
):
    total_steps = len(filler_conversation_parts) * len(tasks) * datapoints_per_time
    progress_bar = tqdm(total=total_steps, unit="answer")

    answers_list_list = []
    for i, part in enumerate(filler_conversation_parts):
        answers_list = []
        for conv, task in zip(conv_list, tasks):
            conv.add_conversation(conversation=part)
            answers = []
            answers = conv.create_batched_answer(
                task["message_end"], datapoints_per_time, temperature=0.5
            )
            progress_bar.update(datapoints_per_time)
            answers_list.append(answers)
            if with_reminders:
                middle_usr_part = task["middle_message"]
                middle_assistant_part = task["middle_answer"]
                middle_conversation_part = [
                    {"role": "usr", "message": middle_usr_part},
                    {"role": "assistant", "message": middle_assistant_part},
                ]
                conv.add_conversation(conversation=middle_conversation_part)
        answers_list_list.append(answers_list)

    progress_bar.close()
    return answers_list_list


def run_experiment(
    model,
    tokenizer,
    tasks,
    filler_conversation,
    datapoints_per_time,
    output_file,
    n_chunk=250,
    with_reminders=False,
):
    conv_list = []
    for task in tasks:
        conv = Conversation(model, tokenizer)
        conv.prompt_llama(task["message_start"])
        conv.print_conversation()
        conv_list.append(conv)

    timestamp_list, filler_conversation_parts = create_timestamps(
        conv_list, filler_conversation, n_chunk
    )

    answers_list_list = generate_datapoints(
        conv_list,
        tasks,
        filler_conversation_parts,
        datapoints_per_time,
        with_reminders,
    )

    save_results(output_file, timestamp_list, answers_list_list)


# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

model = model.to(device)
model.eval()
# %%
messages_path = "filler_conversations/paris.json"
task_path = "tasks.json"
# %%
filler_conversation = load_json_file(messages_path)
tasks = load_json_file(task_path)

datapoints_per_time = 6
n_chunk = 250
# %%

run_experiment(
    model,
    tokenizer,
    tasks,
    filler_conversation,
    datapoints_per_time,
    "no_reminders.json",
    with_reminders=False,
    n_chunk=n_chunk,
)
# %%
run_experiment(
    model,
    tokenizer,
    tasks,
    filler_conversation,
    datapoints_per_time,
    "with_reminders.json",
    with_reminders=True,
    n_chunk=n_chunk,
)
# %%
