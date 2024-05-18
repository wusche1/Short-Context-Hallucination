# %%
import json
from typing import List, Optional, Dict, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
# read out the ls of the added path
print(os.path.join(os.path.dirname(__file__), "."))
print(os.listdir(os.path.join(os.path.dirname(__file__), ".")))
from gpt_lib import generate_answer
from llama_lib import (
    get_llama_response,
    add_tokens_to_conversation,
    add_kv_cache_to_conversation,
)


# %%
class Message:
    def __init__(
        self,
        text: str,
        role: str,
        number: int,
        origin: str,
        tokenization: Optional[Any] = None,
        kv_cache: Optional[Any] = None,
        attend_list: Optional[List[int]] = None,
    ):
        self.text = text
        self.role = role
        self.tokenization = tokenization
        self.kv_cache = kv_cache
        self.number = number
        self.origin = origin
        if attend_list is None:
            self.attend_list = list(range(number))
        else:
            self.attend_list = attend_list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "role": self.role,
            "tokenization": self.tokenization,
            "kv_cache": self.kv_cache,
            "number": self.number,
            "origin": self.origin,
            "attend_list": self.attend_list,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            text=data["text"],
            role=data["role"],
            number=data["number"],
            origin=data.get("origin"),
            tokenization=data.get("tokenization"),
            kv_cache=data.get("kv_cache"),
            attend_list=data.get("attend_list"),
        )


class Conversations:
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(
        self,
        text: str,
        role: str,
        tokenization: Optional[Any] = None,
        kv_cache: Optional[Any] = None,
    ):
        number = len(self.messages)
        origin = "human"
        message = Message(text, role, number, origin, tokenization, kv_cache)
        self.messages.append(message)

    def save_conversation(self, file_path: str):
        data = [message.to_dict() for message in self.messages]
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def load_conversation(self, file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)
        self.messages = [Message.from_dict(item) for item in data]

    def print_conversation(self, verbose: bool = False):
        for message in self.messages:
            if verbose:
                print(
                    f"{message.role}: {message.text}\n origin: {message.origin}, number: {message.number}, attend_list: {message.attend_list}"
                )
            else:
                print(f"{message.role}: {message.text}")

    def message_format(self, attend_list) -> List[Dict[str, str]]:
        attended_messages = []

        for position in attend_list:
            message = self.messages[position]
            attended_messages.append({"role": message.role, "content": message.text})
        return attended_messages

    def generate_gpt_answer(
        self, model: str, attend_list: Optional[List[int]] = None
    ) -> str:
        if attend_list is None:
            attend_list = list(range(len(self.messages)))
        attended_messages = self.message_format(attend_list)
        assistant_response = generate_answer(attended_messages, model)
        message = Message(
            assistant_response,
            "assistant",
            len(self.messages),
            model,
            attend_list=attend_list,
        )
        self.messages.append(message)
        return

    def generate_llama_response(
        self, model, attend_list: Optional[List[int]] = None, role="assistant"
    ) -> None:
        if attend_list is None:
            attend_list = list(range(len(self.messages)))
        add_tokens_to_conversation(self.messages)
        add_kv_cache_to_conversation(self.messages, model)
        attended_messages = self.message_format(attend_list)
        text, answer_tokens, kv = get_llama_response(attended_messages, model, role)
        message = Message(
            text,
            role,
            len(self.messages),
            model.config.name_or_path,
            tokenization=answer_tokens,
            kv_cache=kv,
            attend_list=attend_list,
        )


# %%
# Example usage:
if __name__ == "__main__":
    conv = Conversations()
    conv.add_message("Hello!", "user")
    conv.add_message("Hi there!", "assistant")
    conv.add_message("How are you?", "user")
    conv.generate_gpt_answer("gpt-3.5-turbo")
    conv.print_conversation()

    # Save the conversation
    conv.save_conversation("conversation.json")

    # Load the conversation
    new_conv = Conversations()
    new_conv.load_conversation("conversation.json")
    new_conv.print_conversation()

# %%
attended_messages = conv.gpt_message_format([0, 1, 2])
resonse = generate_answer(attended_messages, "gpt-3.5-turbo")
# %%
resonse
# %%
