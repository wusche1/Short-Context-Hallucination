# %%
import json
import os

from typing import List, Optional, Dict, Any, Tuple
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt_lib import generate_answer

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# %%
class Conversations:
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.messages: List[Dict[str, Any]] = []
        self.tokenization: Optional[t.Tensor] = None
        self.cache: Optional[Tuple[Tuple[t.Tensor]]] = None
        self.message_number_mask: Optional[t.Tensor] = None
        self.model = model
        self.tokenizer = tokenizer
        if self.model is not None:
            self.model_name = self.model.config._name_or_path
        else:
            self.model_name = None

    def add_message(self, text: str, role: str, attend_list=None):
        number = len(self.messages)
        origin = "human"
        if attend_list is None:
            attend_list = list(range(number + 1))
        message = {
            "text": text,
            "role": role,
            "number": number,
            "origin": origin,
            "attend_list": attend_list,
            "cached": False,
        }
        self.messages.append(message)

    def save_conversation(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(self.messages, file, indent=4)
            t.save(
                self.tokenization,
                os.path.join(os.path.dirname(file_path), "tokenization.pt"),
            )
            t.save(self.cache, os.path.join(os.path.dirname(file_path), "cache.pt"))
            t.save(
                self.message_number_mask,
                os.path.join(os.path.dirname(file_path), "message_number_mask.pt"),
            )
            if self.model_name is not None:
                with open(
                    os.path.join(os.path.dirname(file_path), "model_name.txt"), "w"
                ) as model_file:
                    model_file.write(self.model_name)

    def load_conversation(self, file_path: str):
        with open(file_path, "r") as file:
            self.messages = json.load(file)
            self.tokenization = t.load(
                os.path.join(os.path.dirname(file_path), "tokenization.pt")
            )
            self.cache = t.load(os.path.join(os.path.dirname(file_path), "cache.pt"))
            self.message_number_mask = t.load(
                os.path.join(os.path.dirname(file_path), "message_number_mask.pt")
            )
            if os.path.exists(
                os.path.join(os.path.dirname(file_path), "model_name.txt")
            ):
                with open(
                    os.path.join(os.path.dirname(file_path), "model_name.txt"), "r"
                ) as model_file:
                    self.model_name = model_file.read()

    def print_conversation(self, verbose: bool = False):
        for message in self.messages:
            if verbose:
                print(
                    f"{message['role']}: {message['text']}\n origin: {message['origin']}, number: {message['number']}, attend_list: {message['attend_list']}"
                )
            else:
                print(f"{message['role']}: {message['text']}")

    def message_format(self, attend_list: List[int]) -> List[Dict[str, str]]:
        return [
            {"role": self.messages[pos]["role"], "content": self.messages[pos]["text"]}
            for pos in attend_list
        ]

    def generate_gpt_answer(
        self, model: str, attend_list: Optional[List[int]] = None
    ) -> str:
        if attend_list is None:
            attend_list = list(range(len(self.messages) + 1))
        attended_messages = self.message_format(attend_list)
        assistant_response = generate_answer(attended_messages, model)
        message = {
            "text": assistant_response,
            "role": "assistant",
            "number": len(self.messages),
            "origin": model,
            "attend_list": attend_list,
            "cached": False,
        }
        self.messages.append(message)

    def llama_syntaxed_message_tokens(self, message: Dict[str, Any]) -> t.Tensor:
        role = message["role"]
        text = message["text"]
        add_special_tokens = message["number"] == 0
        tokenization = t.tensor(
            self.tokenizer.encode(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>",
                add_special_tokens=add_special_tokens,
            )
        ).to(device)
        return tokenization

    def llama_beginning_of_text_tokens(self, role: str) -> t.Tensor:
        return t.tensor(
            self.tokenizer.encode(f"{role}\n\n", add_special_tokens=False)
        ).to(device)

    def get_attention_mask(
        self, message_number_mask: t.Tensor, attend_list: List[int]
    ) -> t.Tensor:
        if message_number_mask is None:
            return None
        attend_to_tensor = t.tensor(attend_list).to(device)
        attention_mask = message_number_mask.unsqueeze(1) == attend_to_tensor.unsqueeze(
            0
        )
        attention_mask = (
            attention_mask.any(dim=1).long().unsqueeze(0)
        )  # Add batch dimension here
        # print a warning if any entry in the attention mask is zero
        if attention_mask.any() == 0:
            print("Warning: Attention mask is all zeros")
            print(message_number_mask)
            print(attend_list)
            assert 1 == 0
        return attention_mask

    def conditional_tensor_concat(self, tensor_1, tensor_2):
        if tensor_1 is None:
            return tensor_2
        if tensor_2 is None:
            return tensor_1
        return t.cat([tensor_1, tensor_2], dim=-1)

    def add_message_to_cache(self, message: Dict[str, Any]):
        message_tokens = self.llama_syntaxed_message_tokens(message)
        message_number_mask = t.ones_like(message_tokens) * message["number"]
        self.message_number_mask = self.conditional_tensor_concat(
            self.message_number_mask, message_number_mask
        )
        self.tokenization = self.conditional_tensor_concat(
            self.tokenization, message_tokens
        )

        attention_mask = self.get_attention_mask(
            self.message_number_mask, message["attend_list"]
        )
        response = self.model(
            message_tokens.unsqueeze(0),  # Add batch dimension here
            past_key_values=self.cache,
            return_dict=True,
            attention_mask=attention_mask,
        )

        self.cache = response.past_key_values
        message["cached"] = True
        return

    def cache_all_messages(self):
        for message in self.messages:
            if not message["cached"]:
                self.add_message_to_cache(message)
        return

    def generate_llama_response(self, role="assistant", attend_list=None):
        if attend_list is None:
            attend_list = list(range(len(self.messages) + 1))
        attention_mask = self.get_attention_mask(self.message_number_mask, attend_list)
        self.cache_all_messages()
        message_start_tokens = self.llama_beginning_of_text_tokens(role)
        tokens_input = t.cat(
            [self.tokenization, message_start_tokens], dim=-1
        ).unsqueeze(
            0
        )  # Add batch dimension here

        i_response_start = tokens_input.shape[1]

        response = self.model.generate(
            tokens_input,
            past_key_values=self.cache,
            return_dict_in_generate=True,
            attention_mask=attention_mask,
            max_length=len(tokens_input) + 100,
        )
        response_text = self.tokenizer.decode(
            response.sequences[0, i_response_start:-1], skip_special_tokens=True
        )
        message = {
            "text": response_text,
            "role": role,
            "number": len(self.messages),
            "origin": self.model_name,
            "attend_list": attend_list,
            "cached": False,
        }
        self.messages.append(message)
        return


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
if __name__ == "__main__":
    import torch as t
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = model.to(device)
# %%
if __name__ == "__main__":
    conv = Conversations(model, tokenizer)
    conv.add_message("Hello!", "user")
    conv.add_message("Hi there!", "assistant")
    conv.add_message("How are you?", "user")
    conv.generate_llama_response()
    conv.print_conversation()
# %%
if __name__ == "__main__":
    for message in conv.messages:
        print(message["number"])
        print(message["attend_list"])

# %%
if __name__ == "__main__":
    conv = Conversations(model, tokenizer)
    conv.add_message("Hello!", "user")
    conv.add_message("Hi there!", "assistant")
    conv.add_message("How are you?", "user")
    conv.cache_all_messages()

    tokens = conv.tokenization
    new_message_beginning = conv.llama_beginning_of_text_tokens("assistant")
    tokens = t.cat([tokens, new_message_beginning], dim=-1).unsqueeze(0)

    i_new_message_begins = tokens.shape[1]

    generation = model.generate(
        tokens,
        return_dict_in_generate=True,
        max_length=len(tokens) + 100,
    )

    answer_tokens = generation.sequences[:, i_new_message_begins:]
# %%
if __name__ == "__main__":
    print(tokens.shape)
    print(generation.sequences.shape)
    print(answer_tokens.shape)
    print("####################")
    print(
        tokenizer.decode(
            generation.sequences[0, :i_new_message_begins], skip_special_tokens=False
        )
    )
    print("####################")
    print(
        tokenizer.decode(
            generation.sequences[0, i_new_message_begins:], skip_special_tokens=False
        )
    )
    print("####################")


# %%

# %%
if __name__ == "__main__":
    conv = Conversations(model, tokenizer)
    conv.add_message("Hello!", "user")
    conv.add_message("Hi there!", "assistant")
    conv.add_message("How are you?", "user")
    conv.cache_all_messages()

    tokens = conv.tokenization
    cache = conv.cache

    print(tokenizer.decode(tokens[:], skip_special_tokens=False))

    repica_cache = model(tokens.unsqueeze(0), return_dict=True).past_key_values

    for a, b in zip(cache, repica_cache):
        for c, d in zip(a, b):
            # see wether they are close
            print(t.allclose(c, d, atol=1e-4))
# %%
