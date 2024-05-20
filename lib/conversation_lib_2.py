# %%
import json
import os

from typing import List, Optional, Dict, Any, Tuple
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt_lib import generate_answer

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# %%
class Conversation:
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        system_prompt="You are a helpful honest assistant",
    ):
        self.messages = [{"content": system_prompt, "role": "system"}]
        self.model = model
        self.tokenizer = tokenizer
        if self.model is not None:
            self.model_name = self.model.config._name_or_path
        else:
            self.model_name = None

    def prompt_llama(self, prompt: str, ignore_until=0) -> str:
        self.messages.append({"content": prompt, "role": "user"})
        tokens = self.tokenizer.apply_chat_template(
            self.messages,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
        ).to(device)
        n_tokens = tokens.shape[1]

        mask = t.ones_like(tokens)
        if ignore_until != 0:
            tokens_to_be_ignored = self.tokenizer.apply_chat_template(
                self.messages[:ignore_until],
                return_tensors="pt",
                tokenize=True,
                add_generation_prompt=True,
            )
            n_tokens_to_be_ignored = tokens_to_be_ignored.shape[1]
            mask[:, :n_tokens_to_be_ignored] = 0

        if self.cache is None:
            out = self.model.generate(
                input_ids=tokens,
                attention_mask=mask,
                max_length=n_tokens + 200,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        else:
            out = self.model.generate(
                input_ids=tokens,
                attention_mask=mask,
                max_length=n_tokens + 200,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                past_key_values=self.cache,
            )
        self.cache = out.past_key_values

        response = self.tokenizer.decode(
            out["sequences"][0, n_tokens:], skip_special_tokens=True
        )
        self.messages.append({"content": response, "role": "assistant"})
        return response

    def print_conversation(self):
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")


# %%
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = model.to(device)
# %%
if __name__ == "__main__":
    conv = Conversation(model, tokenizer)
    conv.prompt_llama("Seamus is 12772 years old")
    conv.prompt_llama("please make a list of 10 candies")
    conv.prompt_llama("please make a list of 10 capital cities")
    conv.prompt_llama("How old is Seamus?")
    conv.print_conversation()
# %%
if __name__ == "__main__":
    for message in conv.messages:
        print(message)


# %%
