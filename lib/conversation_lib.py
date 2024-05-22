# %%
import json
import os
import pickle

from typing import List, Optional, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt_lib import generate_answer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%


class Conversation:
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        system_prompt="You are a helpful honest assistant",
        system_message_neccecary=False,
    ):
        if system_message_neccecary:
            self.messages = [{"content": system_prompt, "role": "system"}]
        else:
            self.messages = []
        self.model = model
        self.tokenizer = tokenizer
        # self.cache = None
        self.model_name = None

        if self.model and self.tokenizer:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the model cache with the system prompt."""
        # tokens = self._tokenize_messages(add_generation_prompt=False)
        # self.cache = self.model.forward(
        #    input_ids=tokens,
        #    return_dict=True,
        # ).past_key_values
        self.model_name = self.model.config._name_or_path

    def _tokenize_messages(self, add_generation_prompt: bool, conversation=None):
        """Tokenize the current conversation messages."""
        if conversation is None:
            conversation = self.messages
        return self.tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        ).to(device)

    def add_conversation(self, conversation: List[Dict[str, str]]):
        """Add a conversation to the current messages and update the model cache."""
        self.messages += conversation
        # if self.model:
        #    tokens = self._tokenize_messages(
        #        add_generation_prompt=False, conversation=conversation
        #    )
        #    self.cache = self.model.forward(
        #        input_ids=tokens,
        #        return_dict=True,
        #        past_key_values=self.cache,
        #    ).past_key_values

    def prompt_llama(self, prompt: str, keep_answer=True, temperature=0.0):
        """Add a user prompt, generate a response, and update the conversation."""
        self.messages.append({"content": prompt, "role": "user"})
        tokens = self._tokenize_messages(add_generation_prompt=True)
        n_tokens = tokens.shape[1]

        if temperature == 0.0:
            do_sample = False
        else:
            do_sample = True
        out = self.model.generate(
            input_ids=tokens,
            max_length=n_tokens + 200,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
        )

        # self.cache = out.past_key_values

        response = self.tokenizer.decode(out[0, n_tokens:], skip_special_tokens=True)
        if keep_answer:
            self.messages.append(
                {"content": response, "role": "assistant", "origin": self.model_name}
            )
        else:
            self.messages = self.messages[:-1]
        return response

    def prompt_gpt(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        # Add the user prompt to the conversation
        attend_messages = self.messages.copy()
        attend_messages.append({"role": "user", "content": prompt})

        # Generate answer using OpenAI API
        response = generate_answer(attend_messages, model)

        # Add the assistant's response to the conversation
        self.add_conversation(
            [
                {
                    "content": prompt,
                    "role": "user",
                    "origin": "human",
                },
                {
                    "content": response,
                    "role": "assistant",
                    "origin": model,
                },
            ]
        )

        return response

    def print_conversation(self):
        """Print the current conversation messages."""
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")

    def save_conversation(self, folder_name: str):
        """Save the current conversation messages and cache to a folder."""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # Save messages
        messages_path = os.path.join(folder_name, "messages.json")
        with open(messages_path, "w") as f:
            json.dump(self.messages, f, indent=4)
        # Save cache if it exists
        # if self.cache is not None:
        #    cache_path = os.path.join(folder_name, "cache.pkl")
        #    with open(cache_path, "wb") as f:
        #        pickle.dump(self.cache, f)
        print(f"Conversation saved to {folder_name}")

    def load_conversation(self, folder_name: str):
        """Load conversation messages and cache from a folder."""
        messages_path = os.path.join(folder_name, "messages.json")
        cache_path = os.path.join(folder_name, "cache.pkl")

        if os.path.exists(messages_path):
            with open(messages_path, "r") as f:
                self.messages = json.load(f)
        else:
            raise FileNotFoundError(f"{messages_path} not found.")

        # if os.path.exists(cache_path):
        #    with open(cache_path, "rb") as f:
        #        self.cache = pickle.load(f)
        if self.model and self.tokenizer:

            self._initialize_model()
        print(f"Conversation loaded from {folder_name}")


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
    middle_conversation = [
        {"content": "What is the capital of France?", "role": "user"},
        {"content": "Paris", "role": "assistant"},
        {"content": "What is the population of Paris?", "role": "user"},
        {
            "content": "The population of Paris is approximately 2.16 million people.",
            "role": "assistant",
        },
        {"content": "Can you tell me a famous landmark in Paris?", "role": "user"},
        {
            "content": "A famous landmark in Paris is the Eiffel Tower.",
            "role": "assistant",
        },
        {"content": "Who designed the Eiffel Tower?", "role": "user"},
        {
            "content": "The Eiffel Tower was designed by Gustave Eiffel.",
            "role": "assistant",
        },
        {"content": "When was the Eiffel Tower built?", "role": "user"},
        {
            "content": "The Eiffel Tower was built between 1887 and 1889.",
            "role": "assistant",
        },
        {"content": "What is another famous museum in Paris?", "role": "user"},
        {
            "content": "Another famous museum in Paris is the Louvre Museum.",
            "role": "assistant",
        },
        {"content": "What is the Louvre Museum known for?", "role": "user"},
        {
            "content": "The Louvre Museum is known for housing the Mona Lisa.",
            "role": "assistant",
        },
        {"content": "How many visitors does the Louvre get annually?", "role": "user"},
        {
            "content": "The Louvre Museum receives over 9 million visitors annually.",
            "role": "assistant",
        },
        {"content": "What is the River that runs through Paris?", "role": "user"},
        {
            "content": "The river that runs through Paris is the Seine.",
            "role": "assistant",
        },
        {"content": "Can you name a famous cathedral in Paris?", "role": "user"},
        {
            "content": "A famous cathedral in Paris is Notre-Dame Cathedral.",
            "role": "assistant",
        },
    ]

    conv.add_conversation(middle_conversation)
# %%
if __name__ == "__main__":
    conv_folder = "conversation"
    conv.save_conversation(conv_folder)

# %%
if __name__ == "__main__":

    loaded_conv_1 = Conversation(model, tokenizer)
    loaded_conv_1.load_conversation(conv_folder)
    loaded_conv_1.prompt_llama("Please summarize the conversation so far.")
    loaded_conv_1.print_conversation()


# %%
if __name__ == "__main__":
    loaded_conv_2 = Conversation()
    loaded_conv_2.load_conversation(conv_folder)
    loaded_conv_2.prompt_gpt("Please summarize the conversation so far.")
    loaded_conv_2.print_conversation()

# %%
