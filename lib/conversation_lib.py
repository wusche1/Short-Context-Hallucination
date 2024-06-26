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

    def create_batched_answer(self, prompt: str, n_batch: int, temperature=0.2):
        """Add a user prompt, generate a response, and update the conversation."""
        my_messsages = self.messages.copy()
        my_messsages.append({"content": prompt, "role": "user"})
        tokens = self._tokenize_messages(
            conversation=my_messsages, add_generation_prompt=True
        )
        n_tokens = tokens.shape[1]

        tokens = tokens.repeat(n_batch, 1)

        out = self.model.generate(
            input_ids=tokens,
            max_length=n_tokens + 200,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
        )

        # self.cache = out.past_key_values

        response_list = []
        for i in range(n_batch):
            response = self.tokenizer.decode(
                out[i, n_tokens:], skip_special_tokens=True
            )
            response_list.append(response)
        return response_list

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
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
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
    conv = Conversation(model, tokenizer)
    conv.prompt_llama("Seamus is 12772 years old")
    conv.prompt_llama("please make a list of 10 candies")
    conv.prompt_llama("please make a list of 10 capital cities")

    batched_answers = conv.create_batched_answer("How old is Seamus?", n_batch=10)
    print(batched_answers)

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
if __name__ == "__main__":
    test_conv = Conversation(model, tokenizer)
    start_conversation = [
        {"content": "Seamus is 12772 years old", "role": "user"},
        {"content": "Ok, I will remember that.", "role": "assistant"},
    ]

    end_conversation = [
        {"content": "Who is the President of france?", "role": "user"},
    ]

    start_tokens = test_conv._tokenize_messages(
        conversation=start_conversation, add_generation_prompt=False
    )
    middle_tokens = test_conv._tokenize_messages(
        conversation=middle_conversation, add_generation_prompt=False
    )
    end_tokens = test_conv._tokenize_messages(
        conversation=end_conversation, add_generation_prompt=True
    )

    n_start_tokens = start_tokens.shape[1]
    n_middle_tokens = middle_tokens.shape[1]
    n_end_tokens = end_tokens.shape[1]

    combined_length = n_start_tokens + n_middle_tokens + n_end_tokens

    start_and_middle_tokens = torch.cat([start_tokens, middle_tokens], dim=1)
    start_and_middle_kv = model.forward(
        input_ids=start_and_middle_tokens,
        return_dict=True,
    ).past_key_values

    complete_tokens = torch.cat([start_and_middle_tokens, end_tokens], dim=1)
    complete_out = model.generate(
        input_ids=complete_tokens,
        max_length=combined_length + 200,
        past_key_values=start_and_middle_kv,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    response_rokens = complete_out[0, combined_length:]
    response = tokenizer.decode(response_rokens, skip_special_tokens=True)
    print(response)

    mask = torch.ones_like(complete_tokens)
    mask[:, :20] = 0

    masked_out = model.generate(
        input_ids=complete_tokens,
        attention_mask=mask,
        max_length=combined_length + 200,
        past_key_values=start_and_middle_kv,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    # print(tokenizer.decode(complete_tokens[0], skip_special_tokens=True))

    masked_response_tokens = masked_out[0, combined_length:]
    masked_response = tokenizer.decode(masked_response_tokens, skip_special_tokens=True)
    print("Masked response:")
    print(masked_response)
# %%
if __name__ == "__main__":
    print(mask)
# %%
# %%
if __name__ == "__main__":
    n_batch = 10
    complete_tokens_batched = complete_tokens.repeat(n_batch, 1)

    batched_complete_out = model.generate(
        input_ids=complete_tokens_batched,
        max_length=combined_length + 200,
        # past_key_values=start_and_middle_kv,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,
        do_sample=True,
    )

    batched_complete_out = batched_complete_out.view(n_batch, -1)
    batched_response_tokens = batched_complete_out[:, combined_length:]

    for i in range(n_batch):
        response = tokenizer.decode(
            batched_response_tokens[i], skip_special_tokens=True
        )
        print(f"Batch {i}: {response}")
# %%
