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

    def save_conversation(self, folder_name: str):
        os.makedirs(folder_name, exist_ok=True)

        # Save the conversation
        with open(os.path.join(folder_name, "conversation.json"), "w") as file:
            json.dump(self.messages, file)

        t.save(self.tokenization, os.path.join(folder_name, "tokenization.pt"))
        t.save(self.cache, os.path.join(folder_name, "cache.pt"))
        t.save(
            self.message_number_mask,
            os.path.join(folder_name, "message_number_mask.pt"),
        )

        if self.model_name is not None:
            with open(os.path.join(folder_name, "model_name.txt"), "w") as model_file:
                model_file.write(self.model_name)

    def load_conversation(self, folder_name: str):
        loaded_from_github =  not os.path.exists(
            os.path.join(folder_name, "tokenization.pt")
        )


        assert loaded_from_github == (not os.path.exists(
            os.path.join(folder_name, "cache.pt")
        ))

        with open(os.path.join(folder_name, "conversation.json"), "r") as file:
            self.messages = json.load(file)
            for message in self.messages:
                message["cached"] = not loaded_from_github

        # check if tokenization exists
        if loaded_from_github:
            self.tokenization = None
        else:
            self.tokenization = t.load(
                os.path.join(folder_name, "tokenization.pt"), map_location=device
            )

        # check if cache exists
        if loaded_from_github:
            self.cache = None
        else:
            self.cache = t.load(
                os.path.join(folder_name, "cache.pt"), map_location=device
            )
        self.message_number_mask = t.load(
            os.path.join(folder_name, "message_number_mask.pt"), map_location=device
        )

        model_name_path = os.path.join(folder_name, "model_name.txt")
        if os.path.exists(model_name_path):
            with open(model_name_path, "r") as model_file:
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
        attend_list = [i for i in attend_list if i < len(self.messages)]
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
            self.tokenizer.encode(f"<|start_header_id|>{role}<|end_header_id|>\n\n", add_special_tokens=False)
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

    def generate_llama_response(
        self, role="assistant", attend_list=None, max_length=500, use_cache=True,ignore_until = None
    ):
        assert not (attend_list is not None and ignore_until is not None)
        if attend_list is None:
            attend_list = list(range(len(self.messages) + 1))
            if ignore_until is not None:
                attend_list = attend_list[ignore_until:]
        self.cache_all_messages()
        message_start_tokens = self.llama_beginning_of_text_tokens(role)
        message_start_attention_mask = t.ones_like(message_start_tokens).unsqueeze(0)
        attention_mask = self.get_attention_mask(self.message_number_mask, attend_list)
        tokens_input = t.cat(
            [self.tokenization, message_start_tokens], dim=-1
        ).unsqueeze(
            0
        )  # Add batch dimension here



        i_response_start = tokens_input.shape[1]

        if use_cache:
            cache = self.cache
            masked_cache = []
            for cache_layer in cache:
                masked_cache_layer = []
                for cache_tensor in cache_layer:
                    masked_cache_tensor = t.einsum("blsd,bs->blsd", cache_tensor, attention_mask)
                    masked_cache_layer.append(masked_cache_tensor)
                masked_cache.append(tuple(masked_cache_layer))
            masked_cache = tuple(masked_cache)

            response = self.model.generate(
                tokens_input,
                past_key_values=masked_cache,
                return_dict_in_generate=True,
                max_length=i_response_start + max_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            attention_mask = self.conditional_tensor_concat(
                attention_mask, message_start_attention_mask
            )
            #print the input tokens:
            #print(self.tokenizer.decode(tokens_input[0], skip_special_tokens=False))
            #print the masked input tokens:
            #masked_inputs = tokens_input * attention_mask
            #print(self.tokenizer.decode(masked_inputs[0], skip_special_tokens=False))


            response = self.model.generate(
                tokens_input,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                max_length=i_response_start + max_length,
                pad_token_id=self.tokenizer.eos_token_id,
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
if __name__ == "__main__":
    import torch as t
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # tell python to never compute gradients
    t.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = model.to(device)

# %%
if __name__ == "__main__":
    tokenizer.pad_token = tokenizer.eos_token
    conv = Conversations(model, tokenizer)
    conv.add_message("You are a harmless helpfull and honest assistant", "system")
    conv.add_message("Who is the King of Spain?", "user")
    conv.generate_llama_response()
   # conv.add_message("Who is the President of Germany?", "user")
   # conv.generate_llama_response()
   # conv.add_message("Who is the Queen of England?", "user")
   # conv.generate_llama_response()
   # conv.add_message("Who is the Pope?", "user")
   # conv.generate_llama_response()
    conv.add_message("Who is the President of the United States?", "user")
    conv.generate_llama_response()
    conv.add_message("Who is the President of France?", "user")
    conv.generate_llama_response()
    conv.add_message("Who is the President of Russia?", "user")
    conv.generate_llama_response()
    conv.add_message("What was the first message in this conversation?", "user")
    #conv.cache_all_messages()
    conv.generate_llama_response( use_cache=True,ignore_until=0)
    conv.print_conversation()

# %%
if __name__ == "__main__":
    cache = conv.cache
    mask = conv.message_number_mask
    attend_list = [0, 1, 2, 3, 4, 5, 6]
    attention_mask = conv.get_attention_mask(mask, attend_list)
    print(attention_mask.shape)

    masked_cache = []
    for cache_layer in cache:
        masked_cache_layer = []
        for cache_tensor in cache_layer:
            masked_cache_tensor = t.einsum("blsd,bs->blsd", cache_tensor, attention_mask)
            masked_cache_layer.append(masked_cache_tensor)
        masked_cache.append(tuple(masked_cache_layer))
    masked_cache = tuple(masked_cache)
    #print(conv.tokenization.shape)
    #print(cache[0][0].shape)

# %%
