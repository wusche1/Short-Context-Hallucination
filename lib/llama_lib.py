# %%
import sys
import torch as t
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import Optional, List, Tuple

device = t.device("cuda" if t.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token


# %%
class CacheUtil:
    @staticmethod
    def put_cache_on_device(
        cache: Tuple[Tuple[t.Tensor]], device: str
    ) -> Tuple[Tuple[t.Tensor]]:
        """Move cache to the specified device if not already on it."""
        if cache[0][0].device != device:
            cache = tuple(
                tuple(kv.to(device) for kv in cache_layer) for cache_layer in cache
            )
        return cache

    @staticmethod
    def add_caches_along_seq_pos(
        caches: List[Tuple[Tuple[Tensor]]],
    ) -> Tuple[Tuple[Tensor]]:
        """Add multiple caches along the sequence position."""
        # Assuming all caches have the same structure and dimensions
        num_layers = len(caches[0])
        num_tensors_per_layer = len(caches[0][0])

        return tuple(
            tuple(
                t.cat(
                    [cache[layer_idx][tensor_idx] for cache in caches], dim=-2
                )  # Concatenating along the sequence dimension
                for tensor_idx in range(num_tensors_per_layer)
            )
            for layer_idx in range(num_layers)
        )

    @staticmethod
    def snip_first_n_pos_of_cache(cache: Tuple[Tuple[Tensor]], n: int) -> Tuple[Tensor]:
        """Select all but the first n seq positions of the cache."""
        return tuple(
            tuple(kv[:, :, n:, :] for kv in cache_layer) for cache_layer in cache
        )


def add_tokens_to_conversation(messages: List[dict]) -> None:
    for message in messages:
        if message.tokenization is None:
            message_text = f"<|start_header_id|>{message.role}<|end_header_id|>\n\n{message.text}<|eot_id|>"
            if message.number == 0:
                message_text = f"<|begin_of_text|>{message_text}"
            message.tokenization = tokenizer.encode(
                message_text, return_tensors="pt"
            ).to(device)
    return


def add_kv_cache_to_conversation(
    messages: List[dict], model: AutoModelForCausalLM
) -> None:
    for message in messages:
        if message.kv_cache is None:

            kv_input = []
            for i in message.attend_list:
                kv_input.append(messages[i].kv_cache)
            if len(kv_input) != 0:
                kv_input = CacheUtil.add_caches_along_seq_pos(kv_input)
                n_previous_tokens = kv_input[0][0].shape[2]
            else:
                kv_input = None
                n_previous_tokens = 0
            kv = model(message.tokenization, past_key_values=kv_input).past_key_values
            sniped_kv = CacheUtil.snip_first_n_pos_of_cache(kv, n_previous_tokens)
            message.kv_cache = sniped_kv


def get_llama_response(
    messages: List,
    model,
    attended_messages: Optional[List[int]] = None,
    role="assistant",
) -> None:
    add_tokens_to_conversation(messages)
    add_kv_cache_to_conversation(messages, model)
    tokens_input = []
    kv_input = []
    if attended_messages is None:
        attended_messages = list(range(len(messages)))
    for i in attended_messages:
        message = messages[i]
        tokens_input.append(message.tokenization)
        kv_input.append(message.kv_cache)
    message_start_tokens = tokenizer.encode(
        f"<|start_header_id|>{role}<|end_header_id|>\n\n", return_tensors="pt"
    ).to(device)
    tokens_input.append(message_start_tokens)
    tokens_input = t.cat(tokens_input, dim=-1)
    kv_input = CacheUtil.add_caches_along_seq_pos(kv_input)
    n_previous_tokens = kv_input[0][0].shape[2]

    response = model.generate(
        tokens_input,
        past_key_values=kv_input,
        max_length=len(tokens_input) + 1000,
        return_dict_in_generate=True,
        attention_mask=t.ones(tokens_input.shape).to(device),
    )

    answer_tokens = response.sequences[:, n_previous_tokens:]
    kv = CacheUtil.snip_first_n_pos_of_cache(
        response.past_key_values, n_previous_tokens
    )

    text = answer_tokens[:, len(message_start_tokens[0]) : -1]
    text = tokenizer.decode(text[0])

    return text, answer_tokens, kv


def get_llama_response_uncached(
    messages: List,
    model,
    attended_messages: Optional[List[int]] = None,
    role="assistant",
) -> None:
    add_tokens_to_conversation(messages)
    tokens_input = []
    if attended_messages is None:
        attended_messages = list(range(len(messages)))
    for i in attended_messages:
        message = messages[i]
        tokens_input.append(message.tokenization)

    n_previous_tokens = sum([message.tokenization.shape[1] for message in messages])

    message_start_tokens = tokenizer.encode(
        f"<|start_header_id|>{role}<|end_header_id|>\n\n", return_tensors="pt"
    ).to(device)
    tokens_input.append(message_start_tokens)
    tokens_input = t.cat(tokens_input, dim=-1)

    response = model.generate(
        tokens_input,
        max_length=len(tokens_input) + 1000,
        return_dict_in_generate=True,
        attention_mask=t.ones(tokens_input.shape).to(device),
    )

    answer_tokens = response.sequences[:, n_previous_tokens:]
    kv = CacheUtil.snip_first_n_pos_of_cache(
        response.past_key_values, n_previous_tokens
    )

    text = answer_tokens[:, len(message_start_tokens[0]) : -1]
    text = tokenizer.decode(text[0])

    return text, answer_tokens, kv


# %%
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = model.to(device)

# %%
if __name__ == "__main__":
    from conversation_lib import Message, Conversations

    conv = Conversations()
    conv.add_message("Hello!", "user")
    conv.add_message("Hi there!", "assistant")
    conv.add_message("How are you?", "user")

    messages = conv.messages

    add_tokens_to_conversation(messages)
    add_kv_cache_to_conversation(messages, model)
    text, resonse, kv = get_llama_response(messages, model)
    print("text:")
    print(text)
    print("resonse:")
    print(tokenizer.decode(resonse[0]))
    print("kv:")
    print(kv[0][0].shape)

# %%
if __name__ == "__main__":
    from conversation_lib import Message, Conversations

    conv = Conversations()
    conv.add_message("Hello!", "user")
    conv.add_message("Hi there!", "assistant")
    conv.add_message("How are you?", "user")

    messages = conv.messages

    add_tokens_to_conversation(messages)
    text, resonse, kv = get_llama_response_uncached(messages, model)
    print("text:")
    print(text)
    print("resonse:")
    print(tokenizer.decode(resonse[0]))
    print("kv:")
    print(kv[0][0].shape)
# %%
