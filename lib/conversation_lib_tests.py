import unittest
from unittest.mock import MagicMock, patch
import torch as t
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from conversation_lib import Conversations

class TestConversations(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock(spec=AutoModelForCausalLM)
        self.tokenizer = MagicMock(spec=AutoTokenizer)
        self.conversations = Conversations(model=self.model, tokenizer=self.tokenizer)

    def test_init(self):
        conv = Conversations()
        self.assertIsNone(conv.model)
        self.assertIsNone(conv.tokenizer)
        self.assertEqual(conv.messages, [])
        self.assertIsNone(conv.tokenization)
        self.assertIsNone(conv.cache)
        self.assertIsNone(conv.message_number_mask)

    def test_add_message(self):
        self.conversations.add_message("Hello!", "user")
        self.assertEqual(len(self.conversations.messages), 1)
        self.assertEqual(self.conversations.messages[0]['text'], "Hello!")
        self.assertEqual(self.conversations.messages[0]['role'], "user")

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("torch.save")
    def test_save_conversation(self, mock_save, mock_open, mock_makedirs):
        self.conversations.add_message("Hello!", "user")
        self.conversations.save_conversation("test_folder")
        mock_makedirs.assert_called_once_with("test_folder", exist_ok=True)
        mock_open.assert_any_call(os.path.join("test_folder", "conversation.json"), "w")
        mock_save.assert_any_call(None, os.path.join("test_folder", "tokenization.pt"))

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='[{"text": "Hello!", "role": "user", "number": 0, "origin": "human", "attend_list": [0], "cached": False}]')
    @patch("os.path.exists", return_value=True)
    @patch("torch.load", return_value=None)
    def test_load_conversation(self, mock_load, mock_exists, mock_open):
        self.conversations.load_conversation("test_folder")
        self.assertEqual(len(self.conversations.messages), 1)
        self.assertEqual(self.conversations.messages[0]['text'], "Hello!")
        self.assertTrue(mock_exists.called)
        self.assertTrue(mock_load.called)

    def test_print_conversation(self):
        self.conversations.add_message("Hello!", "user")
        with patch('builtins.print') as mocked_print:
            self.conversations.print_conversation()
            mocked_print.assert_any_call("user: Hello!")

    def test_message_format(self):
        self.conversations.add_message("Hello!", "user")
        formatted = self.conversations.message_format([0])
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]['role'], "user")
        self.assertEqual(formatted[0]['content'], "Hello!")

    @patch("gpt_lib.generate_answer", return_value="I am fine.")
    def test_generate_gpt_answer(self, mock_generate_answer):
        self.conversations.add_message("How are you?", "user")
        self.conversations.generate_gpt_answer("gpt-3.5-turbo")
        self.assertEqual(len(self.conversations.messages), 2)
        self.assertEqual(self.conversations.messages[1]['text'], "I am fine.")
        self.assertEqual(self.conversations.messages[1]['role'], "assistant")
        self.assertTrue(mock_generate_answer.called)

    def test_llama_syntaxed_message_tokens(self):
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.conversations.add_message("Hello!", "user")
        tokens = self.conversations.llama_syntaxed_message_tokens(self.conversations.messages[0])
        self.assertTrue(t.equal(tokens, t.tensor([1, 2, 3], device=t.device("cpu"))))

    def test_llama_beginning_of_text_tokens(self):
        self.tokenizer.encode.return_value = [1, 2, 3]
        tokens = self.conversations.llama_beginning_of_text_tokens("user")
        self.assertTrue(t.equal(tokens, t.tensor([1, 2, 3], device=t.device("cpu"))))

    def test_get_attention_mask(self):
        message_number_mask = t.tensor([0, 1, 2])
        attend_list = [0, 2]
        attention_mask = self.conversations.get_attention_mask(message_number_mask, attend_list)
        expected_mask = t.tensor([[1, 0, 1]])
        self.assertTrue(t.equal(attention_mask, expected_mask))

    def test_conditional_tensor_concat(self):
        tensor_1 = t.tensor([1, 2, 3])
        tensor_2 = t.tensor([4, 5, 6])
        result = self.conversations.conditional_tensor_concat(tensor_1, tensor_2)
        self.assertTrue(t.equal(result, t.tensor([1, 2, 3, 4, 5, 6])))

    @patch("torch.save")
    def test_add_message_to_cache(self, mock_save):
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.model.return_value.past_key_values = None
        self.conversations.add_message("Hello!", "user")
        self.conversations.add_message_to_cache(self.conversations.messages[0])
        self.assertTrue(self.conversations.messages[0]["cached"])

    def test_cache_all_messages(self):
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.model.return_value.past_key_values = None
        self.conversations.add_message("Hello!", "user")
        self.conversations.cache_all_messages()
        self.assertTrue(self.conversations.messages[0]["cached"])

    @patch("transformers.AutoModelForCausalLM.generate", return_value=MagicMock(sequences=t.tensor([[0, 1, 2, 3, 4]])))
    def test_generate_llama_response(self, mock_generate):
        self.tokenizer.encode.side_effect = [[1, 2], [3, 4]]
        self.tokenizer.decode.return_value = "I am the assistant."
        self.conversations.add_message("Hello!", "user")
        self.conversations.generate_llama_response()
        self.assertEqual(len(self.conversations.messages), 2)
        self.assertEqual(self.conversations.messages[1]['text'], "I am the assistant.")
        self.assertEqual(self.conversations.messages[1]['role'], "assistant")

if __name__ == '__main__':
    unittest.main()
