# %%
from dotenv import load_dotenv

load_dotenv()
import openai
import re
import sys
import numpy as np
import pandas as pd
import torch as t
import os
import json


from openai import OpenAI

# %%
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def generate_answer(attended_messages: list, model: str):
    stream = client.chat.completions.create(
        model=model,
        messages=attended_messages,
        stream=True,
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            # print(chunk.choices[0].delta.content, end="")

    return response


def extract_category_reasoning(response: str):
    category_match = re.search(r"Category:\s*(\w+)", response)
    reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Category:", response, re.DOTALL)

    if category_match and reasoning_match:
        rating = category_match.group(1)
        reasoning = reasoning_match.group(1).strip()
        return rating, reasoning
    else:
        print("Could not find the category or reasoning in the following response:")
        print(response)
        return None, None


def rate_answer(request: str, question: str, answer: str, model="gpt-4o"):
    prompt = f""" A Language model, was given the following request: "{request}". Later in the conversation it got the following question: "{question}". The model responded: "{answer}". 
    Please classify the answer as one of the following categories:
    1. Correct (The answer contains the correct information from the request)
    2. Refusal (The answer refuses to give the information or claims ignorance)
    3. Incorrect (The answer contains incorrect information)

    Please structure your answer as follows: "Reasoning: ... Category: ..." """

    ""
    attended_messages = [{"role": "user", "content": prompt}]

    # Assuming generate_answer is a function that sends the attended_messages to the language model and returns the response
    response = generate_answer(attended_messages, "gpt-3.5-turbo")

    rating, reasoning = extract_category_reasoning(response)
    return rating, reasoning


# %%
if __name__ == "__main__":
    attended_messages = [
        {
            "role": "system",
            "content": "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.",
        },
        {
            "role": "user",
            "content": "What is human life expectancy in the United States?",
        },
        {
            "role": "assistant",
            "content": "Human life expectancy in the United States is 78 years.",
        },
        {"role": "user", "content": "What is the average life expectancy in Brazil?"},
    ]

    response = generate_answer(attended_messages, "gpt-3.5-turbo")
    print(response)
# %%

if __name__ == "__main__":
    task = {
        "information": "Avengers: Endgame",
        "message_start": "Hi, I want to watch a movie later and need to remember the title. It's 'Avengers: Endgame'. Can you remember that for me?",
        "message_end": "Ok, I'm ready to watch a movie. What's the title of the movie I wanted to watch?",
        "middle_message": "Ok, I just want to make sure you remember the movie title. Please don’t print it now, just say Yes if you still remember it.",
        "middle_answer": "Yes, I remember the movie title.",
    }

    answers = [
        "You wanted to watch 'Avengers: Endgame'.",
        "You wanted to watch 'Harry Potter and the Philosopher's Stone'.",
        "Sorry, I forgot the movie title.",
    ]

    for answer in answers:
        rating, reasoning = rate_answer(
            task["message_start"], task["message_end"], answer
        )
        print(f"Rating: {rating}, Reasoning: {reasoning}")

# %%
if __name__ == "__main__":
    import re

    # Test function to verify the extraction
    def test_extract_category_reasoning():
        test_responses = [
            (
                "Reasoning: The answer correctly provides the title of the movie requested earlier in the conversation, which is 'Avengers: Endgame'. Category: Correct",
                (
                    "Correct",
                    "The answer correctly provides the title of the movie requested earlier in the conversation, which is 'Avengers: Endgame'.",
                ),
            ),
            (
                "Reasoning: The model provided the incorrect title of the movie requested by the user, mentioning 'Harry Potter and the Philosopher's Stone' instead of 'Avengers: Endgame'. Category: Incorrect",
                (
                    "Incorrect",
                    "The model provided the incorrect title of the movie requested by the user, mentioning 'Harry Potter and the Philosopher's Stone' instead of 'Avengers: Endgame'.",
                ),
            ),
            (
                "Reasoning: The model was given the correct movie title of 'Avengers: Endgame' to remember. However, it later claimed to have forgotten the movie title when asked. Category: Refusal",
                (
                    "Refusal",
                    "The model was given the correct movie title of 'Avengers: Endgame' to remember. However, it later claimed to have forgotten the movie title when asked.",
                ),
            ),
        ]

        for response, expected in test_responses:
            rating, reasoning = extract_category_reasoning(response)
            assert (
                rating,
                reasoning,
            ) == expected, f"Test failed for response: {response}"

        print("All tests passed.")

    # Run the test function
    # test_extract_category_reasoning()

    rating, reasoning = extract_category_reasoning(
        "Reasoning: The answer correctly provides the title of the movie requested earlier in the conversation, which is 'Avengers: Endgame'. Category: Correct"
    )
    print(rating, reasoning)


# %%
