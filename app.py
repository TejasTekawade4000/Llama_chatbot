import gradio as gr
import time
from ctransformers import AutoModelForCasuallLM

def load_llm():
    llm = AutoModelForCasuallLM.from_pretrained(
        'codellama-13b-instruct.Q4_K_M.gguf',
        model_type = 'llama',
        max_new_tokens = 1096,
        repetation_penalty = 1.13,
        temperature = 0.1,
    )
    return llm

def llm_function(message, chat_history):
    llm = load_llm()
    response = llm(
        message
    )
    output_texts = response
    return output_texts

title ="Chatbot of LLM using llama"

examples = [
    'Write a python code to connetct with SQL database and list down all the tables.'
    'Write a python code to train a linear regression model using scikit learn.'
    'Write a code to implement a binary tree implementation in c language.'
    'What are the benefits of python progemming language.'
]

gr.ChatInterface(
    fn = llm_function,
    title = title,
    examples = examples,
).launch()