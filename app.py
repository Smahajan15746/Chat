import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the knowledge base
knowledge_base = pd.read_csv("knowledge_base.csv")

# Retrieval function
def retrieve_info(user_input):
    """
    Search the knowledge base for a matching query.
    Returns a relevant answer if found, otherwise None.
    """
    results = knowledge_base[knowledge_base['query'].str.contains(user_input, na=False, case=False)]
    if not results.empty:
        return results.iloc[0]['answer']
    return None

# Chatbot response function
def chatbot_response(user_input):
    """
    Generate a response by first attempting to retrieve information from the knowledge base.
    If no relevant data is found, use GPT-2 to generate a response.
    """
    # Step 1: Try retrieval from knowledge base
    retrieved_info = retrieve_info(user_input)
    if retrieved_info:
        return retrieved_info

    # Step 2: Generate response using GPT-2
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,  # Add randomness
        top_p=0.9,        # Use nucleus sampling
        repetition_penalty=1.2  # Penalize repetitive phrases
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio Interface
def gradio_interface(user_input):
    """
    Wrapper for the Gradio interface.
    Takes user input and returns the chatbot response.
    """
    return chatbot_response(user_input)

# Create Gradio app
interface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="AI Chatbot with RAG",
    description="A chatbot that combines knowledge base retrieval with GPT-2 generation for intelligent responses."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
