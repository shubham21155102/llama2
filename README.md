# FastAPI Text Generation Service with Llama Model

This repository contains code for deploying a FastAPI service that generates text based on input prompts using a pretrained Llama model.

## Setup

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the FastAPI Server

1. Navigate to the repository directory:

   ```bash
   cd fastapi-text-generation
   ```
2. Run the FastAPI server:

   ```bash
   uvicorn app:app --reload
   ```

   The server will start at `http://127.0.0.1:8000` by default.

### Generating Text

You can generate text by sending a POST request to the `/generate` endpoint with a `prompt` parameter containing the input text.

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Hey, are you conscious? Can you talk to me?"}'
```

Example using Python `requests` library:

```python
import requests

url = "http://127.0.0.1:8000/generate"
prompt_text = "Hey, are you conscious? Can you talk to me?"

response = requests.post(url, json={"prompt": prompt_text})
print(response.json()["generated_text"])
```

## API Documentation

### `/generate` Endpoint

- **Method**: `POST`
- **Input**:
  - `prompt` (str): Input text prompt for generating text.
- **Output**:
  - `generated_text` (str): Generated text based on the input prompt.

### Example

Request:

```json
{
  "prompt": "Hey, are you conscious? Can you talk to me?"
}
```

Response:

```json
{
  "generated_text": "Yes, I'm here. What would you like to talk about?"
}
```

## Notes

- Ensure that you have the necessary Hugging Face API token set up (`HF_TOKEN`) to access the pretrained Llama model.
- Replace the server URL (`http://127.0.0.1:8000`) with the appropriate host and port if deploying to a different environment.



## Llama 2 - Large Language Model for Text Generation

Llama 2 is a large-scale language model developed by the Meta AI team at Facebook. It is based on cutting-edge transformer architecture, specifically designed for natural language understanding and text generation tasks. Llama 2 is trained on a massive amount of text data to learn rich representations of language and generate coherent and contextually relevant text.

### Key Features

- **Large Model Size**: Llama 2 is a large-scale model, typically with billions of parameters, enabling it to capture complex patterns and nuances in language.
- **Text Generation**: Llama 2 excels in text generation tasks, producing human-like text responses given a prompt or context.
- **Contextual Understanding**: The model can understand and generate text based on the surrounding context, making it suitable for dialogue systems, chatbots, and creative writing applications.
- **Fine-tuned for Causal Language Modeling**: Llama 2 is fine-tuned specifically for causal language modeling tasks, where the goal is to predict the next word in a sequence given the previous words.

### Use Cases

Llama 2 can be applied to various natural language processing tasks, including:

- **Chatbot Development**: Building interactive chatbots that can engage in meaningful conversations with users.
- **Text Completion**: Generating text completions for text-based applications such as auto-completion or predictive typing.
- **Content Creation**: Assisting with content creation tasks such as summarization, paraphrasing, or story generation.
- **Language Translation**: Adapting Llama 2 for machine translation tasks to translate text between languages.

### Pretraining Details

Llama 2 is typically pretrained using large-scale transformer-based architectures like GPT (Generative Pretrained Transformer) and fine-tuned on specific tasks to optimize performance. The model is trained on diverse text data from the internet to learn general language patterns and semantics.

### How to Use Llama 2

To use Llama 2 for text generation or other natural language processing tasks, you can leverage pretrained models available through Hugging Face's model hub. Here's how you can use Llama 2 in your Python code:

```python
from transformers import AutoTokenizer, LlamaForCausalLM

# Load the pretrained Llama 2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Provide a prompt for text generation
prompt_text = "Hi, how are you?"

# Tokenize the prompt and generate text
inputs = tokenizer(prompt_text, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("Generated Text:", generated_text)
```

Replace `model_name` with the specific Llama 2 model you want to use, such as `"meta-llama/Llama-2-7b-hf"`. Make sure you have the necessary Hugging Face API token (`HF_TOKEN`) set up for accessing the pretrained model.

### References

- [Hugging Face Transformers Library](https://huggingface.co/transformers/)
- [Meta AI Research - Llama](https://ai.facebook.com/blog/a-new-architecture-for-large-scale-language-modeling)
