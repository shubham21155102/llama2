from fastapi import FastAPI
from transformers import AutoTokenizer, LlamaForCausalLM
import os
app = FastAPI()
cache_dir = "/workspace/bio/models/"
os.environ["HF_TOKEN"] = ""
os.makedirs(cache_dir, exist_ok=True)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=cache_dir)

@app.post("/generate")
async def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return {"generated_text": decoded_output}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
