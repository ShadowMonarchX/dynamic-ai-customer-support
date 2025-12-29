from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", device_map="auto", torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct").to("cuda")

prompt = "Hello, how can you help me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
