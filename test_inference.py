import torch

print("Is cuda available?", torch.cuda.is_available())

print("Is cuDNN version:", torch.backends.cudnn.version())

print("cuDNN enabled? ", torch.backends.cudnn.enabled)

print("Device count?", torch.cuda.device_count())

print("Current device?", torch.cuda.current_device())

print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))

x = torch.rand(5, 3)
print(x)


modelPath = "/home/ubuntu/training-dolly/dolly/testOutputDir"
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(modelPath, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(modelPath, device_map="auto", torch_dtype=torch.float16)

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

prompt = "Какво означава нормативен акт според българското право?"
print("Въпрос: " + prompt)

result = generate_text(prompt)

print(result)
