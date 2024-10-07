from transformers import AutoModel

model = AutoModel.from_pretrained('bigscience/mt0-large')
for name, module in model.named_modules():
    print(name)
