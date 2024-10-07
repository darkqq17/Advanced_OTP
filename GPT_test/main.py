import torch
from transformers import AutoTokenizer, OpenAIGPTForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
model = OpenAIGPTForSequenceClassification.from_pretrained("openai-community/openai-gpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    print(logits)

predicted_class_id = logits.argmax().item()
print("Predicted class:", model.config.id2label[predicted_class_id])

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = OpenAIGPTForSequenceClassification.from_pretrained("openai-community/openai-gpt", num_labels=num_labels)

labels = torch.tensor([1])
print(labels)
loss = model(**inputs, labels=labels).loss