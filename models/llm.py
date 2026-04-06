from transformers import AutoModel, AutoTokenizer

class LLMWrapper:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        # Take last hidden state
        return outputs.last_hidden_state[:, -1, :]
