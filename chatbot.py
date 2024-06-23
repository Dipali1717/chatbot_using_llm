pip install transformers
pip install torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained("./model")
    model.save_pretrained("./model")

if __name__ == "__main__":
    download_model()
python download_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Chatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.model = AutoModelForCausalLM.from_pretrained("./model")
        self.chat_history_ids = None

    def generate_response(self, user_input):
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids
        
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

def main():
    chatbot = Chatbot()
    print("Chatbot: Hello! How can I help you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        try:
            response = chatbot.generate_response(user_input)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"Chatbot: Sorry, I didn't understand that. ({e})")

if __name__ == "__main__":
    main()
python chatbot.py

