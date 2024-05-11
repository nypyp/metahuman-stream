import openai

openai.api_base = 'https://uiuiapi.com/v1'

class ChatGPT():
    def __init__(self, model_path = 'gpt-3.5-turbo', api_key = "sk-9xlLxBIZ06sVQE1rCaB9711fC7Df41Cf8b31F91d7f95609a"):
        openai.api_key = api_key
        self.model_path = model_path

    def chat(self, message):
        response = openai.ChatCompletion.create(
            model=self.model_path,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response['choices'][0]['message']['content']
    
if __name__ == "__main__":
    llm = ChatGPT()
    response = llm.chat("玉旋你好？")
    print(response)