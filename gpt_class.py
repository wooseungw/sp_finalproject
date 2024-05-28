import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
import requests

class GPT:
    def __init__(self, api_key, model="gpt-4o-2024-05-13", top_p=1.0, sys_prompt="You are a helpful assistant. please speak Korean."):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.top_p= top_p
        self.messages = [
            {'role': 'system', 'content': f'{sys_prompt}'},
        ]
    
    def _encode_image(self, image_path):
        # 이미지를 base64로 인코딩하는 함수입니다. gpt에 이미지를 넘기기 위해서는 인터넷url이 아닌경우 base64로 인코딩하여 넘겨야합니다.
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate(self, text_prompt, img_prompt:list=None):
        if img_prompt:
            messages = [{"type": "text", "text": text_prompt},]
            for img in img_prompt:
                # 이미지가 입력되면 이미지의 확장자를 가져옵니다. base64로 인코딩하게 될때 url에 이미지 타입이 포함되어 다양한 상황에 대응하기 위함입니다.
                img_tpye = img.split('.')[-1]
                # 이미지를 base64로 인코딩합니다.
                img = self._encode_image(img)
                # 이미지를 메세지에 추가합니다. 다중 이미지를 넘기기 위해서는 여러번 추가하기 위해 for문을 사용합니다.
                messages.append({"type": "image_url", 
                                 "image_url": {"url": f"data:image/{img_tpye};base64,{img}"}
                })
        self.messages.append({'role': 'user', 'content': messages})

        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            top_p=self.top_p,
        )

        self.messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
        return completion.choices[0].message.content

if __name__ == "__main__":     
    load_dotenv()
    openai = os.getenv("OPENAI_API_KEY")
    ## Example
    gpt = GPT(api_key=openai, model="gpt-4o", top_p=1.0)
    img_prompt = ['food.png']
    # img_prompt.append('food.png')
    # img_prompt.append('football.png')
    answer = gpt.generate(text_prompt="이 요리를 만들고싶어, 사진을보고 참고해서 요리의 레시피를 출력해줘", img_prompt=img_prompt)
    print(answer)
