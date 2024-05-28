import os
from dotenv import load_dotenv
from PIL import Image

import google.generativeai as genai

class Gemini:
    def __init__(self, api_key, model_name='gemini-1.0-pro-vision-latest'):
        genai.configure(api_key=api_key)  # Google AI API 키를 구성합니다.
        self.model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')  # Gemini AI 모델을 생성합니다.
        self.chat = self.model.start_chat(history=[])  # 채팅 세션을 시작합니다.

    def generate(self, text_prompt, img_prompt=None):
        messages = [text_prompt]  # 텍스트 프롬프트를 메시지 리스트에 추가합니다.
        if img_prompt:
            for img in img_prompt:
                img = Image.open(img)  # 이미지 파일을 엽니다.
                messages.append(img)  # 이미지를 메시지 리스트에 추가합니다.
        response = model.generate_content(messages)
        
        return response.text  # 응답 텍스트를 반환합니다.
    
if __name__ == "__main__":
    gemini = os.getenv("GOOGLE_API_KEY")  # 환경 변수에서 Google API 키를 가져옵니다.
    model = Gemini(api_key=gemini)  # Gemini 클래스의 인스턴스를 생성합니다.
    text = "인생의 의미를 이 이미지들을 참고해서 말하자면 무엇인가요?"  # 텍스트 프롬프트를 설정합니다.
    img = ['food.png']  # 이미지 프롬프트를 설정합니다.
    print(model.generate(text_prompt=text, img_prompt=img))  # 생성된 모델을 사용하여 텍스트와 이미지를 기반으로 생성된 응답을 출력합니다.

print(123)