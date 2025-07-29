from flask_sqlalchemy import SQLAlchemy
import openai
import dotenv

env = dotenv.dotenv_values(".env")

db = SQLAlchemy()


class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.Text, nullable=False)
    llm_reply = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())


def dummy_llm_service(user_message):
    return f"Вы сказали: {user_message}, но я пока не могу ответить на это."


class LLMService:
    def __init__(self, sys_prompt):
        try:
            # Создаем клиент с вашим токеном
            self.client = openai.OpenAI(
                api_key=env["YA_API_KEY"],
                base_url="https://llm.api.cloud.yandex.net/v1",
            )
            # Формируем системный промпт
            self.sys_prompt = sys_prompt
            # Указываем путь к модели, 
            # Здесь используется идентификатор каталога вашего аккаунта 
            self.model = f"gpt://{env["YA_FOLDER_ID"]}/yandexgpt-lite"

        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

    def chat(self, message):
        try:
            # Обращаемся к API
            response = self.client.chat.completions.create(
                model = self.model,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=1.0,
                max_tokens=1024,
            )

            # Возвращаем ответ
            return response.choices[0].message.content

        except Exception as e:
            return f"Произошла ошибка: {str(e)}"
