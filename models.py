from flask_sqlalchemy import SQLAlchemy
import openai
import dotenv

# Загружаем переменные окружения из файла .env
env = dotenv.dotenv_values(".env")

# Инициализируем SQLAlchemy для работы с базой данных через Flask
db = SQLAlchemy()


class ChatHistory(db.Model):
    """
    Модель SQLAlchemy для хранения истории общения пользователя с LLM.

    Атрибуты:
        id (int): Уникальный идентификатор записи.
        user_message (str): Сообщение пользователя.
        llm_reply (str): Ответ языковой модели.
        timestamp (datetime): Время создания записи.
    """
    id = db.Column(db.Integer, primary_key=True)  # Уникальный идентификатор
    user_message = db.Column(db.Text, nullable=False)  # Текст сообщения пользователя
    llm_reply = db.Column(db.Text, nullable=False)  # Ответ модели
    timestamp = db.Column(db.DateTime, server_default=db.func.now())  # Временная метка создания записи


def dummy_llm_service(user_message):
    """
    Заглушка для сервиса LLM.

    Аргументы:
        user_message (str): Сообщение пользователя.

    Возвращает:
        str: Стандартный ответ, имитирующий работу LLM.
    """
    return f"Вы сказали: {user_message}, но я пока не могу ответить на это."


class LLMService:
    """
    Класс для взаимодействия с внешней языковой моделью (например, YandexGPT).

    Атрибуты:
        sys_prompt (str): Системный промпт для LLM.
        client: Клиент OpenAI для обращения к API.
        model (str): Идентификатор используемой LLM модели.
    """
    def __init__(self, sys_prompt):
        """
        Инициализация сервиса LLM.

        Аргументы:
            sys_prompt (str): Системный промпт для LLM.
        """
        try:
            # Создаём клиента OpenAI с вашим API-ключом и базовым URL
            self.client = openai.OpenAI(
                api_key=env["YA_API_KEY"],
                base_url="https://llm.api.cloud.yandex.net/v1",
            )
            # Сохраняем системный промпт
            self.sys_prompt = sys_prompt
            # Формируем путь к модели с использованием идентификатора каталога из .env
            self.model = f"gpt://{env['YA_FOLDER_ID']}/yandexgpt-lite"

        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

    def chat(self, message):
        """
        Отправляет сообщение к языковой модели и возвращает её ответ.

        Аргументы:
            message (str): Сообщение пользователя.

        Возвращает:
            str: Ответ языковой модели или сообщение об ошибке.
        """
        try:
            # Выполняем запрос к API языковой модели
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=1.0,      # Параметр креативности
                max_tokens=1024,      # Максимальная длина ответа
            )

            # Возвращаем текст ответа модели
            return response.choices[0].message.content

        except Exception as e:
            # В случае ошибки возвращаем её описание
            return f"Произошла ошибка: {str(e)}"
