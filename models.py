from flask_sqlalchemy import SQLAlchemy
import openai
import logging

# Настройка логгирования
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


class LLMService:
    """
    Класс для взаимодействия с языковой моделью (локальной через Ollama).

    Атрибуты:
        sys_prompt (str): Системный промпт для LLM.
        client: Клиент OpenAI для обращения к локальному API Ollama.
        model (str): Идентификатор используемой LLM модели.
    """
    def __init__(self, prompt_file):
        """
        Инициализация сервиса LLM.

        Аргументы:
            prompt_file (str): Путь к файлу с системным промптом для LLM.
        """
        # Читаем системный промпт из файла и сохраняем в атрибут sys_prompt
        with open(prompt_file, encoding='utf-8') as f:
            self.sys_prompt = f.read()

        # Настройка для локальной модели Ollama
        self.client = openai.OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )
        self.model = "gemma3:4b"

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
            logger.error(f"Произошла ошибка: {str(e)}")
            return f"Произошла ошибка: {str(e)}"


llm_1 = LLMService('prompts/prompt_1.txt')


def chat_with_llm(user_message):
    """
    Чат с использованием сервиса LLM.

    Аргументы:
        user_message (str): Сообщение пользователя.

    Возвращает:
        str: Ответ LLM.
    """
    response = llm_1.chat(user_message)
    return response
