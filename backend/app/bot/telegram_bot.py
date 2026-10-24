import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode

from app.config import settings
from app.services.agent_service import AgentService

logger = logging.getLogger("telegram_bot")

dp = Dispatcher()


@dp.message(CommandStart())
async def handle_start(message: types.Message):
    greeting = (
        "👋 **Привет! Я AI-агент Островка для планирования путешествий.**\n\n"
        "Я умею собирать поездку «под ключ» по вашим пожеланиям:\n"
        "• Подбирать направления по настроению и сезону\n"
        "• Находить отели и проверять условия (отмена, питание, pet-friendly)\n"
        "• Рассчитывать стыковку рейса и трансфера в отель\n"
        "• Предупреждать о рисках и форс-мажорах\n\n"
        "Просто напишите мне, чего вам хочется. Например:\n"
        "_«Хочу к теплому морю в ноябре с ребенком и собакой, без долгого перелета»_"
    )
    await message.answer(greeting, parse_mode=ParseMode.MARKDOWN)


@dp.message()
async def handle_user_query(message: types.Message):
    user_id = f"tg_{message.from_user.id}"
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        result = await AgentService.process_message(user_id=user_id, message=message.text)
        await message.answer(result["text"], parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error handling Telegram message: {e}", exc_info=True)
        await message.answer("Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз.")


async def start_telegram_bot():
    token = settings.TELEGRAM_BOT_TOKEN.strip()
    if not token or token in ("mock", "mock_token", "your_telegram_bot_token_here"):
        logger.info("ℹ️ TELEGRAM_BOT_TOKEN не задан. Бот находится в режиме ожидания токена.")
        logger.info("ℹ️ Полноценное тестирование доступно через Web UI на http://localhost:8080")
        return

    try:
        bot = Bot(token=token)
        logger.info(f"🚀 Запуск Telegram-бота @{settings.TELEGRAM_BOT_USERNAME}...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Не удалось запустить Telegram-бота: {e}")
