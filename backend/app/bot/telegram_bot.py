import asyncio
import logging
import re
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
    try:
        await message.answer(greeting, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await message.answer(greeting)


@dp.message()
async def handle_user_query(message: types.Message):
    user_id = f"tg_{message.from_user.id}"
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        result = await AgentService.process_message(user_id=user_id, message=message.text)
        raw_text = result.get("text", "")
        # Вырезаем блок reasoning details из вывода
        clean_text = re.sub(r"<details class=\"agent-reasoning\">[\s\S]*?</details>", "", raw_text).strip()
        if len(clean_text) > 4000:
            clean_text = clean_text[:3990] + "..."
        try:
            await message.answer(clean_text, parse_mode=ParseMode.MARKDOWN)
        except Exception as parse_err:
            logger.warning(f"Telegram parse_mode=MARKDOWN error: {parse_err}. Sending as plain text.")
            await message.answer(clean_text)
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
