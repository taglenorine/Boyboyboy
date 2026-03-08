"""
main.py – BalapBoY v2.0 Telegram bot entry point.

This file sets up the python-telegram-bot Application, registers handlers,
and dispatches incoming messages to the Agentic core (core/pollinations.py).
"""

from __future__ import annotations

import logging
import sys

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import TELEGRAM_BOT_TOKEN
from core.pollinations import run_agent
from database import crud

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start – greet the user and create their DB record."""
    user = update.effective_user
    crud.get_user(user.id, user.username or "")
    await update.message.reply_text(
        "👋 Halo, Bos! Gue *BalapBoY AI v2.0* — autonomous agent siap bantu!\n\n"
        "Cukup ceritain apa yang lo mau, gue yang urus semuanya. 🚀\n\n"
        "Contoh:\n"
        "• _\"Buatin konsep kafe futuristik dan 2 gambarnya\"_\n"
        "• _\"Analisis foto ini dong\"_\n"
        "• _\"Harga Bitcoin hari ini berapa?\"_\n\n"
        "Ketik */help* untuk daftar perintah lengkap.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    await update.message.reply_text(
        "*📖 BalapBoY AI v2.0 – Panduan Singkat*\n\n"
        "Cukup tulis apa yang lo mau dalam bahasa natural — gue akan otomatis "
        "pakai tools yang diperlukan.\n\n"
        "*Perintah tersedia:*\n"
        "• */start* – Mulai / perkenalan\n"
        "• */help* – Panduan ini\n"
        "• */login* – Daftarkan BYOP API Key kamu\n"
        "• */balance* – Cek saldo Pollen kamu\n"
        "• */reset* – Hapus riwayat percakapan\n\n"
        "*Kapabilitas otomatis:*\n"
        "• 🖼 Buat gambar\n"
        "• 💬 Jawab pertanyaan & coding\n"
        "• 👁 Analisis foto yang lo kirim\n"
        "• 🎬 Buat video (Premium / BYOP)\n"
        "• 🔊 Text-to-Speech\n",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /balance – show the user's Pollen balance."""
    user = update.effective_user
    balance = crud.get_pollen_balance(user.id)
    daily_ok = crud.check_daily_grant(user.id)
    await update.message.reply_text(
        f"🌸 *Saldo Pollen kamu:* `{balance:.4f}`\n"
        f"📅 *Request gratis hari ini:* "
        f"{'✅ Masih ada' if daily_ok else '❌ Sudah habis'}",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /login <api_key> – save a BYOP key."""
    user = update.effective_user
    if not context.args:
        await update.message.reply_text(
            "🔑 Gunakan: `/login <API_KEY_KAMU>`\n\n"
            "API Key akan disimpan secara aman dan digunakan untuk semua "
            "request kamu.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    api_key = context.args[0].strip()
    crud.save_byop_key(user.id, api_key)
    # Delete the message containing the key for security
    try:
        await update.message.delete()
    except Exception:
        pass
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "✅ *API Key berhasil disimpan!*\n\n"
            "Pesan yang berisi key kamu sudah dihapus otomatis untuk keamanan. 🔒"
        ),
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset – clear conversation history."""
    user = update.effective_user
    crud.save_context(user.id, [])
    await update.message.reply_text(
        "🗑 *Riwayat percakapan dihapus!*\n\nKita mulai dari awal, Bos. 👍",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── Message handlers ──────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages – route to the Agentic core."""
    user = update.effective_user
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    response = await run_agent(
        user_id=user.id,
        user_message=update.message.text or "",
        username=user.username or "",
    )

    await _send_agent_response(update, response)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages – download the image and run vision analysis."""
    user = update.effective_user
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    # Download the highest-resolution version of the photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()

    caption = update.message.caption or "Analisis gambar ini."

    response = await run_agent(
        user_id=user.id,
        user_message=caption,
        username=user.username or "",
        image_bytes=bytes(image_bytes),
    )

    await _send_agent_response(update, response)


# ── Response dispatcher ───────────────────────────────────────────────────────

async def _send_agent_response(
    update: Update,
    response,
) -> None:
    """Send the AgentResponse to Telegram (text + any media)."""
    chat_id = update.effective_chat.id

    # Send any generated media first
    for media in response.media:
        try:
            if media.kind == "image":
                await update.message.reply_photo(
                    photo=media.data,
                    caption=media.caption[:1024] if media.caption else None,
                )
            elif media.kind == "video":
                await update.message.reply_video(
                    video=media.data,
                    caption=media.caption[:1024] if media.caption else None,
                )
            elif media.kind == "audio":
                await update.message.reply_voice(voice=media.data)
        except Exception as exc:
            logger.error("Failed to send media: %s", exc)

    # Send the text reply (split if too long for a single Telegram message)
    text = response.text
    if not text:
        return

    max_len = 4096
    for i in range(0, len(text), max_len):
        chunk = text[i : i + max_len]
        try:
            await update.message.reply_text(
                chunk, parse_mode=ParseMode.MARKDOWN
            )
        except Exception:
            # Fallback: send without Markdown if parsing fails
            await update.message.reply_text(chunk)


# ── Error handler ─────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by updates."""
    logger.error("Update caused an error:", exc_info=context.error)


# ── Application bootstrap ─────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.critical(
            "TELEGRAM_BOT_TOKEN is not set. Please configure .env and restart."
        )
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("login", cmd_login))
    app.add_handler(CommandHandler("reset", cmd_reset))

    # Messages
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("BalapBoY AI v2.0 is starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
