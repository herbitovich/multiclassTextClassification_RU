import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import sqlite3
import requests
import json
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
import random
# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
# Load the saved model
model = TFBertForSequenceClassification.from_pretrained('rubert-base-cased-multiclass')

categories = ['business', 'science', 'politics', 'sports', 'technology', 'society and accidents']

MAX_NEWS_FROM_CATEGORY = 2

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO                                            
)

connection = sqlite3.connect('categorizer.db')
cursor = connection.cursor()

async def categorize(update: Update, context: ContextTypes.DEFAULT_TYPE, respond=True, text=None):
    if not text:
        text = update.message.text
        application.remove_handler(text_handler)
    encoding = tokenizer(text, truncation=True, padding=True, return_tensors='tf')

    # Make the prediction
    output = model(**encoding)[0]
    predicted_class = tf.argmax(output, axis=1).numpy()[0]
    category = categories[predicted_class]
    if respond:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"The category of the given text is: {category}")
    return category

async def listen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Provide me with a text to categorize.")
    application.add_handler(text_handler)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")
    
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def sources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_channels = update.message.text.split()[1:]
    print("Channels provided: ",user_channels)
    channels = set()
    for ch in user_channels:
        print(ch)
        if ch[0] == '@' and ch in requests.get(f'https://t.me/s/{ch[1:]}').text:
            channels.add(ch)
    if not channels:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No valid channels provided.")
    else:
        if len(channels) > 5:
            channels = channels[:5]
            await context.bot.send_message(chat_id=update.effective_chat.id, text="You can only add up to 5 channels as news sources.")
        username = update.message.from_user.username
        user = cursor.execute("SELECT * FROM Sources WHERE username = ?", (username,)).fetchone()
        if not user:
            cursor.execute("INSERT INTO Sources (username, sources) VALUES (?, ?)", (username, json.dumps(list(channels))))
            connection.commit()
        else:
            cursor.execute("""
                UPDATE Sources
                SET sources = (?)
                WHERE username = (?);
                           """, (json.dumps(list(channels)), username))
            connection.commit()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"The following channels were added as news sources for @{username}: {' '.join(list(channels))}.")

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.message.from_user.username
    user = cursor.execute("SELECT * FROM Sources WHERE username = ?", (username, )).fetchone()
    if not user:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="You have no news sources. Use /sources to add some.")
    else:
        articles = {}
        channels = json.loads(user[2])
        random.shuffle(channels)
        for ch in channels:
            r = requests.get(f'https://t.me/s/{ch[1:]}')
            soup = bs(r.text, 'html.parser')
            posts = soup.find_all('div', class_='tgme_widget_message')
            for post in posts:
                text = post.find('div', {'class':'tgme_widget_message_text'})
                if text:
                    text = text.text
                    text.replace('\n', ' ')
                    text.strip()
                if text:
                    category = await categorize(update=update, context=context, text=text, respond=False)
                    if category not in articles:
                        articles[category] = []
                    text = text[:50] + '...'
                    link = post.find('a', class_='tgme_widget_message_date')['href']
                    date, time = post.find('time', {'class': 'time'})['datetime'].split('T')
                    date = f"{'.'.join(date.split('-')[::-1])} at {time.split('+')[0]}"
                    articles[category].append(f'"{text}" posted by {ch}: {link}, date: {date}')
                if all(len(articles[cat]) >= MAX_NEWS_FROM_CATEGORY for cat in articles):
                    break
        message = ""
        for category in articles:
            message += f"{category}:\n"
            if articles[category]:
                random.shuffle(articles[category])
                count = min(MAX_NEWS_FROM_CATEGORY, len(articles[category]))
                order = 1
                while count:
                    message += f"{order}. "+articles[category].pop(0) + "\n"
                    count -= 1
                    order += 1
                message += "\n\n"
            else:
                message += f"No articles found.\n\n"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=message)
if __name__ == '__main__':
    #application
    application = ApplicationBuilder().token('').build()
 #TODO    
    #command handlers
    start_handler = CommandHandler('start', start)
    categorize_handler = CommandHandler('categorize', listen)
    sources_handler = CommandHandler('sources', sources)
    news_handler = CommandHandler('news', news)
    #message handlers
    text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), categorize)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    #adding command handlers to the application
    application.add_handler(start_handler)
    application.add_handler(news_handler)
    application.add_handler(sources_handler)
    application.add_handler(categorize_handler)




    #application.add_handler(text_handler)
    application.add_handler(unknown_handler)
    #running the application
    application.run_polling()