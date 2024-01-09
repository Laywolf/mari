from llama_cpp import Llama
import discord
from discord.ext import commands
import functools
import typing
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os

load_dotenv()

mari_prompt = os.environ.get('SECRET_PROMPT')

mari_channel_id = int(os.environ.get('MARI_CHANNEL_ID'))

llm = Llama(model_path="./model/ggml-model-q2_k.gguf", n_ctx=2048, verbose=False)

chat_memory = ConversationBufferMemory(human_prefix="선생님", ai_prefix="이오치 마리")

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(mari_prompt)

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

@to_thread
def chat(message):
    print('대답 만드는 중...')
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats['history']

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4096,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    )
    texts = text_splitter.split_text(chat_history_all)

    pages = len(texts)

    if pages >= 2:
        chat_history = f"{texts[pages-2]} {texts[pages-1]}"
    elif pages == 1:
        chat_history = texts[0]
    else:  # 0 page
        chat_history = ""
    # print(chat_history)

    query = CONDENSE_QUESTION_PROMPT.format(question=message, chat_history=chat_history)
    # print(query)

    # make a question using chat history
    if pages >= 1:
        result = llm(query, max_tokens=1024, stop=["선생님:", "\n"], echo=False)
    else:
        result = llm(query, max_tokens=1024, stop=["선생님:", "\n"], echo=False)
    
    answer = str(result["choices"][0]["text"].split("\n")[-1]).replace("이오치 마리 : ", "")

    chat_memory.save_context({"input": message}, {"output": answer})

    return answer

app = commands.Bot(command_prefix="", intents=discord.Intents.all())

@app.event
async def on_ready():
    print("Logged on as {0}!".format(app.user))
    await app.change_presence(
        status=discord.Status.online, activity=discord.Game("선생님의 질문에 대답")
    )


@app.event
async def on_message(message):
    if message.author == app.user:
        return
    if message.channel.id != mari_channel_id:
        return
    
    print(message.author, ": ", message.content)

    if message.content == "ping":
        await message.channel.send("pong {0.author.mention}".format(message))
    else:
        answer = await chat(message=message.content)
        await message.channel.send(answer)

TOKEN = os.environ.get('DISCORD_TOKEN')

app.run(TOKEN)
