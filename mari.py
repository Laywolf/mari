from llama_cpp import Llama
import discord
from discord.ext import commands
import functools
import typing

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os

load_dotenv()

mari_prompt = os.environ.get("SECRET_PROMPT")
mari_channel_id = int(os.environ.get("MARI_CHANNEL_ID"))

llama_model_path = os.environ.get("LLAMA_MODEL_PATH")

sensei = os.environ.get("SENSEI")
mari = os.environ.get("MARI")

llm = Llama(model_path=llama_model_path, n_ctx=4096, verbose=False)

chat_memory = ConversationBufferMemory(human_prefix=sensei, ai_prefix=mari)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(mari_prompt)

now_inferencing = False


def chat(message):
    print("대답 만드는 중...")
    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats["history"]

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=0,
        separators=[sensei + ": "],
        length_function=len,
    )
    texts = text_splitter.split_text(chat_history_all)

    pages = len(texts)

    if pages >= 2:
        chat_history = f"{texts[pages-2]}{texts[pages-1]}\n"
    elif pages == 1:
        chat_history = texts[0] + "\n"
    else:  # 0 page
        chat_history = ""
    # print(chat_history)

    query = CONDENSE_QUESTION_PROMPT.format(question=message, chat_history=chat_history)
    print(query)

    global now_inferencing
    now_inferencing = True

    # make a question using chat history
    result = llm(
        query,
        max_tokens=4096,
        stop=[sensei + ": ", "\n", ":"],
        echo=False,
    )

    print(result)

    answer = str(result["choices"][0]["text"].split("\n")[-1]).replace(mari + ": ", "")

    now_inferencing = False

    chat_memory.save_context({"input": message + "\n"}, {"output": answer + "\n"})

    return answer


app = commands.Bot(command_prefix="", intents=discord.Intents.all())


async def run_blocking(blocking_func: typing.Callable, *args, **kwargs) -> typing.Any:
    """Runs a blocking function in a non-blocking way"""
    func = functools.partial(
        blocking_func, *args, **kwargs
    )  # `run_in_executor` doesn't support kwargs, `functools.partial` does
    return await app.loop.run_in_executor(None, func)


@app.event
async def on_ready():
    print("Logged on as {0}!".format(app.user))
    await app.change_presence(
        status=discord.Status.online, activity=discord.Game("선생님의 질문에 대답")
    )


@app.event
async def on_message(message):
    if now_inferencing:
        return
    if message.author == app.user:
        return
    if message.channel.id != mari_channel_id:
        return

    print(message.author, ": ", message.content)

    if message.content == "ping":
        await message.channel.send("pong {0.author.mention}".format(message))
    else:
        answer = await run_blocking(chat, message=message.content)
        await message.channel.send(answer)


TOKEN = os.environ.get("DISCORD_TOKEN")

app.run(TOKEN)
