import functools
import typing
import time
import wave
import speech_recognition
import asyncio

import discord
from discord.ext import commands, voice_recv, tasks
from discord.opus import Decoder as OpusDecoder

from llama_cpp import Llama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os

load_dotenv()

discord.opus._load_default()

mari_prompt = os.environ.get("SECRET_PROMPT")
mari_channel_id = int(os.environ.get("MARI_CHANNEL_ID"))

mari_witai_key = os.environ.get("MARI_WITAI_KEY")

llama_model_path = os.environ.get("LLAMA_MODEL_PATH")

sensei = os.environ.get("SENSEI")
mari = os.environ.get("MARI")


llm = Llama(model_path=llama_model_path, n_ctx=4096, verbose=False, n_gpu_layers=64)

chat_memory = ConversationBufferMemory(human_prefix=sensei, ai_prefix=mari)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(mari_prompt)

now_inferencing = False


bot = commands.Bot(command_prefix="", intents=discord.Intents.all())
stt = speech_recognition.Recognizer()

idleCounter = 0
currentFile: wave.Wave_write = None
currentFilename = None

vc: voice_recv.VoiceRecvClient = None
mc: int = None

now_recording = 0


async def chat_message(text):
    if mc is None:
        return
    channel = bot.get_channel(mc)

    bot.loop.create_task(channel.send(f"선생님:loudspeaker:: {text}"))

    answer = await run_blocking(chat, message=text)
    if answer is None:
        return
    bot.loop.create_task(channel.send(answer))


async def check_voice():
    record_finished = False
    record_count = 0
    recording_number = 0
    global now_recording, currentFile, currentFilename

    while True:
        # print(
        #     f"checking voice... {recording_number}, {now_recording}, {record_count}, {record_finished}"
        # )
        localFile = currentFile
        localFilename = currentFilename

        if record_finished:
            if (localFile is not None) and (localFilename is not None):
                try:
                    with speech_recognition.AudioFile(localFilename) as source:
                        audio_data = stt.record(source)
                    try:
                        text = stt.recognize_wit(audio_data, key=mari_witai_key)
                        print("인식된 텍스트:", text)
                        await chat_message(text)
                    except speech_recognition.UnknownValueError:
                        print("음성을 인식하지 못했습니다.")
                    except speech_recognition.RequestError as e:
                        print(f"Wit.ai 요청 에러: {e}")
                except:
                    print("File error")

                currentFile = None
                currentFilename = None

                now_recording = 0
                record_finished = False
                record_count = 0
                recording_number = 0

                localFile.close()
        else:
            if (recording_number == now_recording) and (now_recording != 0):
                # print("silent...")

                if record_count < 5:
                    record_count += 1
                else:
                    record_finished = True
            else:
                recording_number = now_recording

        await asyncio.sleep(0.1)


def chat(message):
    global now_inferencing
    if now_inferencing:
        return

    print("대답 만드는 중...")

    now_inferencing = True

    # extract chat history
    chats = chat_memory.load_memory_variables({})
    chat_history_all = chats["history"]

    # use last two chunks of chat history
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        separators=[sensei + ": "],
        length_function=len,
    )
    texts = text_splitter.split_text(chat_history_all)
    pages = len(texts)
    if pages >= 2:
        chat_history = f"{texts[pages-2]}{texts[pages-1]}\n\n"
    elif pages == 1:
        chat_history = texts[0] + "\n\n"
    else:  # 0 page
        chat_history = ""
    # print(chat_history)

    query = CONDENSE_QUESTION_PROMPT.format(question=message, chat_history=chat_history)
    # print(query)

    # make a question using chat history
    result = llm(
        query,
        max_tokens=4096,
        stop=[sensei + ": ", "\n\n"],
        echo=False,
        repeat_penalty=1.2,
    )
    print(result)

    answer = str(result["choices"][0]["text"])
    now_inferencing = False
    chat_memory.save_context({"input": message + "\n"}, {"output": answer + "\n"})
    # print(answer)

    return answer


async def run_blocking(blocking_func: typing.Callable, *args, **kwargs) -> typing.Any:
    """Runs a blocking function in a non-blocking way"""
    func = functools.partial(
        blocking_func, *args, **kwargs
    )  # `run_in_executor` doesn't support kwargs, `functools.partial` does
    return await bot.loop.run_in_executor(None, func)


def voiceCallback(user, data: voice_recv.VoiceData):
    # print(f"Got packet from {user}")
    if user == None:
        return

    # voice power level, how loud the user is speaking
    # ext_data = data.packet.extension_data.get(voice_recv.ExtensionID.audio_power)
    # value = int.from_bytes(ext_data, "big")
    # power = 127 - (value & 127)
    # print("#" * int(power * (79 / 128)))

    global idleCounter, currentFile, currentFilename, stt, now_recording

    # print(idleCounter, currentFilename)

    now_recording += 1

    localFile = currentFile

    if localFile is not None:
        localFile.writeframes(data.pcm)

    else:
        try:
            localFilename = f"./voices/{user}-{round(time.time() * 1000)}.wav"
            print("create file: " + localFilename)

            localFile = wave.open(localFilename, "wb")
            localFile.setnchannels(OpusDecoder.CHANNELS)
            localFile.setsampwidth(OpusDecoder.SAMPLE_SIZE // OpusDecoder.CHANNELS)
            localFile.setframerate(OpusDecoder.SAMPLING_RATE)

            currentFile = localFile
            currentFilename = localFilename
        except:
            print("File sync error")


@bot.event
async def on_ready():
    print("Logged on as {0}!".format(bot.user))
    await bot.change_presence(
        status=discord.Status.online, activity=discord.Game("선생님의 질문에 대답")
    )

    bot.loop.create_task(check_voice())


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return
    if message.channel.id != mari_channel_id:
        return

    print(message.author, ": ", message.content)

    if message.content == "ping":
        await message.channel.send("pong {0.author.mention}".format(message))
    elif message.content == "들어와":
        if message.author.voice and message.author.voice.channel:
            global vc, mc
            vc = await message.author.voice.channel.connect(
                cls=voice_recv.VoiceRecvClient
            )
            vc.listen(voice_recv.BasicSink(voiceCallback))
            mc = message.channel.id
        else:
            await message.channel.send("선생님. 어디 계세요?")
    elif message.content == "나가":
        await message.channel.send("알겠습니다, 선생님. 안녕히 계세요.")
        await bot.voice_clients[0].disconnect()
    else:
        answer = await run_blocking(chat, message=message.content)
        if answer is None:
            return
        await message.channel.send(answer)


TOKEN = os.environ.get("DISCORD_TOKEN")

bot.run(TOKEN)
