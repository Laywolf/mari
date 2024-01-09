from llama_cpp import Llama

llm = Llama(model_path="./model/llama-2-ko-7b-chat-q8-0.gguf")
output = llm("Q: 인생이란 뭘까요?. A: ", max_tokens=1024, stop=["Q:", "\n"], echo=True)
print(output["choices"][0]["text"].replace("▁", " "))
