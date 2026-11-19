from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

messages = [
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = "Tell me about Langchain, in 2 lines.")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)