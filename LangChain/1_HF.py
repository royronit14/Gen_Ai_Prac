from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

result = model.invoke("Average Penis length of India is?")
print(result.content)