'''from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)


# Abb isme problem ye hai ki, isko previous messages yaad hi ni rahegi smjhe baby Girl
'''


from langchain_groq import ChatGroq
# ab hum messages implement krenge
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")
chat_history = [
    SystemMessage(content = "You are an AI Assistant." )
]

while True:
    user_input = input("You: ")
    # user_input ko chathistory mai daal denge
    chat_history.append(HumanMessage(content= user_input))
    if user_input == "exit":
        break
    # model mai puri chat_history denge taaki usko prev context yaad ho
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ", result.content)

print(chat_history)