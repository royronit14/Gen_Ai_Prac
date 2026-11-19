from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain me in a layman understanding about {topic}")

])

'''
    Ye kaam ni krega inside the ChatPromptTemplate: 

    SystemMessage(content = "You are a helpful {domain} expert."),
    HumanMessage(content="Explain me in a layman understanding about {topic}")
'''
    

prompt = chat_template.invoke({'domain': 'cricket', 'topic':'Catch'})

print(prompt)
