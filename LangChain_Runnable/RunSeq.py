from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

# 1. Load Model
model = ChatGroq(model = "openai/gpt-oss-20b")

# 2. Prompt Template Creation
prompt1 = PromptTemplate(
    template="Write a joke about the topic \n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain me the joke in 3lines {joke}",
    input_variables=['joke']
)
parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic': 'AI'})

print(result)