from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

# 1. Load Model
model = ChatGroq(model = "openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2  = PromptTemplate(
    template="Generate a linked post about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedIn' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

# print(result)
print(result['tweet'])
print("<-------------------------------->")
print(result['linkedIn'])