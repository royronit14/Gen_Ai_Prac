from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = "Generate a 5 line of facts about the topic: {topic}",
    input_variable = ["topic"]
)

model = ChatGroq(model = "openai/gpt-oss-20b")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Angela White"})

print(result)

chain.get_graph().print_ascii()