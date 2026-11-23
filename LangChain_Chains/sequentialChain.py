from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template = "Generate a detail report over the research work: {topic}",
    input_variable = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate a summary for the in 3points from the following text: \n {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": " Explainable AI for Optimizing Network Intrusion Detection in Resource-Constrained Environments"})

print(result)

chain.get_graph().print_ascii()