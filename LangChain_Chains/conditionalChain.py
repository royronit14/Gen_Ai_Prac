from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

class Feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description="Give me the sentiment of the feedback.")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}",
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# result = classifier_chain.invoke({'feedback':"This is a terrible smartphone"})

# Runnable Branch Syntax
'''
branch_chain = RunnableBranch(
    (condition1, chain1),
    (condition2, chain2),
    default
)
'''

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

parser = StrOutputParser()

chain1 = prompt2 | model | parser
chain2 = prompt3 | model | parser

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', chain1),  #agar sentiment positive hai to chain1 chalega
    (lambda x : x.sentiment == 'negative', chain2),   #agar sentiment negative hai to chain2 chalega
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': "This is an tatti device"}))