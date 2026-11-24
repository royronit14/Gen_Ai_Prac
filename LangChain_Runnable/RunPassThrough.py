from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

# 1. Load Model
model = ChatGroq(model = "openai/gpt-oss-20b")


prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),    #Ab na 'joke' ke pss na joke hi hai aur kuch process ni kr rha hai vo, so o/p mai joke hi vapas dega
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))