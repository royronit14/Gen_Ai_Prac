from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# 1. Load Model
model1 = ChatGroq(model = "openai/gpt-oss-20b")

model2 = ChatGroq(model = "openai/gpt-oss-20b")

# 2. Design the prompt using PromptTemplate class
prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = "Generate 3 short question answer from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} and \n quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

# 3. Load the parser
parser = StrOutputParser()

# 4. Create Chains
# Create a parallel chain
parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model1 | parser
    }
)

# Merge the chains
merge_chain = prompt3 | model1 | parser

# final chain
chain = parallel_chain | merge_chain

text = """
xAI is an artificial intelligence company focused on developing the Grok family of large language models and multimodal AI systems. Founded by Elon Musk, the company positions Grok as a “maximally truth-seeking” AI assistant designed to provide direct, insightful, and real-time responses. xAI places strong emphasis on reasoning, large-context processing, web-connected generation, and agentic tool usage, while integrating its models across the broader X ecosystem and developer platforms.

The currently supported model families include:

Grok flagship models — xAI’s primary reasoning and multimodal models intended for production AI workloads. The Grok lineup includes:

High-performance reasoning models optimized for coding, analytical workflows, and advanced instruction following.

Fast non-reasoning variants designed for low-latency conversational and agentic applications.

Vision-capable Grok models supporting image understanding and multimodal interaction.
Recent Grok generations also introduce extended context windows, native tool calling, structured outputs, search integration, and real-time external information retrieval capabilities.

All xAI models in this provider are accessed through the standard /v1/chat/completions endpoint, providing a unified OpenAI-compatible integration layer across the entire model catalog.

Supported capabilities vary depending on the specific model, with different models offering different combinations of the features listed below.

Text completions: Build conversational systems and advanced text-processing pipelines.

Function Calling: Utilize tools, APIs, and structured workflows.

Stream mode: Receive partial responses incrementally as tokens are generated.

Vision Tasks: Process and analyze images and visual inputs.

Reasoning Tasks: Execute advanced analytical, coding, and multi-step reasoning workflows.

Large Context Processing: Work with extremely large prompts, documents, and conversational histories.
"""


# 5. Model invoke

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()
