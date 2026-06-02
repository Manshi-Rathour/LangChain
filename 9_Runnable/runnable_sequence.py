from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(model='mistral:7b-instruct')

prompt1 = PromptTemplate(
    template='Write a joke about topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)

print(chain.invoke({'topic': 'AI'}))