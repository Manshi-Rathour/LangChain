from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(model='mistral:7b-instruct')

prompt1 = PromptTemplate(
    template='Write a tweet about the topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a LinkedIn post about topic {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, llm, parser),
    'linkedin': RunnableSequence(prompt2, llm, parser)
})

result = parallel_chain.invoke({'topic': 'AI'})
print(result)
print("\n")
print("=" * 50)
print(f'\nTweet: {result['tweet']}\n')
print("=" * 50)
print(f'\nLinkedIn: {result['linkedin']}')