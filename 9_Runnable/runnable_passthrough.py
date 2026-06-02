from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(model='mistral:7b-instruct')

prompt1 = PromptTemplate(
    template='Write a joke about the topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write explanation about the joke {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, llm, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})
print(result)

print("\n")
print("=" * 50)
print(f'\nJoke: {result['joke']}\n')
print("=" * 50)
print(f'\nExplanation: {result['explanation']}')