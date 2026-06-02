from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(model='mistral:7b-instruct')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write detailed report on topic \n{topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n{text}',
    input_variables=['text']
)

report_gen_chain = prompt1 | llm | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | llm | parser),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

result = final_chain.invoke({'topic': 'Russia vs Ukraine'})

print(result)