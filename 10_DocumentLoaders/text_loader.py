from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model='mistral:7b-instruct')

prompt = PromptTemplate(
    template='Write a summary for the following text \n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('sample_txt_file.txt')

text = loader.load()

# print(type(text))
# print(type(text[0]))
# print(f'\n{text[0].page_content}')
# print(f'\n{text[0].metadata}')

chain = prompt | llm | parser

result = chain.invoke({'text': text[0].page_content})

print(result)