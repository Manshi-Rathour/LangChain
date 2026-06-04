from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model='mistral:7b-instruct')

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.britannica.com/place/India'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | llm | parser

result = chain.invoke({'question': 'What is the topic we are talking about?', 'text': docs[0].page_content})

print(result)