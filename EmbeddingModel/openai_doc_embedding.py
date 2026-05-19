from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

docs = [
    'Delhi is the capital of India',
    'Patna is the capital of Bihar',
    'Paris is the capital of France'
]
vector = embedding.embed_documents(docs)

print(str(vector))