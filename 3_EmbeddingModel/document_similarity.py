from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "Artificial Intelligence is transforming many industries.",
    "Machine learning helps computers learn from data.",
    "Neural networks are inspired by the human brain.",
    "AI-powered chatbots are becoming increasingly popular.",
    "Deep learning models require large amounts of training data."
]

query = "What is neural network"

doc_embedding = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

similarity_score = cosine_similarity([query_embedding], doc_embedding)[0]

print(similarity_score)

index, score = sorted(list(enumerate(similarity_score)), key=lambda x:x[1])[-1]

print(docs[index])
print("similarity score is: ", score)