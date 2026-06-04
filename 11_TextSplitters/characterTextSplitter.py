from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader('test-pdf-nlp.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator=''
)

result = splitter.split_documents(docs)

print('\n')
print(result[50].page_content)