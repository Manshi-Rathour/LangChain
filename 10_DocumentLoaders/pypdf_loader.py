from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("test-pdf-nlp.pdf")

documents = loader.load()

# print(f'\n{documents}\n')
#
# for doc in documents:
#     print(doc.page_content)
#     print('\n')
#
# print(len(documents))

print('\n\n')
print(documents[1].page_content)
print('\n')
print(documents[1].metadata)