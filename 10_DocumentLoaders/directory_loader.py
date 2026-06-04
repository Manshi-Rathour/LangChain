from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='../10_testDir',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

# print('\n')
# print(docs[1].page_content)
# print('\n')
# print(docs[1].metadata)

print('\n')

for doc in docs:
    print(doc.metadata)
    print('\n')
