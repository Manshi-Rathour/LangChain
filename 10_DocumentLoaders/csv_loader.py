from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='test.csv')

data = loader.load()

print(data[1])