from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
class Runnable(ABC):
    @abstractmethod
    def invoke(self):
        pass

class FakeLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'AI stands for Artificial Intelligence'
        ]

        return {'response': random.choice(response_list)}

class FakePromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

class FakeStrOutputParser(Runnable):
    def __init__(slef):
        pass

    def invoke(self, input_data):
        return input_data['response']

class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

template = FakePromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = FakeLLM()

parser = FakeStrOutputParser()
chain = RunnableConnector([template, llm, parser])
result = chain.invoke({'length': 'long', 'topic': 'India'})
print(result)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_text(code)

print(len(chunks))
# print(chunks[0])

for chunk in chunks:
    print(chunk)
