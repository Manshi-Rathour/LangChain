from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on topic {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a five line summary on the following text. /n {text}",
    input_variables=['text']
)

# prompt1 = template1.invoke({'topic': 'black hole'})
# result1 = model.invoke(prompt1)
#
# prompt2 = template2.invoke({'text': result1})
# result2 = model.invoke(prompt2)
#
# print(result2.content)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)