from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.2, max_completion_tokens=20)

result = model.invoke("What is the capital of Bihar?")

print(result)
print(result.content)