from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = OllamaLLM(model='mistral:7b-instruct')

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback text.

{format_instruction}

Feedback:
{feedback}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

# result = classifier_chain.invoke({'feedback': "This is a terrible smartphone"}).sentiment
# print(result)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': "This is a beautiful smartphone"})

print(result)