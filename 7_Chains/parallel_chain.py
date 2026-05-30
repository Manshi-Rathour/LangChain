from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = OllamaLLM(model='mistral:7b-instruct')

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="""
You are given notes and quiz content.
Combine BOTH without removing anything.

NOTES:
{notes}

QUIZ:
{quiz}

Return output in this format:

# Notes
<notes>

# Quiz
<quiz>
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
LangChain is an open source orchestration framework for application development using large language models (LLMs). Available in both Python- and Javascript-based libraries, LangChain’s tools and APIs simplify the process of building LLM-driven applications like chatbots and AI agents.
LangChain serves as a generic interface for nearly any LLM, providing a centralized development environment to build LLM applications and integrate them with external data sources and software workflows. LangChain’s module-based approach allows developers and data scientists to dynamically compare different prompts and even different foundation models with minimal need to rewrite code. This modular environment also allows for programs that use multiple LLMs: for example, an application that uses one LLM to interpret user queries and another LLM to author a response.
Launched by Harrison Chase in October 2022, LangChain enjoyed a meteoric rise to prominence: as of June 2023, it was the single fastest-growing open source project on Github.1 Coinciding with the momentous launch of OpenAI’s ChatGPT the following month, LangChain has played a significant role in making generative AI (genAI) more accessible to enthusiasts and startups in the wake of its widespread popularity. Advancements in accessibility for agentic AI are currently enabling a revolution in automation.
LLMs are not standalone applications: they are pre-trained statistical models that must be paired with an application (and, in some cases, specific data sources) in order to meet their purpose.
For example, Chat-GPT is not an LLM: it is a chatbot application that, depending on the version you’ve chosen, uses the GPT-3.5 or GPT-4 language model. While it’s the GPT model that interprets the user’s input and composes a natural language response, it’s the application that (among other things) provides an interface for the user to type and read and a UX design that governs the chatbot experience. Even at the enterprise level, Chat-GPT is not the only application using the GPT model: Microsoft uses GPT-4 to power Bing Chat.
Furthermore, though foundation models (like those powering LLMs) are pre-trained on massive datasets, they are not omniscient. If a particular task requires access to specific contextual information, like internal documentation or domain expertise, LLMs must be connected to those external data sources. Even if you simply want your model to reflect real-time awareness of current events, it requires external information: a model’s internal data is only up-to-date through the time period during which it was pre-trained.
Likewise, if a given generative AI task requires access to external software workflows—for example, if you wanted your virtual agent to integrate with Slack—then you will need a way to integrate the LLM with the API for that software.
While these integrations can generally be achieved with fully manual code, orchestration frameworks such as LangChain and the IBM watsonx portfolio of artificial intelligence products greatly simplify the process. They also make it much easier to experiment with different LLMs to compare results, as different models can be swapped in and out with minimal changes to code.
"""
result = chain.invoke({'text': text})

print(result)
print(chain.get_graph().print_ascii())