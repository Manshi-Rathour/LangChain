from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Bihar?")

print(result.content)