from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema

class Review(TypedDict):
    key_theme: Annotated[list[str], "Write down all the themes discussed in the review in a list."]
    summary: Annotated[str, "A brief summary of the review."]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either positive, negative or neutral."]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list."]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list."]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(("""
The Samsung Galaxy S24 is a compact flagship smartphone that delivers strong performance, excellent cameras, and long software support. Its premium design and bright AMOLED display make it a great choice for everyday use, gaming, and photography.

Pros
Excellent AMOLED display with vibrant colors and smooth 120Hz refresh rate
Powerful performance for gaming and multitasking
Great camera quality, especially in daylight and portrait shots
Premium and compact design that feels comfortable in hand
Long battery life with fast charging support
Samsung promises several years of Android updates

Cons
Charging speed is slower compared to some competitors
No charger included in the box
Slight heating during heavy gaming sessions
Price is on the higher side for casual users
Final Verdict

The Samsung Galaxy S24 is a reliable premium smartphone with balanced features, strong cameras, and excellent software support. It’s ideal for users who want a compact flagship experience without major compromises.
"""))

print(result)

print("Summary: ", result['summary'])
print("Sentiment: ", result['sentiment'])