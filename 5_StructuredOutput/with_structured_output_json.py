from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {"type": "array", "items": {"type": "string"},
                       "description": "Write down all the themes discussed in the review in a list."},
        "summary": {"type": "string", "description": "A brief summary of the review."},
        "sentiment": {"type": "string", "enum": ["pos", "neg"],
                      "description": "Return sentiment of the review either positive, negative or neutral."}
    }
}

structured_model = model.with_structured_output(json_schema)

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

print("Summary: ", result.summary)
print("Sentiment: ", result.sentiment)