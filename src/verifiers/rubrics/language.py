from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field

from verifiers.models import Message
from verifiers.parsers.xml_parser import XMLParser

client = OpenAI(max_retries=3)


class LanguageRating(BaseModel):
    reasoning: str
    rating: int = Field(description="The rating of the language of the text, between 1 and 5.")


SCHEMA = LanguageRating.model_json_schema()

SYSTEM_PROMPT = """
You are a helpful assistant that reviews a text in English according to the following criteria:
- Grammar
- Punctuation
- Sentence structure
- Overall readability
You will be given a text and you need to rate it on a scale of 1 to 10, where 1 is the worst and 5 is the best.
Ignore the gaps (...) between sentences.

## Examples

### Example 1

Text: First, I need to find the name of the person who sang Beauty and the Beast with Celine Dion.
Rating: 5

### Example 2

Text: The information retrieved notdirectly state the university Elizabeth Harwood attended, but it mentions the Royal Northern College of Music. Now, Ishoudlve find when the Royal Northern College of Music was formed.
Rating: 3

### Example 3

Text: yea information got but but but .... [a\123-1] yes thinking blip blop
Rating: 1

Here is the schema for your response:
{SCHEMA}
"""


def rate_language(text: str) -> float:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-32B-Instruct",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object", "value": SCHEMA},
    )
    parsed = LanguageRating.model_validate_json(response.choices[0].message.content)
    return parsed.rating / 10


def language_reward_func(
    completions: List[List[Message]],
    **kwargs,
) -> List[float]:
    """Reward function that checks if the language of the completion follows grammar rules and is understandable."""
    tag = "think"
    parser = XMLParser(fields=[tag])
    rewards: List[float] = []
    for trajectory in completions:
        assistant_messages = [msg for msg in trajectory if msg["role"] == "assistant"]
        thinking_traces = [getattr(parser.parse(msg["content"]), tag, None) for msg in assistant_messages]
        thinking = "\n".join([t for t in thinking_traces if t is not None])
        rewards.append(rate_language(thinking))
    return rewards
