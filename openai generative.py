import json
import random
from pydantic import BaseModel
import instructor
from openai import OpenAI

#configure
api_key = "OPEN_AI_API_KEY" 
client = instructor.from_openai(OpenAI(api_key=api_key))

# dataset 
with open(r"C:\Users\varni\OneDrive - The Pennsylvania State University\Senior\research\datasets\42_math_random_clean_messages.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#  100 problems
random.seed(42) #seed for reproducibility
subset = random.sample(data, 100)

# define response structure 
class Rating(BaseModel):
    interestingness: int
    explanation: str

prompt_template = """
You are a mathematics expert.
Rate the following math problem for how mathematically interesting it is.
Give an integer rating from 1 (not interesting) to 10 (extremely interesting),
and explain briefly why.

Math problem:
"{problem}"
"""

results = []

for example in subset:
    problem_id, problem_text = next(iter(example.items()))
    prompt = prompt_template.format(problem=problem_text)

    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=Rating,
        messages=[
            {"role": "system", "content": "You are a careful math problem evaluator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        temperature=0.0,
    )

    results.append({
        "id": problem_id,
        "problem": problem_text,
        "rating": response.interestingness,
        "explanation": response.explanation
    })

#output in .txt file

with open("problem_ratings_43_math.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"ID: {r['id']}\nRating: {r['rating']}\nExplanation: {r['explanation']}\n\n")
