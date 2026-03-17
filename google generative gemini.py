import os
import json
import random
import time
from pydantic import BaseModel
from openai import OpenAI  # DeepSeek uses OpenAI-compatible client

# Configure DeepSeek 
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# dataset 
with open(r"C:\Users\varni\OneDrive - The Pennsylvania State University\Senior\research\datasets\42_math_random_clean_messages.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(42) #seed for reproducibility
subset = random.sample(data, 100)

# Rating schema
class Rating(BaseModel):
    interestingness: int
    explanation: str

prompt_template = """
You are a mathematics expert.
Rate the following math problem for how mathematically interesting it is.
Give an integer rating from 1 (not interesting) to 10 (extremely interesting),
and explain briefly why. 

Respond ONLY with valid JSON in this exact format:
{{"interestingness": <int>, "explanation": "<string>"}}

Math problem:
"{problem}"
"""

def call_with_retry(problem_id, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},  # Forces JSON output
                temperature=0.0,
                max_tokens=300,
            )

            raw = response.choices[0].message.content.strip()
            parsed = Rating.model_validate_json(raw)
            return parsed

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_seconds = 60
                print(f"[RATE LIMIT] ID {problem_id} - waiting {wait_seconds}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_seconds)
            else:
                print(f"[ERROR] ID {problem_id}: {e}")
                return None

    print(f"[FAILED] ID {problem_id} after {max_retries} retries.")
    return None

results = []
failed_ids = []

for i, example in enumerate(subset):
    problem_id, problem_text = next(iter(example.items()))
    prompt = prompt_template.format(problem=problem_text)

    print(f"[{i+1}/100] Processing ID {problem_id}...")
    parsed = call_with_retry(problem_id, prompt)

    if parsed is None:
        failed_ids.append((problem_id, problem_text))
    else:
        results.append({
            "id": problem_id,
            "problem": problem_text,
            "rating": parsed.interestingness,
            "explanation": parsed.explanation,
        })

    time.sleep(1)  

# retry failed items 
if failed_ids:
    print(f"\n[RETRY PASS] Waiting 30s before retrying {len(failed_ids)} failed items...")
    time.sleep(30)
    for problem_id, problem_text in failed_ids:
        prompt = prompt_template.format(problem=problem_text)
        print(f"[RETRY] Processing ID {problem_id}...")
        parsed = call_with_retry(problem_id, prompt)
        if parsed:
            results.append({
                "id": problem_id,
                "problem": problem_text,
                "rating": parsed.interestingness,
                "explanation": parsed.explanation,
            })
        time.sleep(1)

print(f"\nDone. {len(results)} rated, {len(failed_ids)} initially failed.")

#output in .txt file

with open("problem_ratings_42_math_deepseek.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"ID: {r['id']}\nRating: {r['rating']}\nExplanation: {r['explanation']}\n\n")
