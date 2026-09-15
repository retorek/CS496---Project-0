import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types


PROMPT_PATH = Path("prompt/prompt.txt")
ITEMS_PATH = Path("data/items.json")
RESULTS_PATH = Path("results/cheap_results.json")

MODEL = os.environ["CHEAP_MODEL"]


def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_items():
    with ITEMS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(prompt_template, specification):
    return prompt_template.replace(
        "{specification}",
        specification,
    )


def run_cheap():
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    prompt_template = load_prompt()
    items = load_items()

    results = []

    for item in items:
        prompt = build_prompt(
            prompt_template,
            item["specification"],
        )

        start = time.perf_counter()

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=256,
                ),
            )

            latency_ms = (time.perf_counter() - start) * 1000

            output = response.text.strip()

            usage = response.usage_metadata

            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0

            results.append({
                "id": item["id"],
                "output": output,
                "latency_ms": round(latency_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "status": "success",
            })

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            results.append({
                "id": item["id"],
                "output": "",
                "latency_ms": round(latency_ms, 2),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "status": "error",
                "error": str(e),
            })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved {len(results)} results to {RESULTS_PATH}"
    )


if __name__ == "__main__":
    run_cheap()