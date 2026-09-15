import json
import os
import time
from dotenv import load_dotenv
from pathlib import Path


from google import genai
from google.genai import types


PROMPT_PATH = Path("prompt/prompt.txt")
ITEMS_PATH = Path("data/items.json")
RESULTS_PATH = Path("results/cheap_results.json")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

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

    # Load existing results if they exist
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open("r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    # Keep successful results
    successful_ids = {
        result["id"]
        for result in results
        if result.get("status") == "success"
    }

    print(f"Already completed: {len(successful_ids)}")

    # Only run items that don't already have a successful result
    items_to_run = [
        item
        for item in items
        if item["id"] not in successful_ids
    ]

    print(f"Remaining: {len(items_to_run)}")

    for item in items_to_run:
        print(f"Running ID {item['id']}...")

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

            # Remove any previous failed result for this ID
            results = [
                r for r in results
                if r["id"] != item["id"]
            ]

            results.append({
                "id": item["id"],
                "output": output,
                "latency_ms": round(latency_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "status": "success",
            })

            print(f"ID {item['id']} completed successfully.")

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            error_message = str(e)

            # Keep the previous result if this was just a failed retry
            # Don't overwrite successful results.
            if not any(r["id"] == item["id"] for r in results):
                results.append({
                    "id": item["id"],
                    "output": "",
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "status": "error",
                    "error": error_message,
                })

            print(f"ID {item['id']} failed: {error_message}")

        # Save after EVERY request so progress isn't lost
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

        with RESULTS_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        time.sleep(50)  # Sleep for 50 seconds to avoid rate limits

    print(
        f"Saved {len(results)} results to {RESULTS_PATH}"
    )

if __name__ == "__main__":
    run_cheap()