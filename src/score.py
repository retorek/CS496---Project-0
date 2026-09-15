import json
import re
from pathlib import Path


ITEMS_PATH = Path("data/items.json")

RESULT_FILES = {
    #"top": Path("results/top_results.json"),
    "cheap": Path("results/cheap_results.json"),
    # "local": Path("results/local_results.json"),
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_regex(regex, item):
    """
    Test one generated regex against one test item.

    Returns:
        correct: bool
        reason: str
    """

    if not regex:
        return False, "empty_output"

    # Remove accidental markdown code fences.
    regex = regex.strip()

    if regex.startswith("```") and regex.endswith("```"):
        lines = regex.splitlines()

        if len(lines) >= 3:
            regex = "\n".join(lines[1:-1]).strip()

    # Check whether the generated regex is valid.
    try:
        pattern = re.compile(regex)
    except re.error:
        return False, "invalid_regex"

    # Every must_match string must match.
    for text in item["must_match"]:
        if pattern.search(text) is None:
            return False, f"failed_must_match:{text}"

    # Every must_not_match string must NOT match.
    for text in item["must_not_match"]:
        if pattern.search(text) is not None:
            return False, f"failed_must_not_match:{text}"

    return True, "correct"


def score_model(items, results):
    results_by_id = {
        result["id"]: result
        for result in results
    }

    scored = []

    for item in items:
        result = results_by_id.get(item["id"])

        if result is None:
            scored.append({
                "id": item["id"],
                "correct": False,
                "reason": "missing_result",
                "output": "",
            })
            continue

        if result.get("status") != "success":
            scored.append({
                "id": item["id"],
                "correct": False,
                "reason": result.get("status", "error"),
                "output": result.get("output", ""),
            })
            continue

        correct, reason = score_regex(
            result.get("output", ""),
            item,
        )

        scored.append({
            "id": item["id"],
            "correct": correct,
            "reason": reason,
            "output": result.get("output", ""),
        })

    return scored


def print_summary(model, scored):
    total = len(scored)
    correct = sum(x["correct"] for x in scored)

    print(
        f"{model}: {correct}/{total} "
        f"({correct / total * 100:.2f}%)"
    )

    wrong = [x for x in scored if not x["correct"]]

    if wrong:
        print("Wrong answers:")

        for result in wrong[:3]:
            print(
                f"  {result['id']}: "
                f"{result['reason']}"
            )


def main():
    items = load_json(ITEMS_PATH)

    for model, result_path in RESULT_FILES.items():
        results = load_json(result_path)

        scored = score_model(items, results)

        output_path = Path(
            f"results/{model}_scored.json"
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                scored,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print_summary(model, scored)


if __name__ == "__main__":
    main()