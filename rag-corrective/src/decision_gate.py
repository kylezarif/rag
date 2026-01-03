from typing import List, Literal, Tuple

from openai import OpenAI

from src.config import Settings

GateDecision = Literal["correct", "ambiguous", "incorrect"]


def grade_documents(settings: Settings, question: str, contexts: List[str]) -> GateDecision:
    """
    Use a lightweight model to grade retrieved chunks as correct/ambiguous/incorrect.
    """
    if not contexts:
        return "incorrect"

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = (
        "You are evaluating retrieved travel context for a question.\n"
        "Label as one of: Correct, Ambiguous, Incorrect.\n"
        "- Correct: The context clearly answers or supports the question.\n"
        "- Ambiguous: The context is related but incomplete, location-mismatched, or possibly outdated.\n"
        "- Incorrect: The context is unrelated, contradictory, or missing.\n\n"
        f"Question: {question}\n\n"
        f"Contexts:\n{format_contexts(contexts)}\n\n"
        "Answer with only one word: Correct, Ambiguous, or Incorrect."
    )
    response = client.chat.completions.create(
        model=settings.grader_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You grade retrieved passages."},
            {"role": "user", "content": prompt},
        ],
    )
    label = (response.choices[0].message.content or "").strip().lower()
    if "correct" in label:
        return "correct"
    if "ambiguous" in label:
        return "ambiguous"
    return "incorrect"


def format_contexts(contexts: List[str]) -> str:
    return "\n\n".join(f"- {ctx}" for ctx in contexts)
