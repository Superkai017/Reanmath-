"""Ad-hoc manual checks -- run with `python -m reanmath.testing`."""
from reanmath.llm_client import call_claude
from reanmath.prompts import SYSTEM_PROMPT, build_augmented_prompt
from reanmath.retrieval.retriever import retrieve_context


def run_sample_query(question: str) -> None:
    contexts = retrieve_context(question)
    print(f"--- retrieved {len(contexts)} chunk(s) ---")
    for c in contexts:
        print(c[:120].replace("\n", " "), "...")

    system = build_augmented_prompt(SYSTEM_PROMPT, contexts)
    reply = call_claude(system=system, message=question)
    print("\n--- reply ---")
    print(reply)


if __name__ == "__main__":
    run_sample_query("សូមពន្យល់ពីលីមីតនៃអនុគមន៍")
