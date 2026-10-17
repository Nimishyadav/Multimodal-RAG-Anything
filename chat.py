"""Terminal Q&A using the same grounded retrieval path as Streamlit."""
import os
import sys
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parent)
    from src.qa import answer_question

    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("Ask a question: ").strip()
    if not query:
        print("Please enter a question.")
        return
    answer, sources, _ = answer_question(query)
    print(f"\nAnswer:\n\n{answer}")
    if sources:
        print("\nSources used:")
        for source in sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()
