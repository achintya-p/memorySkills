import json
from pathlib import Path
from dataclasses import asdict
from pipeline import PromptEntry, agent_step, view_memory, write_memory, run_test, generate_report, MEMORY, MODEL

# Implement the cuda parrallelism to speed up the sha-256 hashing
#client = openai.Client()
def write_prompt_history_to_jsonl(history: list[PromptEntry], file_path: Path):
    with file_path.open("a") as f:
        for entry in history:
            f.write(json.dumps(asdict(entry)) + "\n")

def repl():
    print(f"Memory Tester | Model: {MODEL}")
    print("Commands: /memory, /clear, /inject <text>, /test, /exit")
    
    DATA_DIR = Path(__file__).parent.parent / "storedData"
    DATA_DIR.mkdir(exist_ok=True)
    PROMPTS_FILE = DATA_DIR / "prompts.jsonl"
    
    prompt_history: list[PromptEntry] = []
    
    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input == "/exit":
            break
        elif user_input == "/memory":
            view_memory()
        elif user_input == "/clear":
            MEMORY.clear()
            print("Memory cleared.")
        elif user_input.startswith("/inject "):
            write_memory(user_input[8:])
        elif user_input == "/test":
            test_results = run_test(PROMPTS_FILE)
            generate_report(test_results)
        else:
            try:
                entry = agent_step(user_input)
                prompt_history.append(entry)
                print(f"\nAgent> {entry.agent_response}")
            except Exception as e:
                print(f"[ERROR] {e}")
    
    if prompt_history:
        write_prompt_history_to_jsonl(prompt_history, PROMPTS_FILE)

if __name__ == "__main__":
    repl()