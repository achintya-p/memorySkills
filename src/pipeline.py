import requests
import json
import re
from typing import Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class PromptEntry:
    timestamp: str
    user_input: str
    agent_response: str

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"
SYSTEM_PROMPT = """You are an agent with memory. Respond with valid JSON only:
{"thought": "reasoning", "memory_write": null or "string", "final": "response"}"""

MEMORY: list[str] = []

def retrieve_memory(k: int = 5) -> list[str]:
    return MEMORY[-k:]

def write_memory(text: str):
    MEMORY.append(text)

def view_memory():
    for i, m in enumerate(MEMORY):
        print(f"  {i}: {m}")

def ollama(messages: list[dict]) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={"model": MODEL, "messages": messages, "stream": False}, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama not running? Try: ollama serve")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")

def extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'(\{.*\})']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None

def agent_step(user_input: str) -> PromptEntry:
    mem = retrieve_memory()
    mem_text = "\n".join([f"- {m}" for m in mem]) if mem else "(none)"
    timestamp = datetime.now().isoformat()
    
    user_msg = f"MEMORY:\n{mem_text}\n\nUSER INPUT: {user_input}"
    raw = ollama([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    
    data = extract_json(raw)
    if data is None:
        return PromptEntry(timestamp=timestamp, user_input=user_input, agent_response="[JSON parse failed]")
    
    if data.get("memory_write"):
        write_memory(data["memory_write"])
    
    return PromptEntry(timestamp=timestamp, user_input=user_input, agent_response=data.get("final", ""))

def run_test(prompt_file: Path) -> list[dict]:
    if not prompt_file.exists():
        return []
    results = []
    with prompt_file.open("r") as f:
        for line in f:
            entry_data = json.loads(line)
            test_entry = agent_step(entry_data["user_input"])
            results.append({"original": entry_data, "test": asdict(test_entry)})
    return results

def generate_report(results: list[dict]):
    if not results:
        print("No test results.")
        return
    total = len(results)
    print(f"\nTest Report - Total: {total}")
