import os
import json
import re
from typing import Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from openai import OpenAI
from memory_manager import memory_store, make_semantic_key, make_working_key

@dataclass
class PromptEntry:
    timestamp: str
    user_input: str
    agent_response: str

# OpenAI Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You are an agent with memory. Respond with valid JSON only:
{"thought": "reasoning", "memory_write": null or "string", "final": "response"}"""

# Legacy MEMORY for compatibility
MEMORY: list[str] = []

def retrieve_memory(k: int = 5) -> list[str]:
    entries = memory_store.retrieve(k=k)
    return [e.value for e in entries]

def write_memory(text: str):
    MEMORY.append(text)
    memory_store.write(
        memory_type="working.token",
        canonical_key=make_working_key(thread_id="default", turn_range="current"),
        value=text,
        reason="user_stated"
    )

def view_memory():
    entries = memory_store.retrieve(k=10)
    for i, entry in enumerate(entries):
        print(f"  {i}: [{entry.memory_type}] {entry.value[:60]}")

def ollama(messages: list[dict]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")

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
