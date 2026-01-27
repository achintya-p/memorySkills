import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal
from datetime import datetime
from memory_ranker import MemoryMetrics

MemoryType = Literal[
    "working.token",
    "semantic.latent",
    "episodic.latent",
    "procedural.latent",
    "parametric.model"
]

@dataclass
class MemoryEntry:
    memory_type: MemoryType
    memory_key_hash: str
    memory_id_hash: str
    canonical_key: str
    value: str
    embedding_model: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)
    metrics: MemoryMetrics = field(default_factory=MemoryMetrics)

def hash_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def make_memory_key_hash(memory_type: MemoryType, canonical_key: str) -> str:
    """Hash the stable slot (concept identifier)."""
    key_string = f"v1|{memory_type}|{canonical_key}"
    return hash_sha256(key_string)

def make_memory_id_hash(memory_key_hash: str, canonical_value: str, form_meta: Optional[dict] = None) -> str:
    """Hash the exact content (content-addressed)."""
    meta_str = json.dumps(form_meta or {}, sort_keys=True)
    id_string = f"v1|{memory_key_hash}|{canonical_value}|{meta_str}"
    return hash_sha256(id_string)

# --- Canonical Key Templates by Memory Type ---

def make_semantic_key(scope: str, entity: str, attribute: str) -> str:
    """Factual / Semantic: (scope, entity, attribute)"""
    return f"{scope}|{entity}|{attribute}"

def make_episodic_key(scope: str, event_type: str, time_bucket: str, participants: str) -> str:
    """Experiential / Episodic: (scope, event_type, time_bucket, participants)"""
    return f"{scope}|{event_type}|{time_bucket}|{participants}"

def make_procedural_key(scope: str, procedure_name: str, version: str) -> str:
    """Procedural: (scope, procedure_name, version)"""
    return f"{scope}|{procedure_name}|{version}"

def make_working_key(thread_id: str, turn_range: str) -> str:
    """Working: (thread_id, turn_range)"""
    return f"thread:{thread_id}|turns:{turn_range}"

# --- Memory Store with Observability ---

class MemoryStore:
    def __init__(self):
        self.store: dict[str, MemoryEntry] = {}  # memory_key_hash -> MemoryEntry
        self.log: list[dict] = []
    
    def write(
        self, 
        memory_type: MemoryType, 
        canonical_key: str, 
        value: str,
        reason: str = "user_stated",
        embedding_model: Optional[str] = None,
        form_meta: Optional[dict] = None,
        importance: float = 0.5
    ) -> MemoryEntry:
        """Write or update a memory entry."""
        key_hash = make_memory_key_hash(memory_type, canonical_key)
        id_hash = make_memory_id_hash(key_hash, value, form_meta)
        
        is_update = key_hash in self.store
        
        if is_update:
            # Update: preserve metrics, increment access count
            existing = self.store[key_hash]
            metrics = existing.metrics
            metrics.access_count += 1
            metrics.last_accessed = datetime.now().isoformat()
        else:
            # New: create fresh metrics
            metrics = MemoryMetrics(importance_score=importance)
        
        entry = MemoryEntry(
            memory_type=memory_type,
            memory_key_hash=key_hash,
            memory_id_hash=id_hash,
            canonical_key=canonical_key,
            value=value,
            embedding_model=embedding_model,
            metadata=form_meta or {},
            metrics=metrics
        )
        
        self.store[key_hash] = entry
        
        self.log.append({
            "action": "update" if is_update else "write",
            "memory_type": memory_type,
            "memory_key_hash": key_hash,
            "memory_id_hash": id_hash,
            "reason": reason,
            "timestamp": entry.timestamp
        })
        
        return entry
    
    def retrieve(self, memory_type: Optional[MemoryType] = None, k: int = 5) -> list[MemoryEntry]:
        """Retrieve memories, optionally filtered by type."""
        entries = list(self.store.values())
        
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        
        # Update access metrics
        for entry in entries:
            entry.metrics.access_count += 1
            entry.metrics.last_accessed = datetime.now().isoformat()
        
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        result = entries[:k]
        
        self.log.append({
            "action": "retrieve",
            "memory_type": memory_type,
            "candidate_count": len(result),
            "candidates": [
                {"memory_id_hash": e.memory_id_hash, "age": e.timestamp}
                for e in result
            ],
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_log(self) -> list[dict]:
        """Return observability log."""
        return self.log
    
    def clear(self):
        """Clear all memories."""
        self.store.clear()
        self.log.append({
            "action": "clear",
            "timestamp": datetime.now().isoformat()
        })

# Singleton instance
memory_store = MemoryStore()
