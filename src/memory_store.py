"""
Core MemoryStore with pluggable backends.

Provides:
- ListMemoryStore: naive recency/substring
- KVMemoryStore: keyed preferences, latest-write-wins
- Eviction policies: max entries, TTL, LRU
- Full logging for write/retrieve/update/evict
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Literal, Callable
from enum import Enum

NamespaceType = Literal["episodic", "semantic", "preferences", "tool_traces", "skills", "working"]
EvictionPolicy = Literal["lru", "oldest-first", "ttl"]


@dataclass
class MemoryEntry:
    """A single memory entry with metadata and metrics."""
    namespace: NamespaceType
    key: str
    value: str
    entry_id: str = field(default_factory=lambda: hashlib.sha256(
        f"{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    trust_score: float = 1.0  # For defense: how much to trust this entry
    source: str = "user"  # "user", "tool", "system", "attacker"

    def is_expired(self) -> bool:
        """Check if entry has exceeded TTL."""
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(self.timestamp)
        return (datetime.now() - created).total_seconds() > self.ttl_seconds

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryLog:
    """Log entry for all memory operations."""
    operation: Literal["write", "retrieve", "update", "evict", "delete"]
    timestamp: str
    entry_id: str
    entry_key: str
    namespace: NamespaceType
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryStoreBase(ABC):
    """Abstract base class for memory backends."""

    def __init__(self):
        self.log: list[MemoryLog] = []
        self._eviction_policy: EvictionPolicy = "lru"
        self._max_per_namespace: dict[NamespaceType, int] = {
            "episodic": 100,
            "semantic": 100,
            "preferences": 50,
            "tool_traces": 100,
            "skills": 50,
            "working": 20,
        }

    @abstractmethod
    def write(
        self,
        namespace: NamespaceType,
        key: str,
        value: str,
        metadata: Optional[dict] = None,
        ttl_seconds: Optional[int] = None,
        source: str = "user",
        trust_score: float = 1.0,
    ) -> str:
        """Write entry. Returns entry_id."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
        namespaces: Optional[list[NamespaceType]] = None,
    ) -> list[MemoryEntry]:
        """Retrieve top-k entries for query."""

    @abstractmethod
    def update(self, entry_id: str, new_value: str) -> bool:
        """Update entry by id. Returns success."""

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete entry by id. Returns success."""

    @abstractmethod
    def all_entries(self, namespace: Optional[NamespaceType] = None) -> list[MemoryEntry]:
        """Get all entries, optionally filtered by namespace."""

    @abstractmethod
    def evict(self, namespace: Optional[NamespaceType] = None) -> list[str]:
        """Evict entries according to policy. Returns evicted ids."""

    def _log_operation(self, op: MemoryLog):
        """Internal: log operation."""
        self.log.append(op)

    def get_logs(self, last_n: Optional[int] = None) -> list[MemoryLog]:
        """Retrieve operation logs."""
        if last_n is None:
            return self.log
        return self.log[-last_n:]


class ListMemoryStore(MemoryStoreBase):
    """Naive list-based backend: recency + substring matching."""

    def __init__(self):
        super().__init__()
        self.entries: list[MemoryEntry] = []

    def write(
        self,
        namespace: NamespaceType,
        key: str,
        value: str,
        metadata: Optional[dict] = None,
        ttl_seconds: Optional[int] = None,
        source: str = "user",
        trust_score: float = 1.0,
    ) -> str:
        """Write entry. Latest-write-wins for same key."""
        # Check if key already exists in namespace
        existing_idx = next(
            (i for i, e in enumerate(self.entries)
             if e.namespace == namespace and e.key == key),
            None
        )

        entry = MemoryEntry(
            namespace=namespace,
            key=key,
            value=value,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
            source=source,
            trust_score=trust_score,
        )

        if existing_idx is not None:
            # Update existing
            self.entries[existing_idx] = entry
            self._log_operation(MemoryLog(
                operation="update",
                timestamp=datetime.now().isoformat(),
                entry_id=entry.entry_id,
                entry_key=key,
                namespace=namespace,
                details={"action": "update_same_key"}
            ))
        else:
            # Add new
            self.entries.append(entry)
            self._log_operation(MemoryLog(
                operation="write",
                timestamp=datetime.now().isoformat(),
                entry_id=entry.entry_id,
                entry_key=key,
                namespace=namespace,
                details={"source": source, "trust_score": trust_score}
            ))

            # Check if we exceeded max per namespace
            ns_count = sum(1 for e in self.entries if e.namespace == namespace)
            if ns_count > self._max_per_namespace.get(namespace, 100):
                self.evict(namespace)

        return entry.entry_id

    def retrieve(
        self,
        query: str,
        k: int = 5,
        namespaces: Optional[list[NamespaceType]] = None,
    ) -> list[MemoryEntry]:
        """Substring match, sorted by recency."""
        query_lower = query.lower()

        # Filter by namespace
        candidates = [
            e for e in self.entries
            if (namespaces is None or e.namespace in namespaces)
            and not e.is_expired()
        ]

        # Score by substring match (how early the match, how long the match)
        scored = []
        for entry in candidates:
            value_lower = entry.value.lower()
            if query_lower in value_lower:
                # Score: position of match + length of entry
                pos = value_lower.index(query_lower)
                score = -pos - len(query_lower)  # Earlier matches score higher
                scored.append((score, entry))

        # Sort by score, then by recency
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        # Log retrieve
        result = [e for _, e in scored[:k]]
        for e in result:
            e.access_count += 1
            e.last_accessed = datetime.now().isoformat()

        self._log_operation(MemoryLog(
            operation="retrieve",
            timestamp=datetime.now().isoformat(),
            entry_id="multi",
            entry_key=query,
            namespace="working",
            details={"query": query, "k": k, "results": len(result)}
        ))

        return result

    def update(self, entry_id: str, new_value: str) -> bool:
        """Update entry by id."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                old_value = entry.value
                entry.value = new_value
                entry.timestamp = datetime.now().isoformat()
                self._log_operation(MemoryLog(
                    operation="update",
                    timestamp=datetime.now().isoformat(),
                    entry_id=entry_id,
                    entry_key=entry.key,
                    namespace=entry.namespace,
                    details={"old": old_value[:50], "new": new_value[:50]}
                ))
                return True
        return False

    def delete(self, entry_id: str) -> bool:
        """Delete entry by id."""
        initial_len = len(self.entries)
        self.entries = [e for e in self.entries if e.entry_id != entry_id]
        deleted = len(self.entries) < initial_len
        if deleted:
            self._log_operation(MemoryLog(
                operation="delete",
                timestamp=datetime.now().isoformat(),
                entry_id=entry_id,
                entry_key="unknown",
                namespace="working",
            ))
        return deleted

    def all_entries(self, namespace: Optional[NamespaceType] = None) -> list[MemoryEntry]:
        """Get all entries, optionally filtered by namespace."""
        if namespace is None:
            return self.entries[:]
        return [e for e in self.entries if e.namespace == namespace]

    def evict(self, namespace: Optional[NamespaceType] = None) -> list[str]:
        """Evict by LRU or oldest-first."""
        max_ns = self._max_per_namespace.get(namespace, 100) if namespace else 100

        if namespace:
            candidates = [e for e in self.entries if e.namespace == namespace]
            if len(candidates) <= max_ns:
                return []

            # LRU eviction
            candidates.sort(key=lambda e: e.last_accessed)
            to_evict = candidates[:len(candidates) - max_ns + 1]
        else:
            if len(self.entries) <= sum(self._max_per_namespace.values()):
                return []
            candidates = self.entries[:]
            candidates.sort(key=lambda e: e.last_accessed)
            to_evict = candidates[:10]  # Evict 10 at a time

        evicted_ids = [e.entry_id for e in to_evict]
        self.entries = [e for e in self.entries if e.entry_id not in evicted_ids]

        self._log_operation(MemoryLog(
            operation="evict",
            timestamp=datetime.now().isoformat(),
            entry_id="multi",
            entry_key="eviction",
            namespace=namespace or "all",
            details={"evicted_count": len(evicted_ids), "policy": self._eviction_policy}
        ))

        return evicted_ids


class KVMemoryStore(MemoryStoreBase):
    """KV-based backend: keyed preferences, latest-write-wins."""

    def __init__(self):
        super().__init__()
        self.store: dict[str, dict[str, MemoryEntry]] = {
            "episodic": {},
            "semantic": {},
            "preferences": {},
            "tool_traces": {},
            "skills": {},
            "working": {},
        }

    def write(
        self,
        namespace: NamespaceType,
        key: str,
        value: str,
        metadata: Optional[dict] = None,
        ttl_seconds: Optional[int] = None,
        source: str = "user",
        trust_score: float = 1.0,
    ) -> str:
        """Write entry. Latest-write-wins."""
        entry = MemoryEntry(
            namespace=namespace,
            key=key,
            value=value,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
            source=source,
            trust_score=trust_score,
        )

        is_update = key in self.store[namespace]
        self.store[namespace][key] = entry

        self._log_operation(MemoryLog(
            operation="write" if not is_update else "update",
            timestamp=datetime.now().isoformat(),
            entry_id=entry.entry_id,
            entry_key=key,
            namespace=namespace,
            details={"source": source, "is_update": is_update}
        ))

        # Check namespace size
        if len(self.store[namespace]) > self._max_per_namespace.get(namespace, 100):
            self.evict(namespace)

        return entry.entry_id

    def retrieve(
        self,
        query: str,
        k: int = 5,
        namespaces: Optional[list[NamespaceType]] = None,
    ) -> list[MemoryEntry]:
        """Keyword match on keys, sorted by trust_score."""
        query_lower = query.lower()
        candidates = []

        search_namespaces = namespaces or list(self.store.keys())
        for ns in search_namespaces:
            if ns not in self.store:
                continue
            for key, entry in self.store[ns].items():
                if not entry.is_expired() and query_lower in key.lower():
                    candidates.append(entry)

        # Sort by trust_score (desc), then recency
        candidates.sort(key=lambda e: (-e.trust_score, e.timestamp), reverse=True)

        result = candidates[:k]
        for e in result:
            e.access_count += 1
            e.last_accessed = datetime.now().isoformat()

        self._log_operation(MemoryLog(
            operation="retrieve",
            timestamp=datetime.now().isoformat(),
            entry_id="multi",
            entry_key=query,
            namespace="working",
            details={"query": query, "k": k, "results": len(result)}
        ))

        return result

    def update(self, entry_id: str, new_value: str) -> bool:
        """Update entry by id."""
        for ns_dict in self.store.values():
            for key, entry in ns_dict.items():
                if entry.entry_id == entry_id:
                    entry.value = new_value
                    entry.timestamp = datetime.now().isoformat()
                    self._log_operation(MemoryLog(
                        operation="update",
                        timestamp=datetime.now().isoformat(),
                        entry_id=entry_id,
                        entry_key=key,
                        namespace=entry.namespace,
                    ))
                    return True
        return False

    def delete(self, entry_id: str) -> bool:
        """Delete entry by id."""
        for ns_dict in self.store.values():
            for key in list(ns_dict.keys()):
                if ns_dict[key].entry_id == entry_id:
                    del ns_dict[key]
                    self._log_operation(MemoryLog(
                        operation="delete",
                        timestamp=datetime.now().isoformat(),
                        entry_id=entry_id,
                        entry_key=key,
                        namespace=ns_dict[key].namespace,
                    ))
                    return True
        return False

    def all_entries(self, namespace: Optional[NamespaceType] = None) -> list[MemoryEntry]:
        """Get all entries."""
        if namespace:
            return list(self.store[namespace].values())
        result = []
        for ns_dict in self.store.values():
            result.extend(ns_dict.values())
        return result

    def evict(self, namespace: Optional[NamespaceType] = None) -> list[str]:
        """Evict by LRU."""
        if namespace:
            ns_dict = self.store[namespace]
            max_size = self._max_per_namespace.get(namespace, 100)
            if len(ns_dict) <= max_size:
                return []

            entries = list(ns_dict.values())
            entries.sort(key=lambda e: e.last_accessed)
            to_evict = entries[:len(entries) - max_size + 1]
            evicted_ids = [e.entry_id for e in to_evict]

            for e in to_evict:
                # Find key by entry_id
                for k in list(ns_dict.keys()):
                    if ns_dict[k].entry_id == e.entry_id:
                        del ns_dict[k]
                        break
        else:
            # Global eviction
            all_entries = []
            for ns, ns_dict in self.store.items():
                all_entries.extend([(ns, k, v) for k, v in ns_dict.items()])

            all_entries.sort(key=lambda x: x[2].last_accessed)
            to_evict = all_entries[:10]
            evicted_ids = []

            for ns, key, entry in to_evict:
                del self.store[ns][key]
                evicted_ids.append(entry.entry_id)

        self._log_operation(MemoryLog(
            operation="evict",
            timestamp=datetime.now().isoformat(),
            entry_id="multi",
            entry_key="eviction",
            namespace=namespace or "all",
            details={"evicted_count": len(evicted_ids)}
        ))

        return evicted_ids
