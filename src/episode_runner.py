"""
Episode runner: execute standardized episodes with reproducible traces.

Features:
- Run episode sequences with labels (track, threat level, expected outputs)
- Per-turn event logging (POLICY_DECISION, RETRIEVE, WRITE, SKILL_SELECT, TOOL_CALL, OUTPUT, EVICT)
- Post-hoc attribution of failures to write/retrieve/apply
- JSONL trace output
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional, Literal
from enum import Enum


EventType = Literal[
    "POLICY_DECISION",
    "RETRIEVE",
    "WRITE",
    "UPDATE",
    "DELETE",
    "SKILL_SELECT",
    "TOOL_CALL",
    "OUTPUT",
    "EVICT",
    "ERROR",
]


@dataclass
class TraceEvent:
    """A single event in an execution trace."""
    timestamp: str
    event_type: EventType
    turn_number: int
    episode_id: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeTurn:
    """A single turn in an episode."""
    turn_number: int
    user_input: str
    agent_thought: Optional[str] = None
    agent_response: str = ""
    memory_writes: list[str] = field(default_factory=list)
    memory_reads: list[str] = field(default_factory=list)
    skill_used: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Episode:
    """A complete episode (sequence of turns)."""
    episode_id: str
    track_id: str  # "benign_prefs", "benign_skills", "r1_poison", "r2_persistence", etc.
    threat_level: str  # "none", "low", "medium", "high"
    title: str
    description: str
    initial_state: dict = field(default_factory=dict)  # Pre-populated memory
    expected_outputs: dict = field(default_factory=dict)  # Rubric checks
    turns: list[EpisodeTurn] = field(default_factory=list)
    success: bool = False
    failure_reason: Optional[str] = None
    failure_attribution: Optional[dict] = None  # {"write_fault", "retrieve_fault", "apply_fault"}

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "track_id": self.track_id,
            "threat_level": self.threat_level,
            "title": self.title,
            "description": self.description,
            "initial_state": self.initial_state,
            "expected_outputs": self.expected_outputs,
            "turns": [t.to_dict() for t in self.turns],
            "success": self.success,
            "failure_reason": self.failure_reason,
            "failure_attribution": self.failure_attribution,
        }


class EpisodeRunner:
    """Execute episodes and generate traces."""

    def __init__(self):
        self.episodes: list[Episode] = []
        self.traces: list[TraceEvent] = []

    def load_episodes_from_jsonl(self, jsonl_path: str) -> int:
        """Load episodes from JSONL. Returns count."""
        count = 0
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                ep = self._episode_from_dict(data)
                self.episodes.append(ep)
                count += 1
        return count

    def _episode_from_dict(self, d: dict) -> Episode:
        """Convert dict to Episode."""
        turns = []
        for t_data in d.get("turns", []):
            turn = EpisodeTurn(
                turn_number=t_data.get("turn_number", 0),
                user_input=t_data.get("user_input", ""),
                agent_thought=t_data.get("agent_thought"),
                agent_response=t_data.get("agent_response", ""),
                memory_writes=t_data.get("memory_writes", []),
                memory_reads=t_data.get("memory_reads", []),
                skill_used=t_data.get("skill_used"),
                tool_calls=t_data.get("tool_calls", []),
                errors=t_data.get("errors", []),
            )
            turns.append(turn)

        return Episode(
            episode_id=d.get("episode_id", "unknown"),
            track_id=d.get("track_id", "unknown"),
            threat_level=d.get("threat_level", "none"),
            title=d.get("title", ""),
            description=d.get("description", ""),
            initial_state=d.get("initial_state", {}),
            expected_outputs=d.get("expected_outputs", {}),
            turns=turns,
            success=d.get("success", False),
            failure_reason=d.get("failure_reason"),
            failure_attribution=d.get("failure_attribution"),
        )

    def add_episode(self, episode: Episode) -> None:
        """Add episode to runner."""
        self.episodes.append(episode)

    def add_trace_event(
        self,
        event_type: EventType,
        episode_id: str,
        turn_number: int,
        details: Optional[dict] = None,
    ) -> None:
        """Log a trace event."""
        event = TraceEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            turn_number=turn_number,
            episode_id=episode_id,
            details=details or {},
        )
        self.traces.append(event)

    def run_episode(
        self,
        episode: Episode,
        step_callback=None,  # Called for each turn: (episode, turn, results)
    ) -> Episode:
        """
        Execute an episode turn-by-turn.

        step_callback: function(episode, turn_number, step_result)
        """
        for i, turn in enumerate(episode.turns):
            step_result = {}

            # Callback
            if step_callback:
                step_callback(episode, i, step_result)

            # Log completion
            self.add_trace_event(
                event_type="OUTPUT",
                episode_id=episode.episode_id,
                turn_number=i,
                details={"response": turn.agent_response[:100]}
            )

        return episode

    def save_traces_to_jsonl(self, output_path: str) -> int:
        """Save all traces to JSONL. Returns count."""
        with open(output_path, "w") as f:
            for trace in self.traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        return len(self.traces)

    def save_episodes_to_jsonl(self, output_path: str) -> int:
        """Save all episodes to JSONL. Returns count."""
        with open(output_path, "w") as f:
            for episode in self.episodes:
                f.write(json.dumps(episode.to_dict()) + "\n")
        return len(self.episodes)

    def get_trace_events(self, episode_id: Optional[str] = None, last_n: Optional[int] = None) -> list[TraceEvent]:
        """Get trace events, optionally filtered."""
        events = self.traces
        if episode_id:
            events = [e for e in events if e.episode_id == episode_id]
        if last_n:
            events = events[-last_n:]
        return events

    def compute_attribution(self, episode: Episode) -> dict:
        """
        Post-hoc: analyze if failure was due to write, retrieve, or apply fault.

        Returns: {"write_fault": bool, "retrieve_fault": bool, "apply_fault": bool, "reason": str}
        """
        if episode.success:
            return {"write_fault": False, "retrieve_fault": False, "apply_fault": False, "reason": "success"}

        attribution = {"write_fault": False, "retrieve_fault": False, "apply_fault": False, "reason": ""}

        # Analyze turns
        for turn in episode.turns:
            if turn.errors:
                if any("memory" in err.lower() and "write" in err.lower() for err in turn.errors):
                    attribution["write_fault"] = True
                if any("retrieve" in err.lower() or "query" in err.lower() for err in turn.errors):
                    attribution["retrieve_fault"] = True
                if any("apply" in err.lower() or "execute" in err.lower() for err in turn.errors):
                    attribution["apply_fault"] = True

        if not any([attribution["write_fault"], attribution["retrieve_fault"], attribution["apply_fault"]]):
            attribution["reason"] = "unknown / no explicit errors"
        else:
            reasons = []
            if attribution["write_fault"]:
                reasons.append("write")
            if attribution["retrieve_fault"]:
                reasons.append("retrieve")
            if attribution["apply_fault"]:
                reasons.append("apply")
            attribution["reason"] = " + ".join(reasons)

        return attribution

    def get_summary(self) -> dict:
        """Compute summary stats across all episodes."""
        total = len(self.episodes)
        successful = sum(1 for ep in self.episodes if ep.success)
        failed = total - successful

        by_track = {}
        for ep in self.episodes:
            track = ep.track_id
            if track not in by_track:
                by_track[track] = {"total": 0, "successful": 0}
            by_track[track]["total"] += 1
            if ep.success:
                by_track[track]["successful"] += 1

        return {
            "total_episodes": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "by_track": by_track,
            "total_events": len(self.traces),
        }
