"""
SkillRegistry with YAML-defined skills and heuristic router.

Features:
- YAML skill definitions with name, description, tools, preconditions, I/O schema
- Safety constraint validation
- Heuristic router: intent keywords + precondition gating
- Extensive logging of skill selection + scores
"""

import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Tuple, List
from enum import Enum
from datetime import datetime


@dataclass
class IOSchema:
    """Input/output schema for a skill."""
    inputs: dict = field(default_factory=dict)  # field_name -> type
    outputs: dict = field(default_factory=dict)


@dataclass
class SkillSpec:
    """Specification for a single skill."""
    name: str
    description: str
    allowed_tools: List[str] = field(default_factory=list)
    preconditions: dict = field(default_factory=dict)  # e.g., {"memory_required": ["episodic"]}
    io_schema: IOSchema = field(default_factory=IOSchema)
    safety_constraints: List[str] = field(default_factory=list)  # e.g., ["no_external_calls", "no_code_execution"]
    version: str = "1.0"

    @staticmethod
    def from_dict(d: dict) -> "SkillSpec":
        """Load from dict (parsed YAML)."""
        return SkillSpec(
            name=d.get("name", "unknown"),
            description=d.get("description", ""),
            allowed_tools=d.get("allowed_tools", []),
            preconditions=d.get("preconditions", {}),
            io_schema=IOSchema(
                inputs=d.get("io_schema", {}).get("inputs", {}),
                outputs=d.get("io_schema", {}).get("outputs", {}),
            ),
            safety_constraints=d.get("safety_constraints", []),
            version=d.get("version", "1.0"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SkillRouterScore:
    """Score for a skill during routing."""
    skill_name: str
    score: float
    rationale: dict[str, Any] = field(default_factory=dict)
    preconditions_met: bool = False
    safety_passed: bool = True


@dataclass
class RouterLog:
    """Log entry for skill selection."""
    timestamp: str
    query: str
    candidates: List[str]
    scores: List["SkillRouterScore"]
    selected: Optional[str]
    confidence: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        scores_dicts = [
            {
                "skill_name": s.skill_name,
                "score": s.score,
                "rationale": s.rationale,
                "preconditions_met": s.preconditions_met,
                "safety_passed": s.safety_passed,
            }
            for s in self.scores
        ]
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "candidates": self.candidates,
            "scores": scores_dicts,
            "selected": self.selected,
            "confidence": self.confidence,
            "details": self.details,
        }


class SkillRegistry:
    """Registry of available skills with router."""

    def __init__(self):
        self.skills: dict = {}
        self.router_logs: List = []

    def load_from_yaml(self, yaml_path: str) -> int:
        """Load skills from YAML file. Returns count loaded."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            return 0

        skills_data = data.get("skills", [])
        for skill_data in skills_data:
            spec = SkillSpec.from_dict(skill_data)
            self.skills[spec.name] = spec

        return len(skills_data)

    def add_skill(self, spec: SkillSpec) -> None:
        """Add skill programmatically."""
        self.skills[spec.name] = spec

    def get(self, skill_name: str) -> Optional[SkillSpec]:
        """Get skill by name."""
        return self.skills.get(skill_name)

    def list(self) -> list[str]:
        """List all skill names."""
        return list(self.skills.keys())

    def validate_skill(self, spec: SkillSpec) -> Tuple[bool, List[str]]:
        """Validate skill spec. Returns (is_valid, errors)."""
        errors = []

        if not spec.name:
            errors.append("Skill must have a name")
        if not spec.description:
            errors.append("Skill must have a description")

        # Check for required fields in preconditions
        for key in spec.preconditions.keys():
            if key not in ["memory_required", "user_context_required"]:
                errors.append(f"Unknown precondition key: {key}")

        # Check safety constraints
        valid_constraints = {"no_external_calls", "no_code_execution", "no_memory_write", "no_system_access"}
        for constraint in spec.safety_constraints:
            if constraint not in valid_constraints:
                errors.append(f"Unknown safety constraint: {constraint}")

        return len(errors) == 0, errors

    def select_skill(
        self,
        query: str,
        memory_available: Optional[dict] = None,
    ) -> Tuple[Optional[SkillSpec], SkillRouterScore, "RouterLog"]:
        """
        Heuristic routing: score all skills, return top one.

        query: user input / intent
        memory_available: dict of namespace -> entry count, used for precondition checking
        """
        memory_available = memory_available or {}
        candidates = list(self.skills.keys())
        scores: list[SkillRouterScore] = []

        for skill_name in candidates:
            spec = self.skills[skill_name]
            score = self._score_skill(query, spec, memory_available)
            scores.append(score)

        # Sort by score (descending)
        scores.sort(key=lambda s: s.score, reverse=True)

        selected = scores[0] if scores and scores[0].score > 0 else None
        selected_name = selected.skill_name if selected else None
        confidence = (selected.score / 100.0) if selected else 0.0

        log = RouterLog(
            timestamp=datetime.now().isoformat(),
            query=query,
            candidates=candidates,
            scores=scores,
            selected=selected_name,
            confidence=confidence,
            details={
                "selection_method": "heuristic_keyword_match",
                "top_score": scores[0].score if scores else 0,
            }
        )

        self.router_logs.append(log)

        selected_spec = self.skills.get(selected_name) if selected_name else None
        return selected_spec, selected, log

    def _score_skill(
        self,
        query: str,
        spec: SkillSpec,
        memory_available: dict,
    ) -> SkillRouterScore:
        """Score a skill for a given query."""
        score = 0.0
        rationale = {}

        query_lower = query.lower()
        desc_lower = spec.description.lower()

        # 1. Keyword matching on description
        keywords = desc_lower.split()
        query_keywords = query_lower.split()
        keyword_matches = sum(1 for qk in query_keywords if qk in keywords)
        keyword_score = (keyword_matches / len(query_keywords)) * 50 if query_keywords else 0
        score += keyword_score
        rationale["keyword_match"] = keyword_matches

        # 2. Check preconditions
        preconditions_met = self._check_preconditions(spec, memory_available)
        if preconditions_met:
            score += 30
            rationale["preconditions_bonus"] = 30
        else:
            score -= 20
            rationale["preconditions_penalty"] = -20

        # 3. Safety constraints
        safety_passed = True  # For now, accept all; later add safety scoring
        rationale["safety_passed"] = safety_passed

        return SkillRouterScore(
            skill_name=spec.name,
            score=score,
            rationale=rationale,
            preconditions_met=preconditions_met,
            safety_passed=safety_passed,
        )

    def _check_preconditions(self, spec: SkillSpec, memory_available: dict) -> bool:
        """Check if preconditions are met."""
        preconds = spec.preconditions

        # Check memory_required
        if "memory_required" in preconds:
            required = preconds["memory_required"]
            for ns in required:
                if memory_available.get(ns, 0) == 0:
                    return False

        # Check user_context_required
        if "user_context_required" in preconds:
            if not preconds["user_context_required"]:
                return False

        return True

    def get_router_logs(self, last_n: Optional[int] = None) -> List["RouterLog"]:
        """Get router logs."""
        if last_n is None:
            return self.router_logs
        return self.router_logs[-last_n:]


# Default skills YAML content
DEFAULT_SKILLS_YAML = """
skills:
  - name: "memory_recall"
    description: "Retrieve and recall facts from memory"
    allowed_tools: []
    preconditions:
      memory_required: ["semantic", "episodic"]
    io_schema:
      inputs:
        query: "string"
      outputs:
        recalled_facts: "list[string]"
    safety_constraints: []
    version: "1.0"

  - name: "tool_use"
    description: "Select and use an appropriate tool"
    allowed_tools: ["calculator", "web_search", "file_read"]
    preconditions: {}
    io_schema:
      inputs:
        intent: "string"
        tool_args: "dict"
      outputs:
        tool_result: "any"
    safety_constraints: ["no_code_execution"]
    version: "1.0"

  - name: "procedural_execute"
    description: "Execute a stored procedure or workflow"
    allowed_tools: []
    preconditions:
      memory_required: ["skills"]
    io_schema:
      inputs:
        procedure_name: "string"
        params: "dict"
      outputs:
        result: "any"
    safety_constraints: ["no_external_calls"]
    version: "1.0"

  - name: "preference_update"
    description: "Store or update user preferences"
    allowed_tools: []
    preconditions: {}
    io_schema:
      inputs:
        preference_key: "string"
        preference_value: "string"
      outputs:
        success: "bool"
    safety_constraints: ["no_system_access"]
    version: "1.0"
"""


def create_default_skills_yaml(output_path: str) -> None:
    """Create default skills.yaml in project."""
    with open(output_path, "w") as f:
        f.write(DEFAULT_SKILLS_YAML)
