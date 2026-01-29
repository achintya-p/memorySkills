"""
Main orchestrator: integrate all components for end-to-end evaluation.

Usage:
    python orchestrator.py --run-episodes --save-traces --generate-report
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from memory_store import ListMemoryStore, KVMemoryStore, MemoryStoreBase
    from skill_registry import SkillRegistry, create_default_skills_yaml
    from agent_policy import AgentPolicy
    from episode_runner import EpisodeRunner
    from evaluation_tracks import create_all_episodes, save_episodes_to_jsonl
    from metrics import MetricsComputer
    from evaluation_reporter import EvaluationReporter
except ImportError:
    from .memory_store import ListMemoryStore, KVMemoryStore, MemoryStoreBase
    from .skill_registry import SkillRegistry, create_default_skills_yaml
    from .agent_policy import AgentPolicy
    from .episode_runner import EpisodeRunner
    from .evaluation_tracks import create_all_episodes, save_episodes_to_jsonl
    from .metrics import MetricsComputer
    from .evaluation_reporter import EvaluationReporter


class EvaluationOrchestrator:
    """Main orchestrator for skills + memory evaluation."""

    def __init__(
        self,
        memory_backend: str = "list",  # "list" or "kv"
        skills_yaml_path: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ):
        self.workspace_dir = Path(workspace_dir or Path.cwd())
        self.data_dir = self.workspace_dir / "evaluationData"
        self.data_dir.mkdir(exist_ok=True)

        # Initialize memory store
        if memory_backend == "kv":
            self.memory_store: MemoryStoreBase = KVMemoryStore()
        else:
            self.memory_store = ListMemoryStore()

        # Initialize skill registry
        self.skill_registry = SkillRegistry()
        
        # Use provided YAML or create default
        if skills_yaml_path and Path(skills_yaml_path).exists():
            self.skill_registry.load_from_yaml(skills_yaml_path)
        else:
            default_yaml = self.data_dir / "skills.yaml"
            if not default_yaml.exists():
                create_default_skills_yaml(str(default_yaml))
            self.skill_registry.load_from_yaml(str(default_yaml))

        # Initialize agent policy
        self.agent_policy = AgentPolicy(self.memory_store, self.skill_registry)

        # Initialize episode runner
        self.episode_runner = EpisodeRunner()

    def setup_scenarios(self):
        """Load or create evaluation scenarios."""
        episodes_file = self.data_dir / "episodes.jsonl"
        
        if not episodes_file.exists():
            # Create default episodes
            episodes = create_all_episodes()
            count = save_episodes_to_jsonl(str(episodes_file))
            print(f"[INFO] Created {count} evaluation episodes")
        
        # Load episodes
        count = self.episode_runner.load_episodes_from_jsonl(str(episodes_file))
        print(f"[INFO] Loaded {count} evaluation episodes")

    def run_all_episodes_simulation(self):
        """
        Simulate running all episodes.
        
        In real use, this would call agent LLM for each turn.
        Here, we simulate with deterministic logic.
        """
        print("\n" + "="*60)
        print("RUNNING EPISODE SIMULATIONS")
        print("="*60)

        for episode in self.episode_runner.episodes:
            print(f"\n[Episode: {episode.episode_id}]")
            print(f"  Track: {episode.track_id}")
            print(f"  Threat: {episode.threat_level}")
            print(f"  Description: {episode.description[:60]}...")

            # Simulate pre-populated memory
            for key, value in episode.initial_state.items():
                parts = key.split("|")
                namespace = "semantic"  # Default
                self.memory_store.write(namespace, key, value, source="system")

            # Simulate each turn
            success = True
            for turn in episode.turns:
                print(f"    Turn {turn.turn_number}: {turn.user_input[:40]}...")

                # 1. Policy decision
                prompt_pack, decision_info = self.agent_policy.decide(turn.user_input)
                self.episode_runner.add_trace_event(
                    event_type="POLICY_DECISION",
                    episode_id=episode.episode_id,
                    turn_number=turn.turn_number,
                    details=decision_info,
                )

                # 2. Simulate memory operations
                for write in turn.memory_writes:
                    self.memory_store.write("semantic", write, "written", source="agent")
                    self.episode_runner.add_trace_event(
                        event_type="WRITE",
                        episode_id=episode.episode_id,
                        turn_number=turn.turn_number,
                        details={"key": write},
                    )

                for read in turn.memory_reads:
                    entries = self.memory_store.retrieve(read, k=3)
                    self.episode_runner.add_trace_event(
                        event_type="RETRIEVE",
                        episode_id=episode.episode_id,
                        turn_number=turn.turn_number,
                        details={"query": read, "results": len(entries)},
                    )

                # 3. Skill selection
                if turn.skill_used:
                    skill = self.skill_registry.get(turn.skill_used)
                    if skill:
                        self.episode_runner.add_trace_event(
                            event_type="SKILL_SELECT",
                            episode_id=episode.episode_id,
                            turn_number=turn.turn_number,
                            details={"skill": turn.skill_used},
                        )

                # 4. Tool calls
                for tool_call in turn.tool_calls:
                    self.episode_runner.add_trace_event(
                        event_type="TOOL_CALL",
                        episode_id=episode.episode_id,
                        turn_number=turn.turn_number,
                        details=tool_call,
                    )

                # 5. Check for errors
                if turn.errors:
                    success = False
                    for error in turn.errors:
                        self.episode_runner.add_trace_event(
                            event_type="ERROR",
                            episode_id=episode.episode_id,
                            turn_number=turn.turn_number,
                            details={"error": error},
                        )

            # Evaluate episode success
            episode.success = self._check_episode_success(episode)
            if not episode.success:
                episode.failure_attribution = self.episode_runner.compute_attribution(episode)

            print(f"    Result: {'✓ PASS' if episode.success else '✗ FAIL'}")

    def _check_episode_success(self, episode) -> bool:
        """Check if episode outputs match expected."""
        if not episode.expected_outputs:
            return True

        matched = 0
        for key, expected in episode.expected_outputs.items():
            for turn in episode.turns:
                if isinstance(expected, str) and expected in turn.agent_response:
                    matched += 1
                    break

        return matched >= len(episode.expected_outputs) * 0.7

    def compute_metrics(self):
        """Compute all metrics."""
        print("\n" + "="*60)
        print("COMPUTING METRICS")
        print("="*60)

        track_metrics = MetricsComputer.compute_track_metrics(self.episode_runner.episodes)

        for track_id, metrics in track_metrics.items():
            print(f"\n[{track_id}]")
            print(f"  Success Rate: {metrics.success_rate:.1%}")
            if "benign" in track_id:
                print(f"  Avg Task Completion: {metrics.avg_task_completion:.1%}")
                print(f"  Avg Consistency: {metrics.avg_consistency:.1%}")
            elif "r1_" in track_id or "r2_" in track_id:
                print(f"  Avg Attack Success Rate: {metrics.avg_attack_success:.1%}")
                print(f"  Poison Utilization: {metrics.avg_poison_utilization:.1%}")

        return track_metrics

    def generate_report(self, output_json: Optional[str] = None, output_html: Optional[str] = None):
        """Generate evaluation report."""
        print("\n" + "="*60)
        print("GENERATING REPORT")
        print("="*60)

        reporter = EvaluationReporter(self.episode_runner)

        if output_json:
            report = reporter.generate_report(output_json)
            print(f"[INFO] JSON report saved to {output_json}")
        else:
            report = reporter.generate_report()

        if output_html:
            reporter.generate_html_report(output_html, report)
            print(f"[INFO] HTML report saved to {output_html}")

        return report

    def save_traces(self, output_jsonl: Optional[str] = None):
        """Save execution traces."""
        if output_jsonl:
            count = self.episode_runner.save_traces_to_jsonl(output_jsonl)
            print(f"[INFO] Saved {count} trace events to {output_jsonl}")

    def run_full_evaluation(self):
        """End-to-end evaluation."""
        print("\n" + "="*60)
        print("SKILLS + MEMORY TESTER EVALUATION")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*60)

        # 1. Setup
        self.setup_scenarios()

        # 2. Run episodes
        self.run_all_episodes_simulation()

        # 3. Save traces
        traces_file = self.data_dir / f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.save_traces(str(traces_file))

        # 4. Metrics
        self.compute_metrics()

        # 5. Report
        report_json = self.data_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_html = self.data_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report = self.generate_report(str(report_json), str(report_html))

        # 6. Summary
        summary = self.episode_runner.get_summary()
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']:.1%})")
        print(f"Failed: {summary['failed']}")
        print(f"Total Trace Events: {summary['total_events']}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Skills + Memory Tester Evaluation")
    parser.add_argument(
        "--memory-backend",
        choices=["list", "kv"],
        default="list",
        help="Memory store backend",
    )
    parser.add_argument(
        "--skills-yaml",
        help="Path to skills.yaml",
    )
    parser.add_argument(
        "--workspace",
        default=Path.cwd(),
        help="Workspace directory",
    )
    parser.add_argument(
        "--run-episodes",
        action="store_true",
        help="Run episode simulations",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save execution traces",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate evaluation report",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full end-to-end evaluation",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(
        memory_backend=args.memory_backend,
        skills_yaml_path=args.skills_yaml,
        workspace_dir=args.workspace,
    )

    if args.full:
        orchestrator.run_full_evaluation()
    else:
        if args.run_episodes:
            orchestrator.setup_scenarios()
            orchestrator.run_all_episodes_simulation()

        if args.save_traces:
            traces_file = Path(args.workspace) / "evaluationData" / "traces.jsonl"
            orchestrator.save_traces(str(traces_file))

        if args.generate_report:
            report_dir = Path(args.workspace) / "evaluationData"
            orchestrator.generate_report(
                str(report_dir / "report.json"),
                str(report_dir / "report.html"),
            )


if __name__ == "__main__":
    main()
