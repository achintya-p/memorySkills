#!/usr/bin/env python
"""
Integration test: verify all components work together.

Run: python verify_installation.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test all module imports."""
    print("[1/7] Testing imports...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from src.memory_store import ListMemoryStore, KVMemoryStore
        from src.skill_registry import SkillRegistry
        from src.agent_policy import AgentPolicy
        from src.episode_runner import EpisodeRunner
        from src.evaluation_tracks import create_all_episodes
        from src.metrics import MetricsComputer
        from src.evaluation_reporter import EvaluationReporter
        from src.orchestrator import EvaluationOrchestrator
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_memory_store():
    """Test memory store functionality."""
    print("[2/7] Testing MemoryStore...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from src.memory_store import ListMemoryStore
        store = ListMemoryStore()
        
        # Write
        id1 = store.write("semantic", "test_key", "test_value", source="test")
        assert id1 is not None
        
        # Retrieve
        results = store.retrieve("test", k=5)
        assert len(results) > 0
        
        # Update
        assert store.update(id1, "updated_value")
        
        # Delete
        assert store.delete(id1)
        
        print("  ✓ MemoryStore functional")
        return True
    except Exception as e:
        print(f"  ✗ MemoryStore test failed: {e}")
        return False


def test_skill_registry():
    """Test skill registry."""
    print("[3/7] Testing SkillRegistry...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from src.skill_registry import SkillRegistry, SkillSpec
        
        registry = SkillRegistry()
        
        # Create and add skill
        spec = SkillSpec(
            name="test_skill",
            description="A test skill",
            allowed_tools=["tool_a"],
        )
        registry.add_skill(spec)
        
        # Get skill
        retrieved = registry.get("test_skill")
        assert retrieved is not None
        assert retrieved.name == "test_skill"
        
        # List skills
        skills = registry.list()
        assert "test_skill" in skills
        
        print("  ✓ SkillRegistry functional")
        return True
    except Exception as e:
        print(f"  ✗ SkillRegistry test failed: {e}")
        return False


def test_agent_policy():
    """Test agent policy."""
    print("[4/7] Testing AgentPolicy...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from src.memory_store import ListMemoryStore
        from src.skill_registry import SkillRegistry
        from src.agent_policy import AgentPolicy
        
        memory = ListMemoryStore()
        skills = SkillRegistry()
        policy = AgentPolicy(memory, skills)
        
        # Memory selection policy
        mem_policy = policy.select_memory_policy("What are my preferences?")
        assert mem_policy is not None
        assert mem_policy.k > 0
        
        # Prompt building
        prompt_pack = policy.build_prompt_pack("Test query")
        assert prompt_pack.system_prompt
        assert prompt_pack.user_message
        
        print("  ✓ AgentPolicy functional")
        return True
    except Exception as e:
        print(f"  ✗ AgentPolicy test failed: {e}")
        return False


def test_episode_runner():
    """Test episode runner."""
    print("[5/7] Testing EpisodeRunner...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from src.episode_runner import EpisodeRunner, Episode, EpisodeTurn
        
        runner = EpisodeRunner()
        
        # Create test episode
        episode = Episode(
            episode_id="test_001",
            track_id="test_track",
            threat_level="none",
            title="Test Episode",
            description="Test",
            turns=[
                EpisodeTurn(
                    turn_number=0,
                    user_input="Test input",
                    agent_response="Test response",
                )
            ]
        )
        
        runner.add_episode(episode)
        runner.add_trace_event("OUTPUT", "test_001", 0, {"test": "data"})
        
        events = runner.get_trace_events()
        assert len(events) > 0
        
        print("  ✓ EpisodeRunner functional")
        return True
    except Exception as e:
        print(f"  ✗ EpisodeRunner test failed: {e}")
        return False


def test_evaluation_tracks():
    """Test evaluation tracks."""
    print("[6/7] Testing Evaluation Tracks...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from src.evaluation_tracks import create_all_episodes
        
        episodes = create_all_episodes()
        assert len(episodes) > 0
        assert any("benign_" in ep.track_id for ep in episodes)
        assert any("r1_" in ep.track_id or "r2_" in ep.track_id for ep in episodes)
        
        print(f"  ✓ Evaluation tracks loaded ({len(episodes)} episodes)")
        return True
    except Exception as e:
        print(f"  ✗ Evaluation tracks test failed: {e}")
        return False


def test_full_orchestrator():
    """Test full orchestrator."""
    print("[7/7] Testing Orchestrator (mini run)...")
    try:
        import sys
        import io
        sys.path.insert(0, str(Path(__file__).parent))
        from src.orchestrator import EvaluationOrchestrator
        import tempfile
        
        # Suppress output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = EvaluationOrchestrator(
                memory_backend="list",
                workspace_dir=tmpdir,
            )
            orchestrator.setup_scenarios()
            assert len(orchestrator.episode_runner.episodes) > 0
        
        sys.stdout = old_stdout
        print("  ✓ Orchestrator functional")
        return True
    except Exception as e:
        sys.stdout = old_stdout
        print(f"  ✗ Orchestrator test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SKILLS + MEMORY TESTER: INSTALLATION VERIFICATION")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_memory_store,
        test_skill_registry,
        test_agent_policy,
        test_episode_runner,
        test_evaluation_tracks,
        test_full_orchestrator,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Unexpected error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nYou can now run: python src/orchestrator.py --full")
        print("="*60 + "\n")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total})")
        print("\nPlease check errors above and ensure dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
