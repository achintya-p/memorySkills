import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from memory_manager import memory_store, make_semantic_key
from memory_ranker import MemoryRanker

def test_memory_ranking():
    """Test quantitative memory ranking"""
    print("=" * 70)
    print("MEMORY RANKING TEST")
    print("=" * 70)
    
    memory_store.clear()
    ranker = MemoryRanker(
        recency_weight=0.3,
        frequency_weight=0.2,
        importance_weight=0.3,
        relevance_weight=0.2
    )
    
    # Create memories with different characteristics
    print("\n1. Writing memories with different properties...")
    
    # Memory A: Recent, high importance
    memA = memory_store.write(
        "semantic.latent",
        make_semantic_key("user", "pref", "color"),
        "Blue is my favorite color",
        importance=0.9,
        reason="user_stated"
    )
    print(f"   ✓ Memory A (high importance, fresh): {memA.memory_key_hash[:8]}")
    
    # Memory B: Old but frequently accessed
    memB = memory_store.write(
        "semantic.latent",
        make_semantic_key("system", "config", "timeout"),
        "API timeout is 30 seconds",
        importance=0.5,
        reason="configuration"
    )
    memB.metrics.access_count = 15  # Simulate frequent access
    memB.metrics.last_accessed = "2026-01-20T10:00:00"  # Old access
    print(f"   ✓ Memory B (frequently accessed, old): {memB.memory_key_hash[:8]}")
    
    # Memory C: Low importance, rarely used
    memC = memory_store.write(
        "semantic.latent",
        make_semantic_key("archive", "data", "old"),
        "This is old archived data",
        importance=0.2,
        reason="archive"
    )
    memC.metrics.access_count = 1
    print(f"   ✓ Memory C (low importance): {memC.memory_key_hash[:8]}")
    
    # Test ranking
    print("\n2. Scoring individual memories...")
    memories = [memA, memB, memC]
    
    for i, mem in enumerate(memories, 1):
        scores = ranker.compute_rank_score(mem.metrics)
        print(f"\n   Memory {chr(64+i)}:")
        print(f"      Total Score: {scores['total_score']:.3f}")
        print(f"      Recency:     {scores['components']['recency']:.3f}")
        print(f"      Frequency:   {scores['components']['frequency']:.3f}")
        print(f"      Importance:  {scores['components']['importance']:.3f}")
        print(f"      Confidence:  {scores['confidence']:.1%}")
    
    # Test ranking with sorting
    print("\n3. Ranking all memories...")
    ranked = ranker.rank_memories(memories, top_k=None)
    
    for rank, item in enumerate(ranked, 1):
        mem = item["memory"]
        score = item["score"]
        print(f"   {rank}. Score={score:.3f} | {mem.value[:40]}")
    
    # Test with filtering
    print("\n4. Filtering by min score (0.5)...")
    filtered = ranker.rank_memories(memories, min_score=0.5)
    print(f"   Memories above 0.5: {len(filtered)}/{len(memories)}")
    for item in filtered:
        print(f"      {item['score']:.3f} - {item['memory'].value[:40]}")
    
    print("\n" + "=" * 70)
    print("✓ Memory ranking test complete")
    print("=" * 70)

if __name__ == "__main__":
    test_memory_ranking()
