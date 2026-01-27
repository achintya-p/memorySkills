import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class MemoryMetrics:
    """Quantitative metrics for memory ranking"""
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance_score: float = 0.5  # 0.0 to 1.0, user-set or inferred
    relevance_score: float = 0.0  # 0.0 to 1.0, context-dependent
    confidence: float = 0.8  # 0.0 to 1.0, how verified is this memory

class MemoryRanker:
    """Quantitative ranking for memory retrieval"""
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.2,
        importance_weight: float = 0.3,
        relevance_weight: float = 0.2,
        decay_days: float = 7.0
    ):
        """
        Args:
            recency_weight: Weight for time-based decay (0-1)
            frequency_weight: Weight for access frequency (0-1)
            importance_weight: Weight for user-set importance (0-1)
            relevance_weight: Weight for context matching (0-1)
            decay_days: Half-life for exponential decay
        """
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.importance_weight = importance_weight
        self.relevance_weight = relevance_weight
        self.decay_days = decay_days
        
        # Normalize weights to sum to 1.0
        total = sum([recency_weight, frequency_weight, importance_weight, relevance_weight])
        self.recency_weight /= total
        self.frequency_weight /= total
        self.importance_weight /= total
        self.relevance_weight /= total
    
    def score_recency(self, metrics: MemoryMetrics) -> float:
        """
        Score based on age with exponential decay.
        Recent memories score higher.
        
        Formula: score = 2^(-age_days / half_life)
        """
        try:
            last_accessed = datetime.fromisoformat(metrics.last_accessed)
            age_days = (datetime.now() - last_accessed).days
            recency = 2 ** (-age_days / self.decay_days)
            return max(0.0, min(1.0, recency))
        except:
            return 0.5
    
    def score_frequency(self, metrics: MemoryMetrics, max_freq: int = 100) -> float:
        """
        Score based on access frequency.
        Logarithmic scale to avoid dominance.
        
        Formula: score = log(access_count + 1) / log(max_freq + 1)
        """
        normalized = min(metrics.access_count, max_freq)
        freq_score = math.log(normalized + 1) / math.log(max_freq + 1)
        return max(0.0, min(1.0, freq_score))
    
    def score_importance(self, metrics: MemoryMetrics) -> float:
        """
        Direct score from user-assigned importance.
        Usually 0.0 (low) to 1.0 (high).
        """
        return max(0.0, min(1.0, metrics.importance_score))
    
    def score_relevance(self, metrics: MemoryMetrics, query_context: Optional[str] = None) -> float:
        """
        Score based on context matching.
        Can be pre-computed (metrics.relevance_score) or computed from query.
        """
        return max(0.0, min(1.0, metrics.relevance_score))
    
    def compute_rank_score(
        self, 
        metrics: MemoryMetrics,
        query_context: Optional[str] = None,
        confidence_threshold: float = 0.0
    ) -> dict:
        """
        Compute composite ranking score with component breakdown.
        
        Returns:
            {
                "total_score": float (0-1),
                "components": {
                    "recency": float,
                    "frequency": float,
                    "importance": float,
                    "relevance": float,
                    "confidence_adjusted": float
                },
                "confidence": float
            }
        """
        recency = self.score_recency(metrics)
        frequency = self.score_frequency(metrics)
        importance = self.score_importance(metrics)
        relevance = self.score_relevance(metrics, query_context)
        
        # Apply confidence as a multiplier (penalize low-confidence memories)
        confidence_factor = metrics.confidence
        
        total_score = (
            recency * self.recency_weight +
            frequency * self.frequency_weight +
            importance * self.importance_weight +
            relevance * self.relevance_weight
        ) * confidence_factor
        
        return {
            "total_score": max(0.0, min(1.0, total_score)),
            "components": {
                "recency": recency,
                "frequency": frequency,
                "importance": importance,
                "relevance": relevance
            },
            "confidence_adjusted": total_score,
            "confidence": metrics.confidence
        }
    
    def rank_memories(
        self,
        memories: list,
        query_context: Optional[str] = None,
        min_score: float = 0.0,
        top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Rank a list of memories by score.
        
        Args:
            memories: List of memory objects with .metrics attribute
            query_context: Optional context string for relevance scoring
            min_score: Filter out memories below this score
            top_k: Return only top K results
        
        Returns:
            Sorted list of {memory, score_breakdown}
        """
        scored = []
        
        for memory in memories:
            if not hasattr(memory, 'metrics'):
                continue
            
            score_data = self.compute_rank_score(
                memory.metrics, 
                query_context=query_context
            )
            
            if score_data["total_score"] >= min_score:
                scored.append({
                    "memory": memory,
                    "score": score_data["total_score"],
                    "breakdown": score_data
                })
        
        # Sort by total score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        if top_k:
            scored = scored[:top_k]
        
        return scored
    
    def explain_ranking(self, scored_memory: dict) -> str:
        """Human-readable explanation of why a memory ranked as it did"""
        breakdown = scored_memory["breakdown"]
        score = scored_memory["score"]
        
        lines = [
            f"Overall Score: {score:.3f}",
            f"Confidence: {breakdown['confidence']:.1%}",
            "Component Scores:",
            f"  Recency:   {breakdown['components']['recency']:.3f}",
            f"  Frequency: {breakdown['components']['frequency']:.3f}",
            f"  Importance: {breakdown['components']['importance']:.3f}",
            f"  Relevance:  {breakdown['components']['relevance']:.3f}"
        ]
        
        return "\n".join(lines)
