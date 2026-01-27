import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from memory_manager import (
    memory_store, 
    make_semantic_key,
    make_episodic_key,
    make_procedural_key,
    make_working_key,
    make_memory_key_hash,
    make_memory_id_hash
)

class TestRunner:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def test_hash_stability(self):
        """Test: Hash remains stable for identical content"""
        key = make_semantic_key("test", "entity", "attr")
        value = "test value"
        
        hash1 = make_memory_key_hash("semantic.latent", key)
        hash2 = make_memory_key_hash("semantic.latent", key)
        
        if hash1 == hash2:
            self.passed += 1
            self.results.append({"test": "hash_stability", "status": "PASS"})
        else:
            self.failed += 1
            self.results.append({"test": "hash_stability", "status": "FAIL", "error": "Hashes differ"})
    
    def test_content_hash_change(self):
        """Test: Different content produces different ID hash"""
        key_hash = "test_key"
        val1 = "content A"
        val2 = "content B"
        
        id_hash1 = make_memory_id_hash(key_hash, val1)
        id_hash2 = make_memory_id_hash(key_hash, val2)
        
        if id_hash1 != id_hash2:
            self.passed += 1
            self.results.append({"test": "content_hash_change", "status": "PASS"})
        else:
            self.failed += 1
            self.results.append({"test": "content_hash_change", "status": "FAIL", "error": "ID hashes should differ"})
    
    def test_write_retrieve_cycle(self):
        """Test: Write then retrieve returns correct value"""
        memory_store.clear()
        
        key = make_semantic_key("user:test", "pref", "color")
        value = "blue"
        
        memory_store.write("semantic.latent", key, value, reason="test")
        retrieved = memory_store.retrieve("semantic.latent", k=1)
        
        if retrieved and retrieved[0].value == value:
            self.passed += 1
            self.results.append({"test": "write_retrieve_cycle", "status": "PASS"})
        else:
            self.failed += 1
            self.results.append({"test": "write_retrieve_cycle", "status": "FAIL"})
    
    def test_deduplication(self):
        """Test: Duplicate writes detected via ID hash"""
        memory_store.clear()
        
        key = make_semantic_key("project", "model", "name")
        value = "gpt-4o-mini"
        
        entry1 = memory_store.write("semantic.latent", key, value, reason="init")
        entry2 = memory_store.write("semantic.latent", key, value, reason="duplicate")
        
        if entry1.memory_id_hash == entry2.memory_id_hash:
            self.passed += 1
            self.results.append({"test": "deduplication", "status": "PASS", "detail": "Same ID hash detected"})
        else:
            self.failed += 1
            self.results.append({"test": "deduplication", "status": "FAIL"})
    
    def test_update_preservation(self):
        """Test: Update preserves key hash but changes ID hash"""
        memory_store.clear()
        
        key = make_semantic_key("config", "version", "current")
        
        entry1 = memory_store.write("semantic.latent", key, "v1", reason="init")
        entry2 = memory_store.write("semantic.latent", key, "v2", reason="update")
        
        key_hash_match = entry1.memory_key_hash == entry2.memory_key_hash
        id_hash_diff = entry1.memory_id_hash != entry2.memory_id_hash
        
        if key_hash_match and id_hash_diff:
            self.passed += 1
            self.results.append({"test": "update_preservation", "status": "PASS"})
        else:
            self.failed += 1
            self.results.append({"test": "update_preservation", "status": "FAIL"})
    
    def test_type_isolation(self):
        """Test: Different memory types don't collide"""
        memory_store.clear()
        
        memory_store.write("semantic.latent", "key1", "value1")
        memory_store.write("episodic.latent", "key1", "value1")
        memory_store.write("procedural.latent", "key1", "value1")
        
        semantic = memory_store.retrieve("semantic.latent", k=10)
        episodic = memory_store.retrieve("episodic.latent", k=10)
        procedural = memory_store.retrieve("procedural.latent", k=10)
        
        if len(semantic) == 1 and len(episodic) == 1 and len(procedural) == 1:
            self.passed += 1
            self.results.append({"test": "type_isolation", "status": "PASS", "detail": "3 separate entries"})
        else:
            self.failed += 1
            self.results.append({"test": "type_isolation", "status": "FAIL"})
    
    def test_observability_logging(self):
        """Test: All actions logged to audit trail"""
        memory_store.clear()
        
        memory_store.write("semantic.latent", "key", "val", reason="test")
        memory_store.retrieve(k=5)
        
        log = memory_store.get_log()
        
        actions = [entry.get("action") for entry in log]
        if "write" in actions and "retrieve" in actions:
            self.passed += 1
            self.results.append({"test": "observability_logging", "status": "PASS", "log_count": len(log)})
        else:
            self.failed += 1
            self.results.append({"test": "observability_logging", "status": "FAIL"})
    
    def test_retrieve_order(self):
        """Test: Recent entries retrieved first (time order)"""
        memory_store.clear()
        
        memory_store.write("semantic.latent", "key1", "val1")
        memory_store.write("semantic.latent", "key2", "val2")
        memory_store.write("semantic.latent", "key3", "val3")
        
        retrieved = memory_store.retrieve(k=3)
        
        # Should be in reverse chronological order
        if len(retrieved) == 3 and retrieved[0].canonical_key == "key3":
            self.passed += 1
            self.results.append({"test": "retrieve_order", "status": "PASS"})
        else:
            self.failed += 1
            self.results.append({"test": "retrieve_order", "status": "FAIL"})
    
    def test_memory_overflow(self):
        """Test: Handle large number of entries"""
        memory_store.clear()
        
        try:
            for i in range(1000):
                memory_store.write("semantic.latent", f"key{i}", f"value{i}")
            
            retrieved = memory_store.retrieve(k=100)
            
            if len(retrieved) == 100:
                self.passed += 1
                self.results.append({"test": "memory_overflow", "status": "PASS", "entries": 1000})
            else:
                self.failed += 1
                self.results.append({"test": "memory_overflow", "status": "FAIL"})
        except Exception as e:
            self.failed += 1
            self.results.append({"test": "memory_overflow", "status": "FAIL", "error": str(e)})
    
    def test_canonical_key_formats(self):
        """Test: All canonical key makers work correctly"""
        try:
            keys = [
                make_semantic_key("scope", "entity", "attr"),
                make_episodic_key("scope", "event", "2026-01-26", "participants"),
                make_procedural_key("scope", "proc", "v1"),
                make_working_key("thread", "0-10")
            ]
            
            if all(isinstance(k, str) and len(k) > 0 for k in keys):
                self.passed += 1
                self.results.append({"test": "canonical_key_formats", "status": "PASS"})
            else:
                self.failed += 1
                self.results.append({"test": "canonical_key_formats", "status": "FAIL"})
        except Exception as e:
            self.failed += 1
            self.results.append({"test": "canonical_key_formats", "status": "FAIL", "error": str(e)})
    
    def run_all(self):
        """Run all tests"""
        print("=" * 60)
        print("RELIABILITY & ROBUSTNESS TEST SUITE")
        print("=" * 60)
        
        self.test_hash_stability()
        self.test_content_hash_change()
        self.test_write_retrieve_cycle()
        self.test_deduplication()
        self.test_update_preservation()
        self.test_type_isolation()
        self.test_observability_logging()
        self.test_retrieve_order()
        self.test_memory_overflow()
        self.test_canonical_key_formats()
        
        print("\nRESULTS:")
        for result in self.results:
            status_icon = "✓" if result["status"] == "PASS" else "✗"
            print(f"  {status_icon} {result['test']}: {result['status']}")
            if "error" in result:
                print(f"      Error: {result['error']}")
            if "detail" in result:
                print(f"      {result['detail']}")
        
        print(f"\n{'=' * 60}")
        print(f"PASSED: {self.passed} | FAILED: {self.failed}")
        print(f"Success Rate: {self.passed / (self.passed + self.failed) * 100:.1f}%")
        print("=" * 60)
        
        return self.results

if __name__ == "__main__":
    runner = TestRunner()
    runner.run_all()
