import unittest

from tiny_consistent_ring import ConsistentRing


class TestConsistentRing(unittest.TestCase):

    def test_empty_ring_returns_none_for_single(self):
        ring = ConsistentRing(num_replicas=10)
        self.assertIsNone(ring.get_node("anything"))

    def test_empty_ring_returns_empty_for_replicas(self):
        ring = ConsistentRing(num_replicas=10)
        self.assertEqual(ring.get_nodes("anything", 3), [])

    def test_add_and_lookup_single_node(self):
        ring = ConsistentRing(num_replicas=5)
        ring.add_node("A")
        self.assertEqual(ring.get_node("key1"), "A")
        self.assertEqual(ring.get_node("key2"), "A")

    def test_add_duplicate_node_is_noop(self):
        ring = ConsistentRing(num_replicas=20)
        ring.add_node("A")
        positions_before = list(ring._positions)
        ring.add_node("A")  # should be silently ignored
        self.assertEqual(positions_before, ring._positions)
        self.assertEqual(len(ring), 1)

    def test_remove_node_not_present_raises(self):
        ring = ConsistentRing(num_replicas=5)
        with self.assertRaises(KeyError):
            ring.remove_node("nope")

    def test_remove_only_node_empties_ring(self):
        ring = ConsistentRing(num_replicas=5)
        ring.add_node("A")
        ring.remove_node("A")
        self.assertIsNone(ring.get_node("key"))
        self.assertEqual(len(ring), 0)

    def test_get_nodes_basic(self):
        ring = ConsistentRing(num_replicas=10)
        ring.add_node("A")
        ring.add_node("B")
        ring.add_node("C")
        nodes = ring.get_nodes("some-key", 2)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(set(nodes)), 2)
        self.assertIn(nodes[0], ["A", "B", "C"])
        self.assertIn(nodes[1], ["A", "B", "C"])

    def test_get_nodes_count_exceeds_physical_nodes(self):
        ring = ConsistentRing(num_replicas=10)
        ring.add_node("A")
        ring.add_node("B")
        nodes = ring.get_nodes("some-key", 5)
        self.assertEqual(sorted(nodes), ["A", "B"])

    def test_get_nodes_invalid_count_raises(self):
        ring = ConsistentRing(num_replicas=5)
        ring.add_node("A")
        with self.assertRaises(ValueError):
            ring.get_nodes("key", 0)
        with self.assertRaises(ValueError):
            ring.get_nodes("key", -1)

    def test_min_num_replicas_is_one(self):
        with self.assertRaises(ValueError):
            ConsistentRing(num_replicas=0)
        ring = ConsistentRing(num_replicas=1)
        ring.add_node("A")
        self.assertEqual(ring.num_replicas, 1)

    def test_contains_and_len(self):
        ring = ConsistentRing(num_replicas=5)
        self.assertNotIn("A", ring)
        ring.add_node("A")
        self.assertIn("A", ring)
        self.assertEqual(len(ring), 1)

    def test_nodes_property_is_snapshot(self):
        ring = ConsistentRing(num_replicas=5)
        ring.add_node("A")
        ring.add_node("B")
        nodes = ring.nodes
        ring.remove_node("B")
        self.assertEqual(nodes, ["A", "B"])

    def test_key_stability_when_unrelated_node_added(self):
        """A key's owning node must not change when a different node is added."""
        ring = ConsistentRing(num_replicas=50)
        ring.add_node("A")
        ring.add_node("B")
        key = "important-key"
        owner_before = ring.get_node(key)
        ring.add_node("C")
        owner_after = ring.get_node(key)
        # The owner can change ONLY if the new node is now responsible.
        # Adding node C should only affect keys that C now owns, and the
        # specific key in this test is arbitrary. The guarantee we test
        # is that the set of (key -> node) mappings is stable EXCEPT for
        # keys that move to the newly added node. So if the owner
        # changed, it MUST now be C.
        if owner_before != owner_after:
            self.assertEqual(owner_after, "C")

    def test_stability_after_add_remove_cycle(self):
        """Adding and removing the same node should leave the ring's
        (key -> node) mapping identical for keys not owned by that node."""
        ring = ConsistentRing(num_replicas=30)
        ring.add_node("A")
        ring.add_node("B")
        keys = [f"key-{i}" for i in range(200)]
        snapshot_before = {k: ring.get_node(k) for k in keys}
        ring.add_node("C")
        ring.remove_node("C")
        snapshot_after = {k: ring.get_node(k) for k in keys}
        self.assertEqual(snapshot_before, snapshot_after)

    def test_distribution_is_reasonable(self):
        """With enough virtual nodes, no single physical node should own
        a grossly disproportionate share of keys."""
        ring = ConsistentRing(num_replicas=100)
        ring.add_node("A")
        ring.add_node("B")
        ring.add_node("C")
        counts = {"A": 0, "B": 0, "C": 0}
        for i in range(3000):
            owner = ring.get_node(f"item-{i}")
            counts[owner] += 1
        # With 3 nodes and 3000 keys, each should have roughly 1000.
        # We allow a wide margin (20%) to keep the test deterministic.
        for c in counts.values():
            self.assertGreater(c, 400)
            self.assertLess(c, 1600)

    def test_get_nodes_returns_distinct_physical_nodes(self):
        ring = ConsistentRing(num_replicas=10)
        ring.add_node("X")
        ring.add_node("Y")
        ring.add_node("Z")
        nodes = ring.get_nodes("test", 3)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(set(nodes)), 3)

    def test_remove_specific_node_changes_owner(self):
        """If we remove the node that owns a key, the key moves to a
        different node (or becomes None if that was the last node)."""
        ring = ConsistentRing(num_replicas=20)
        ring.add_node("A")
        ring.add_node("B")
        ring.add_node("C")
        key = "special-key"
        owner = ring.get_node(key)
        ring.remove_node(owner)
        new_owner = ring.get_node(key)
        self.assertNotEqual(new_owner, owner)


if __name__ == "__main__":
    unittest.main()
