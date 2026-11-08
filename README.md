# tiny_consistent_ring

A small consistent-hash ring with virtual nodes, for distributing keys across a changing set of nodes with minimal disruption.

## Usage

```python
from tiny_consistent_ring import ConsistentRing

ring = ConsistentRing(num_replicas=100)
ring.add_node("alpha")
ring.add_node("beta")
ring.add_node("gamma")

# Single owner
owner = ring.get_node("user:42")

# Replication: first three distinct physical nodes in clockwise order
replicas = ring.get_nodes("user:42", 3)

# Membership
len(ring)            # -> 3
"alpha" in ring      # -> True
ring.nodes           # -> ["alpha", "beta", "gamma"]

ring.remove_node("alpha")
```

## Why this exists

When you shard data or route requests across N servers, hashing a key to `hash(key) % N` collapses the moment N changes: almost every key remaps to a different server, and you rebuild your entire world. Consistent hashing fixes this by placing both nodes and keys onto a ring; adding or removing a node only moves the keys that fall in the small arc the new node owns. Virtual nodes smear each physical node across many ring positions, so the load is even instead of clustered.

This library implements exactly that, with no third-party dependencies. The trade-off: it is in-memory only. There is no persistence, no network transport, no cluster membership protocol. If you need distributed consensus, look elsewhere; if you need a deterministic mapping function you can rebuild at any time, this is it.

## The awkward edge

`get_nodes(key, count)` returns distinct physical nodes in clockwise order. If you ask for more replicas than physical nodes exist, you get back only as many as are available — no node appears twice. This is almost certainly what you want for replication, but it means the returned list can be shorter than `count`, and your replication logic must handle that rather than assuming the length.

## Implementation choices

- The ring space is 2**128 positions wide; key and node positions are derived from md5 of the key string. md5 is used only for its uniform, deterministic distribution, not for any cryptographic property.
- `add_node` called twice with the same name is a silent no-op, not an error. This keeps idempotent provisioning code simple.
- `remove_node` for a missing name raises `KeyError`, matching built-in collection semantics rather than swallowing the mistake.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

## Contributing

Issues and pull requests are welcome. Please keep the dependency list empty —
that constraint is the point of the project, not an oversight.

