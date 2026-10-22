from __future__ import annotations

import hashlib
import bisect
from typing import Dict, List, Optional, Tuple, Iterable


def _hash(value: str) -> int:
    """Stable 128-bit hash of a string key.

    We use md5 purely for stable, uniform hash values; the value is not
    used cryptographically, and the hash need not be collision-resistant.
    Using a cryptographic hash as the basis for ring positions avoids the
    clustering that a trivial hash would introduce on real-world input,
    and md5 is fast and deterministic on every standard-library build.
    """
    h = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(h, "big")


class ConsistentRing:
    """A consistent hash ring supporting virtual nodes.

    Virtual nodes ("replicas") are used to smear each physical node's
    keys across the ring, producing a more even key distribution than a
    single slot would.

    The maximum ring position is fixed at 2**128. All input keys are
    positions in a closed interval [0, 2**128 - 1].
    """

    RING_MOD = 1 << 128

    def __init__(self, num_replicas: int = 100) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be at least 1")
        self._num_replicas = num_replicas
        # Sorted lists maintained together; kept parallel.
        self._positions: List[int] = []
        self._slot_to_nodes: List[List[str]] = []
        # Tracks virtual-node count per physical node so removal of a
        # node that happens to share positions with others stays correct.
        self._node_replicas: Dict[str, int] = {}

    # Internal helpers -------------------------------------------------

    def _vnode_key(self, name: str, replica_index: int) -> str:
        return f"{name}#{replica_index}"

    def _insert_slot(self, pos: int, name: str) -> None:
        idx = bisect.bisect_left(self._positions, pos)
        if idx < len(self._positions) and self._positions[idx] == pos:
            # Position collision: multiple physical nodes share a slot.
            # Append rather than overwrite so removal is well-defined.
            self._slot_to_nodes[idx].append(name)
        else:
            self._positions.insert(idx, pos)
            self._slot_to_nodes.insert(idx, [name])

    def _remove_from_slot(self, idx: int, name: str) -> None:
        slot = self._slot_to_nodes[idx]
        slot.remove(name)
        if not slot:
            del self._positions[idx]
            del self._slot_to_nodes[idx]

    def _slot_index_for_key(self, key: str) -> int:
        pos = _hash(key) % self.RING_MOD
        idx = bisect.bisect_left(self._positions, pos)
        if idx == len(self._positions):
            idx = 0
        return idx

    # Public API -------------------------------------------------------

    def add_node(self, name: str) -> None:
        """Add a physical node (and all its virtual nodes) to the ring.

        If the name is already present, the call is a no-op. This avoids
        surprising users with silent key migrations on duplicate calls.
        """
        if name in self._node_replicas:
            return
        self._node_replicas[name] = self._num_replicas
        for i in range(self._num_replicas):
            pos = _hash(self._vnode_key(name, i)) % self.RING_MOD
            self._insert_slot(pos, name)

    def remove_node(self, name: str) -> None:
        """Remove a physical node and all its virtual nodes.

        KeyError is raised if the node is not present, matching the
        behaviour of built-in collection types.
        """
        if name not in self._node_replicas:
            raise KeyError(name)
        del self._node_replicas[name]
        for i in range(self._num_replicas):
            pos = _hash(self._vnode_key(name, i)) % self.RING_MOD
            idx = bisect.bisect_left(self._positions, pos)
            # idx cannot be out of range; the slot must exist.
            self._remove_from_slot(idx, name)

    def get_node(self, key: str) -> Optional[str]:
        """Return the responsible physical node for the given key.

        Returns None if the ring is empty. Walking the key to the next
        position clockwise is the standard consistent-hash lookup.
        """
        if not self._positions:
            return None
        idx = self._slot_index_for_key(key)
        # The first node in a slot's list is the owner; if multiple
        # physical nodes share the slot, they co-own it and we just pick
        # one deterministically.
        return self._slot_to_nodes[idx][0]

    def get_nodes(self, key: str, count: int) -> List[str]:
        """Return up to `count` distinct physical nodes responsible for `key`,
        in clockwise order.

        Used for replication: the first node owns the key, the next ones
        hold replicas. If fewer than `count` distinct physical nodes
        exist, returns as many as are available.

        Asking for `count <= 0` is a programmer error; we raise rather
        than silently return an empty list, which would mask bugs.
        """
        if count <= 0:
            raise ValueError("count must be at least 1")
        if not self._positions:
            return []

        idx = self._slot_index_for_key(key)
        seen: set = set()
        result: List[str] = []
        n_slots = len(self._positions)
        for step in range(n_slots):
            slot_idx = (idx + step) % n_slots
            for node in self._slot_to_nodes[slot_idx]:
                if node not in seen:
                    seen.add(node)
                    result.append(node)
                    if len(result) == count:
                        return result
        return result

    @property
    def num_replicas(self) -> int:
        return self._num_replicas

    @property
    def nodes(self) -> List[str]:
        """Physical nodes currently on the ring, in insertion order."""
        return list(self._node_replicas.keys())

    def __len__(self) -> int:
        """Number of physical nodes on the ring."""
        return len(self._node_replicas)

    def __contains__(self, name: object) -> bool:
        return name in self._node_replicas
