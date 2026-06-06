"""
图后处理 — 去重、环检测、类型推断
"""

from typing import List, Dict, Set
from collections import defaultdict


def has_cycle(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """DFS 检测 is-a 层次中的环."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    cycles, path = [], []

    def dfs(node):
        color[node] = GRAY
        path.append(node)
        for neighbor in graph.get(node, set()):
            if color[neighbor] == GRAY:
                start = path.index(neighbor)
                cycles.append(path[start:])
            elif color[neighbor] == WHITE:
                dfs(neighbor)
        path.pop()
        color[node] = BLACK

    for node in list(graph.keys()):
        if color[node] == WHITE:
            dfs(node)
    return cycles


def postprocess_triples(triples: List[List[str]]) -> List[List[str]]:
    """
    清理 + 去重 + 环修复 + 重建.

    Args:
        triples: [(subject, predicate, object), ...]

    Returns:
        cleaned list of triples
    """
    if not triples:
        return []

    # ── 1. 清理 ──
    cleaned = []
    for t in triples:
        if len(t) != 3:
            continue
        s, p, o = t[0].strip(), t[1].strip(), t[2].strip()
        if not s or not p or not o or s.lower() == o.lower():
            continue
        cleaned.append([s, p, o])

    # ── 2. 去重 ──
    seen = set()
    deduped = []
    for t in cleaned:
        key = (t[0].lower(), t[1].lower(), t[2].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    # ── 3. 环检测与修复 ──
    is_a_tuples = [(s, o) for s, p, o in deduped if p == "is-a"]
    instance_tuples = [(s, o) for s, p, o in deduped if p == "instance-of"]
    other = [t for t in deduped if t[1] not in ("is-a", "instance-of")]

    type_graph = defaultdict(set)
    for s, o in is_a_tuples:
        type_graph[s.lower()].add(o.lower())

    cycles = has_cycle(type_graph)
    if cycles:
        remove = {(c[-1], c[0]) for c in cycles}
        is_a_tuples = [(s, o) for s, o in is_a_tuples
                       if (s.lower(), o.lower()) not in remove]

    # ── 4. 重建 ──
    result = []
    result.extend([[s, "instance-of", o] for s, o in instance_tuples])
    result.extend([[s, "is-a", o] for s, o in is_a_tuples])
    result.extend(other)
    return result
