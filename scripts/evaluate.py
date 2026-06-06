#!/usr/bin/env python
"""
本地评估 — Edge F1 / Neighborhood / Taxonomy / Graph Similarity

Usage:
    python scripts/evaluate.py data/train_task_a.json submission.json
"""

import json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import os
from collections import defaultdict


def edge_f1(gold, pred):
    g = {(t[0].strip(), t[1].strip(), t[2].strip()) for t in gold if len(t) == 3}
    p = {(t[0].strip(), t[1].strip(), t[2].strip()) for t in pred if len(t) == 3}
    tp = len(g & p)
    pr = tp / len(p) if p else 0
    rc = tp / len(g) if g else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
    return {"precision": pr, "recall": rc, "f1": f1}


def neighborhood_sim(gold, pred):
    gn, pn = defaultdict(set), defaultdict(set)
    for triples, n in [(gold, gn), (pred, pn)]:
        for t in triples:
            if len(t) == 3:
                n[t[0].strip()].add((t[1].strip(), t[2].strip()))
    nodes = set(gn) | set(pn)
    if not nodes:
        return 1.0
    return sum(len(gn[n] & pn[n]) / len(gn[n] | pn[n]) if (gn[n] | pn[n]) else 0 for n in nodes) / len(nodes)


def taxonomy_sim(gold, pred):
    def ancestors(triples):
        children = defaultdict(set)
        nodes = set()
        for t in triples:
            if len(t) == 3 and t[1].strip() == "is-a":
                children[t[0].strip()].add(t[2].strip())
                nodes |= {t[0].strip(), t[2].strip()}
        anc = defaultdict(set)

        def dfs(n, visited=None):
            if visited is None:
                visited = frozenset()
            if n in visited:
                return set()
            visited |= {n}
            r = set()
            for p in children.get(n, set()):
                r.add(p)
                r |= dfs(p, visited)
            return r

        for n in nodes:
            anc[n] = dfs(n)
        return anc

    ga = ancestors(gold)
    pa = ancestors(pred)
    nodes = set(ga) | set(pa)
    if not nodes:
        return 1.0
    return sum(len(ga[n] & pa[n]) / len(ga[n] | pa[n]) if (ga[n] | pa[n]) else 0 for n in nodes) / len(nodes)


def evaluate(gold_path, pred_path):
    with open(gold_path, encoding='utf-8') as f:
        gold_data = json.load(f)
    with open(pred_path, encoding='utf-8') as f:
        pred_data = json.load(f)

    gmap = {d["id"]: d.get("primitive-ontology-triples", []) for d in gold_data}
    pmap = {d["id"]: d.get("primitive-ontology-triples", []) for d in pred_data}

    te, tn, tt, n = 0, 0, 0, 0
    for sid in gmap:
        if sid not in pmap:
            continue
        te += edge_f1(gmap[sid], pmap[sid])["f1"]
        tn += neighborhood_sim(gmap[sid], pmap[sid])
        tt += taxonomy_sim(gmap[sid], pmap[sid])
        n += 1

    e, ns, ts = te / n, tn / n, tt / n
    gs = (e + ns + ts) / 3
    print("=" * 50)
    print(f"Samples evaluated: {n}")
    print(f"  Edge F1:                {e:.4f}")
    print(f"  Neighborhood Similarity: {ns:.4f}")
    print(f"  Taxonomy Similarity:    {ts:.4f}")
    print(f"  {'─' * 28}")
    print(f"  Graph Similarity Score: {gs:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        evaluate(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python scripts/evaluate.py gold.json pred.json")
