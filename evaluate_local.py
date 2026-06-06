"""
LLMs4OL 2026 — 向后兼容入口 → 实际逻辑在 scripts/evaluate.py
"""
import subprocess, sys, os
_script = os.path.join(os.path.dirname(__file__), "scripts", "evaluate.py")
sys.exit(subprocess.run([sys.executable, _script] + sys.argv[1:]).returncode)
