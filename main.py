"""
LLMs4OL 2026 — 统一入口

Usage:
    python main.py train    --gpu auto --epochs 3
    python main.py predict  --model_path output/.../final_model
    python main.py evaluate data/train_task_a.json submission.json
    python main.py check    # GPU 诊断
"""
import subprocess, sys, os

SCRIPTS = os.path.join(os.path.dirname(__file__), "scripts")
CMDS = {
    "train":    "train.py",
    "predict":  "predict.py",
    "evaluate": "evaluate.py",
    "check":    "check_gpu.py",
    "env":      "check_env.py",
    "github":   "init_github.sh",
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("help", "-h", "--help"):
        print(__doc__)
        print("Available commands:", ", ".join(CMDS.keys()))
        sys.exit(0)

    cmd = args[0]
    if cmd not in CMDS:
        print(f"Unknown command: {cmd}")
        print("Available:", ", ".join(CMDS.keys()))
        sys.exit(1)

    script = os.path.join(SCRIPTS, CMDS[cmd])
    if script.endswith(".sh"):
        sys.exit(subprocess.run(["bash", script] + args[1:]).returncode)
    else:
        sys.exit(subprocess.run([sys.executable, script] + args[1:]).returncode)
