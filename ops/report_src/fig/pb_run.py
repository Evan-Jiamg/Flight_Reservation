"""Run PaperBanana (skill/run.py) with the OpenRouter key read from ~/.config/openrouter/key into the child's
environment only. The key is never printed or written anywhere. Usage: python pb_run.py <run.py args...>"""
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PB = HERE.parent / "tools" / "paperbanana"
env = dict(os.environ)
if not env.get("OPENROUTER_API_KEY"):
    f = pathlib.Path.home() / ".config" / "openrouter" / "key"
    lines = f.read_text(encoding="utf-8-sig").strip().splitlines() if f.exists() else []
    if not lines or not lines[0].strip():
        sys.exit("no OpenRouter key in ~/.config/openrouter/key")
    env["OPENROUTER_API_KEY"] = lines[0].strip()
env["PYTHONIOENCODING"] = "utf-8"
py = PB / ".venv" / "Scripts" / "python.exe"
sys.exit(subprocess.call([str(py), "skill/run.py"] + sys.argv[1:], cwd=str(PB), env=env))
