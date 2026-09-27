import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sep-sim"))
import train_planner_rl as T  # noqa: E402
from test_rl_advantages import args, make_splits  # noqa: E402

d = tempfile.mkdtemp()
sp, out = make_splits(d), os.path.join(d, "run")
T.main(args(sp, out, controller="llm", updates=4))
T.main(args(sp, out, "--reselect-seeds", *[str(i) for i in range(8)], controller="llm", updates=4))
r = subprocess.run([sys.executable, os.path.join(HERE, "v11ops", "reselect_boot.py"), out], capture_output=True, text=True)
print(r.stdout[-3000:])
print(r.stderr[-2000:])
print("rc", r.returncode)
