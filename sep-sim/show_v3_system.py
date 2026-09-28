"""Print the v3 system prompt for manual review."""
import os
import sys

os.environ["SEPSIM_ACT_PRIOR"] = "nostopclobber"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit_src"))
import planner_prompt_v3 as V3  # noqa: E402

print(V3.system_prompt_v3())
