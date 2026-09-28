C=/tmp2/mzjiang_usersim/grpo_planner/code_snapshots/pend_v8; PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
cd $C; ls pytest.ini setup.cfg tox.ini conftest.py pyproject.toml 2>/dev/null; ls test_pend.py test_rl_advantages.py
export PYTHONNOUSERSITE=1
$PY -c "import pytest; print('pytest', pytest.__version__, pytest.__file__)" 2>&1 | tail -2
E1R_TREE=/tmp2/mzjiang_usersim/grpo_planner/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator $PY -m pytest --collect-only -q -p no:cacheprovider test_pend.py test_llm4_controller.py 2>&1 | tail -15
grep -n "=== 2" -A8 /tmp2/mzjiang_usersim/grpo_planner/run_v8_smoke.log | head -12
