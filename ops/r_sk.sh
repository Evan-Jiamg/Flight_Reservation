for e in blackwell-kto-test consistent-test consistent; do echo -n "$e: "; PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/$e/bin/python -c "import sklearn, numpy; print(sklearn.__version__, numpy.__version__)" 2>&1 | tail -1; done
ls /home/mzjiang/miniconda3/envs/
tail -2 /tmp2/mzjiang_usersim/grpo_planner/prism_probe_v1.log | cut -c1-120
