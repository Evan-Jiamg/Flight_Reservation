cd /home/mzjiang/Sep-Simulator && ls sepsim scripts | head -60 && wc -l sepsim/*.py scripts/run_v2.py | tail -30
tar czf /tmp2/mzjiang_usersim/sepsim_src_snapshot.tgz sepsim/*.py scripts/run_v2.py && sha256sum /tmp2/mzjiang_usersim/sepsim_src_snapshot.tgz
