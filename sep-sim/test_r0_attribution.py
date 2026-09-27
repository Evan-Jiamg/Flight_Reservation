import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_pipeline as V  # noqa: E402


def ep(total, own, tok="p1"):
    r = {"r0_len_truncated_total": total, "episode_counters": {"r0_len_truncated": own}}
    if tok is not None:
        r["process_token"] = tok
    return r


def fails(rows):
    rep = V.Report()
    V.check_r0_attribution(rows, rep)
    return rep.checks["trunc.r0_attributed"]["fail"]


class R0Attribution(unittest.TestCase):
    def test_attributed_truncation_passes(self):
        # the v11 u8 case: one truncation, charged to its episode; later rows keep the running total 1
        self.assertEqual(fails([ep(0, 0), ep(1, 1), ep(1, 0), ep(1, 0)]), 0)

    def test_unattributed_truncation_fails(self):
        self.assertEqual(fails([ep(0, 0), ep(1, 0), ep(1, 0)]), 1)

    def test_processes_counted_separately(self):
        # a resume restarts the total; each process must balance on its own
        self.assertEqual(fails([ep(1, 1, "a"), ep(1, 0, "b")]), 1)
        self.assertEqual(fails([ep(1, 1, "a"), ep(2, 2, "b")]), 0)

    def test_legacy_rows_pooled(self):
        self.assertEqual(fails([ep(1, 1, None), ep(1, 0, None)]), 0)
        self.assertEqual(fails([ep(1, 0, None)]), 1)

    def test_clean_flag_still_required(self):
        rep = V.Report()
        row = {"emitted_user_turns": 3, "clean": True, "episode_counters": {"r0_len_truncated": 1}, "trace": []}
        V.check_pend([row], rep)
        self.assertGreater(rep.checks["pend.clean_flag"]["fail"], 0)


if __name__ == "__main__":
    unittest.main()
