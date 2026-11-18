"""Unit tests for the pure helpers. Run: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import llama_gauges as lg  # noqa: E402
import lmstudio_flags as lf  # noqa: E402

METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 10
llamacpp:tokens_predicted_total 300
llamacpp:tokens_predicted_seconds_total 5.21504
llamacpp:spec_decode_num_draft_tokens_total 423
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 84
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 44
llamacpp:requests_deferred 0
"""


class ParseMetrics(unittest.TestCase):
    def test_counters_and_labels(self):
        m = lg.parse_metrics(METRICS)
        self.assertEqual(m["prompt_tokens_total"], 10.0)
        self.assertAlmostEqual(m["tokens_predicted_seconds_total"], 5.21504)
        self.assertEqual(m["spec_decode_num_accepted_tokens_per_pos_total"], {0: 84.0, 1: 44.0})
        self.assertNotIn("# HELP", "".join(m))


class Cmdline(unittest.TestCase):
    def test_flags_and_values(self):
        cmd = ["llama-server.exe", "--model", "C:/m/x.gguf", "--port", "56542", "--metrics",
               "--n-predict", "-1", "--api-key-file", "k.txt", "--no-webui"]
        o = lg.opts_from_cmdline(cmd)
        self.assertEqual(o["--port"], "56542")
        self.assertIsNone(o["--metrics"])
        self.assertEqual(o["--n-predict"], "-1")
        self.assertEqual(o["--api-key-file"], "k.txt")
        self.assertIn("--no-webui", o)


class Format(unittest.TestCase):
    def test_fk(self):
        self.assertEqual(lg.fk(32000), "32k")
        self.assertEqual(lg.fk(32768), "32K")
        self.assertEqual(lg.fk(409), "409")
        self.assertEqual(lg.fk(1_500_000), "1.5M")

    def test_bar_width(self):
        for frac in (0, 0.33, 0.5, 1, 2, float("nan")):
            self.assertEqual(len(lg.bar(frac, 20, "white").plain), 20)

    def test_spark_width(self):
        self.assertEqual(len(lg.spark([1, 5, 10], 10, 12, "white").plain), 12)


class ServerProgress(unittest.TestCase):
    def make(self, slots):
        s = lg.Server(dict(pid=None, host="127.0.0.1", port=1, model="m", ctx="32000", parallel="4",
                           key_file=None, metrics_flag=True, spec=None, ncpumoe=None, runtime=""), None)
        s.slots = slots
        return s

    def test_states(self):
        idle = self.make([{"id": 0, "n_ctx": 32000, "is_processing": False}])
        self.assertEqual(idle.state(), "idle")
        self.assertIsNone(idle.progress())
        reading = self.make([{"id": 0, "n_ctx": 32000, "is_processing": True, "id_task": 1,
                              "n_prompt_tokens": 100, "n_prompt_tokens_processed": 40,
                              "n_prompt_tokens_cache": 10, "next_token": [{"n_decoded": 0, "n_remain": -1}]}])
        self.assertEqual(reading.state(), "reading")
        self.assertAlmostEqual(reading.progress()[0], 0.5)
        gen = self.make([{"id": 0, "n_ctx": 32000, "is_processing": True, "id_task": 1, "n_prompt_tokens": 100,
                          "next_token": [{"n_decoded": 30, "n_remain": 90}]}])
        self.assertEqual(gen.state(), "generating")
        self.assertAlmostEqual(gen.progress()[0], 0.25)


class FlagsMerge(unittest.TestCase):
    FLAGS = [{"key": "--metrics", "value": ""}, {"key": "--api-key-file", "value": "/k.txt"}]

    def test_add_keeps_existing_and_is_idempotent(self):
        fields = [{"key": lf.FIELD, "value": {"enabled": True, "overrideParameters": [
            {"key": "--no-repack", "value": ""}, {"key": " --api-key-file", "value": "/old"}]}}]
        self.assertTrue(lf.add_flags(fields, self.FLAGS))
        params = fields[0]["value"]["overrideParameters"]
        self.assertEqual([p["key"] for p in params], ["--no-repack", "--api-key-file", "--metrics"])
        self.assertEqual(params[1]["value"], "/k.txt")
        self.assertEqual(lf.add_flags(fields, self.FLAGS), "")

    def test_add_to_empty_and_remove(self):
        fields = []
        self.assertEqual(lf.add_flags(fields, self.FLAGS), "added")
        self.assertTrue(lf.remove_flags(fields, self.FLAGS))
        self.assertEqual(fields, [])
        self.assertEqual(lf.remove_flags(fields, self.FLAGS), "")

    def test_hub_load_detection(self):
        lines = ["model: a/b", "config:", '  load: {"fields": []}', "  operation:", "other: 1"]
        self.assertEqual(lf.read_hub_load(lines), (1, 2))
        self.assertEqual(lf.read_hub_load(["model: a/b", "config:", "  operation:"]), (1, None))
        self.assertEqual(lf.read_hub_load(["model: a/b"]), (None, None))


if __name__ == "__main__":
    unittest.main()
