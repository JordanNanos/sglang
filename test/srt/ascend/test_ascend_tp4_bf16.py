import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

TEST_MODEL_MATRIX = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "accuracy": 0.90,
        "latency": 180,
        "output_throughput": 20,
    },
}


class TestAscendTp4Bf16(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.7,
            "--max-running-requests",
            32,
            "--attention-backend",
            "ascend",
            "--disable-radix-cache",
            "--cuda-graph-max-bs",
            32,
            "--tp-size",
            4,
        ]

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1800,
                    other_args=[
                        *self.common_args,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=self.model,
                        eval_name="gsm8k",
                        num_examples=1319,
                        num_threads=128,
                    )
                    metrics = run_eval_few_shot_gsm8k(args)
                    self.assertGreaterEqual(
                        metrics["score"],
                        TEST_MODEL_MATRIX[model]["accuracy"],
                    )
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
