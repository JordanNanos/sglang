import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

TEST_MODEL_MATRIX = {
    "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V2-Lite-W8A8": {
        "accuracy": 0.34,
        "latency": 1000,
        "output_throughput": 6,
    },
}


class TestAscendMlaW8A8Int8(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--tp-size",
            2,
            "--disable-radix-cache",
        ]

    def test_a_gsm8k(self):
        os.environ["ASCEND_USE_FIA"] = "true"
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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
