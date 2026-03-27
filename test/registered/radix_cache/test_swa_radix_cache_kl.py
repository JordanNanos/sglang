import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

MODEL = "openai/gpt-oss-20b"

register_cuda_ci(est_time=100, suite="stage-b-test-1-gpu-large")


class TestSWARadixCacheKL(KLDivergenceMixin, DefaultServerBase):
    model = MODEL
    kl_div_thres = 0.002
    kl_div_decode_max_new_tokens = 2048
    other_args = [
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.70",
        "--disable-piecewise-cuda-graph",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
