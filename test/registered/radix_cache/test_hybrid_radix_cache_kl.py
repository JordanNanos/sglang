import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
    get_input_ids,
)
from sglang.test.test_utils import (
    DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    flush_cache_with_retry,
    popen_launch_server,
)

register_cuda_ci(est_time=900, suite="stage-c-test-2gpu-h200")

MAMBA_CACHE_CHUNK_SIZE = 64
MAMBA_TRACK_INTERVAL = 16
TRACK_BOUNDARY_TRIM_LENS = [16, 32, 48, 64]
DECODE_SUFFIX_LENS = [24, 32, 40, 48]
MULTITURN_SHARED_PREFIX_BASE = 640
MULTITURN_SHARED_PREFIX_STEP = 32
MULTITURN_TURN1_BRANCH_LENS = [160, 192, 224]
MULTITURN_TURN2_BRANCH_LENS = [48, 64, 80]


class MambaReplayKLMixin:
    model = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    kl_div_thres = 0.0025
    max_samples = 50
    prefill_max_new_tokens = 256
    decode_max_new_tokens = 256
    replay_batch_size = 2
    other_args = [
        "--tp-size",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "16",
        "--enable-hybrid-radix-tree",
    ]

    @classmethod
    def setUpClass(cls):
        port = find_available_port(32000)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
        )
        assert flush_cache_with_retry(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)
            time.sleep(2.0)

    @classmethod
    def _acc_thresholds(cls):
        return {cls.model: {"kl_div": cls.kl_div_thres}}

    def flush_cache(self):
        assert flush_cache_with_retry(self.base_url)

    def _assert_cache_hit(self, result: dict, min_cached_tokens: int, label: str):
        cached_tokens = result["meta_info"]["cached_tokens"]
        self.assertGreaterEqual(
            cached_tokens,
            min_cached_tokens,
            f"{label}: expected real prefix-cache hit, got cached_tokens={cached_tokens}",
        )

    def _assert_exact_cached_tokens(
        self, result: dict, expected_cached_tokens: int, label: str
    ):
        cached_tokens = result["meta_info"]["cached_tokens"]
        self.assertEqual(
            cached_tokens,
            expected_cached_tokens,
            f"{label}: expected cached_tokens={expected_cached_tokens}, got {cached_tokens}",
        )

    def _assert_prefill_cached_tokens_near_expected(
        self, result: dict, expected_upper_bound: int, label: str
    ):
        cached_tokens = result["meta_info"]["cached_tokens"]
        expected_lower_bound = max(0, expected_upper_bound - MAMBA_CACHE_CHUNK_SIZE)
        self.assertGreaterEqual(
            cached_tokens,
            expected_lower_bound,
            f"{label}: expected cached_tokens >= {expected_lower_bound}, got {cached_tokens}",
        )
        self.assertLessEqual(
            cached_tokens,
            expected_upper_bound,
            f"{label}: expected cached_tokens <= {expected_upper_bound}, got {cached_tokens}",
        )

    def _assert_cold_replay(self, result: dict, label: str):
        self.assertEqual(
            result["meta_info"]["cached_tokens"],
            0,
            f"{label}: replay path should be cold after flush",
        )

    @staticmethod
    def _expected_prefill_cached_tokens(prefix_len: int) -> int:
        return (prefix_len // MAMBA_CACHE_CHUNK_SIZE) * MAMBA_CACHE_CHUNK_SIZE

    @staticmethod
    def _expected_decode_cached_tokens(history_len: int, output_len: int) -> int:
        if output_len <= 0:
            return history_len
        seq_len = history_len + output_len - 1
        return (seq_len // MAMBA_TRACK_INTERVAL) * MAMBA_TRACK_INTERVAL

    def _get_input_logprobs_batched(
        self, new_input_ids: list[list[int]], output_logprobs: list[list[float]]
    ) -> list[list[float]]:
        input_logprobs = []
        for start in range(0, len(new_input_ids), self.replay_batch_size):
            end = start + self.replay_batch_size
            input_logprobs.extend(
                _get_input_logprobs(
                    self.base_url,
                    new_input_ids[start:end],
                    output_logprobs[start:end],
                )
            )
        return input_logprobs

    def _compare_replay_kl(
        self,
        *,
        label: str,
        replay_input_ids: list[list[int]],
        output_logprobs: list[list[float]],
    ) -> None:
        input_logprobs = self._get_input_logprobs_batched(
            replay_input_ids, output_logprobs
        )
        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            self._acc_thresholds(),
            self.model,
            label,
        )

    @staticmethod
    def _build_replay_item(
        prompt_input_ids: list[int], result: dict
    ) -> tuple[list[int], list[float]]:
        return prompt_input_ids + result["output_ids"], _extract_output_logprobs(result)

    @staticmethod
    def _build_prefill_boundary_inputs(
        input_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        prefix_input_ids = []
        full_input_ids = []
        for i, ids in enumerate(input_ids):
            tail_len = TRACK_BOUNDARY_TRIM_LENS[i % len(TRACK_BOUNDARY_TRIM_LENS)]
            max_full_len = min(len(ids), 1700 + i * 24)
            prefix_len = (
                max(256, max_full_len - tail_len) // MAMBA_CACHE_CHUNK_SIZE
            ) * MAMBA_CACHE_CHUNK_SIZE
            full_len = prefix_len + tail_len
            prefix_input_ids.append(ids[:prefix_len])
            full_input_ids.append(ids[:full_len])
        return prefix_input_ids, full_input_ids

    @staticmethod
    def _build_decode_branch_inputs(
        input_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        first_turn_input_ids = []
        second_turn_suffixes = []
        for i, ids in enumerate(input_ids):
            first_turn_len = min(len(ids), 960 + i * 24)
            branch_suffix_len = DECODE_SUFFIX_LENS[i % len(DECODE_SUFFIX_LENS)]
            branch_suffix = ids[first_turn_len : first_turn_len + branch_suffix_len]
            if len(branch_suffix) < 8:
                branch_suffix = ids[:8]
            first_turn_input_ids.append(ids[:first_turn_len])
            second_turn_suffixes.append(branch_suffix)
        return first_turn_input_ids, second_turn_suffixes

    @staticmethod
    def _build_multiturn_branch_inputs(
        raw_input_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        shared_prefixes = []
        turn1_suffixes = []
        turn2_suffixes = []
        for group_idx in range(6):
            base_ids = raw_input_ids[group_idx * 3]
            prefix_len = min(
                len(base_ids),
                MULTITURN_SHARED_PREFIX_BASE + group_idx * MULTITURN_SHARED_PREFIX_STEP,
            )
            shared_prefix = base_ids[:prefix_len]

            for branch_idx in range(3):
                source_ids = raw_input_ids[group_idx * 3 + branch_idx]
                turn1_start = min(prefix_len, max(128, len(source_ids) // 4))
                turn1_len = MULTITURN_TURN1_BRANCH_LENS[branch_idx] + group_idx * 8
                turn1_suffix = source_ids[turn1_start : turn1_start + turn1_len]
                if len(turn1_suffix) < 48:
                    turn1_suffix = source_ids[:48]

                turn2_start = min(
                    turn1_start + turn1_len,
                    max(turn1_start + 16, len(source_ids) // 2),
                )
                turn2_len = MULTITURN_TURN2_BRANCH_LENS[branch_idx]
                turn2_suffix = source_ids[turn2_start : turn2_start + turn2_len]
                if len(turn2_suffix) < 16:
                    turn2_suffix = source_ids[-16:]

                shared_prefixes.append(shared_prefix)
                turn1_suffixes.append(turn1_suffix)
                turn2_suffixes.append(turn2_suffix)
        return shared_prefixes, turn1_suffixes, turn2_suffixes

    def _run_prefill_replay_case(
        self,
        *,
        label: str,
        prefix_input_ids: list[list[int]],
        full_input_ids: list[list[int]],
        max_new_tokens: int,
        min_cached_tokens: int,
    ):
        self.flush_cache()
        _generate(self.base_url, prefix_input_ids, max_new_tokens=0)

        results = _generate(
            self.base_url,
            full_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(results), len(full_input_ids))

        replay_input_ids = []
        output_logprobs = []
        for i, result in enumerate(results):
            expected_cached_tokens = self._expected_prefill_cached_tokens(
                len(prefix_input_ids[i])
            )
            self._assert_prefill_cached_tokens_near_expected(
                result,
                expected_cached_tokens,
                f"{label}[{i}]",
            )
            self._assert_cache_hit(result, min_cached_tokens, f"{label}[{i}]")
            replay_item, output_logprobs_item = self._build_replay_item(
                full_input_ids[i], result
            )
            replay_input_ids.append(replay_item)
            output_logprobs.append(output_logprobs_item)

        self._compare_replay_kl(
            label=label,
            replay_input_ids=replay_input_ids,
            output_logprobs=output_logprobs,
        )

    def _run_decode_replay_case(
        self,
        *,
        label: str,
        first_turn_input_ids: list[list[int]],
        second_turn_suffixes: list[list[int]],
        max_new_tokens: int,
    ):
        self.flush_cache()
        first_turn_results = _generate(
            self.base_url,
            first_turn_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(first_turn_results), len(first_turn_input_ids))

        second_turn_input_ids = [
            first_turn_input_ids[i]
            + first_turn_results[i]["output_ids"]
            + second_turn_suffixes[i]
            for i in range(len(first_turn_input_ids))
        ]

        second_turn_results = _generate(
            self.base_url,
            second_turn_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(second_turn_results), len(second_turn_input_ids))

        replay_input_ids = []
        output_logprobs = []
        for i, result in enumerate(second_turn_results):
            expected_cached_tokens = self._expected_decode_cached_tokens(
                len(first_turn_input_ids[i]),
                len(first_turn_results[i]["output_ids"]),
            )
            self._assert_exact_cached_tokens(
                result,
                expected_cached_tokens,
                f"{label}[{i}]",
            )
            self._assert_cache_hit(
                result,
                expected_cached_tokens,
                f"{label}[{i}]",
            )
            replay_item, output_logprobs_item = self._build_replay_item(
                second_turn_input_ids[i], result
            )
            replay_input_ids.append(replay_item)
            output_logprobs.append(output_logprobs_item)

        self._compare_replay_kl(
            label=label,
            replay_input_ids=replay_input_ids,
            output_logprobs=output_logprobs,
        )

    def _run_multiturn_branching_replay_case(
        self,
        *,
        label: str,
        shared_prefixes: list[list[int]],
        turn1_suffixes: list[list[int]],
        turn2_suffixes: list[list[int]],
        max_new_tokens: int,
    ):
        self.flush_cache()
        _generate(self.base_url, shared_prefixes, max_new_tokens=0)

        turn1_input_ids = [
            shared_prefixes[i] + turn1_suffixes[i] for i in range(len(shared_prefixes))
        ]
        turn1_results = _generate(
            self.base_url,
            turn1_input_ids,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(turn1_results), len(turn1_input_ids))

        for i, result in enumerate(turn1_results):
            expected_cached_tokens = self._expected_prefill_cached_tokens(
                len(shared_prefixes[i])
            )
            self._assert_exact_cached_tokens(
                result,
                expected_cached_tokens,
                f"{label}[turn1][{i}]",
            )
            self._assert_cache_hit(
                result,
                expected_cached_tokens,
                f"{label}[turn1][{i}]",
            )

        turn2_input_ids = [
            turn1_input_ids[i] + turn1_results[i]["output_ids"] + turn2_suffixes[i]
            for i in range(len(turn1_input_ids))
        ]

        # Interleave branches from different shared trunks so the tree is exercised
        # under realistic multi-session progression instead of single-session replay.
        interleaved_order = []
        branch_stride = 3
        num_groups = len(turn2_input_ids) // branch_stride
        for branch_idx in range(branch_stride):
            for group_idx in range(num_groups):
                interleaved_order.append(group_idx * branch_stride + branch_idx)

        ordered_turn2_inputs = [turn2_input_ids[i] for i in interleaved_order]
        turn2_results = _generate(
            self.base_url,
            ordered_turn2_inputs,
            max_new_tokens=max_new_tokens,
            return_logprob=True,
        )
        self.assertEqual(len(turn2_results), len(ordered_turn2_inputs))

        replay_input_ids = []
        output_logprobs = []
        for ordered_idx, result in enumerate(turn2_results):
            original_idx = interleaved_order[ordered_idx]
            expected_cached_tokens = self._expected_decode_cached_tokens(
                len(turn1_input_ids[original_idx]),
                len(turn1_results[original_idx]["output_ids"]),
            )
            self._assert_exact_cached_tokens(
                result,
                expected_cached_tokens,
                f"{label}[turn2][{original_idx}]",
            )
            self._assert_cache_hit(
                result,
                expected_cached_tokens,
                f"{label}[turn2][{original_idx}]",
            )
            replay_item, output_logprobs_item = self._build_replay_item(
                ordered_turn2_inputs[ordered_idx], result
            )
            replay_input_ids.append(replay_item)
            output_logprobs.append(output_logprobs_item)

        self._compare_replay_kl(
            label=label,
            replay_input_ids=replay_input_ids,
            output_logprobs=output_logprobs,
        )


class TestHybridMambaReplayKL(MambaReplayKLMixin, unittest.TestCase):
    def test_prefill_replay_kl_across_track_boundaries(self):
        input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=2200,
            num_samples=self.max_samples,
        )[: self.max_samples]
        prefix_input_ids, full_input_ids = self._build_prefill_boundary_inputs(
            input_ids
        )

        self._run_prefill_replay_case(
            label="test_prefill_replay_kl_across_track_boundaries",
            prefix_input_ids=prefix_input_ids,
            full_input_ids=full_input_ids,
            max_new_tokens=self.prefill_max_new_tokens,
            min_cached_tokens=256,
        )

    def test_decode_replay_kl_with_branching_suffixes(self):
        input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=1400,
            num_samples=4,
        )[:4]
        first_turn_input_ids, second_turn_suffixes = self._build_decode_branch_inputs(
            input_ids
        )

        self._run_decode_replay_case(
            label="test_decode_replay_kl_with_branching_suffixes",
            first_turn_input_ids=first_turn_input_ids,
            second_turn_suffixes=second_turn_suffixes,
            max_new_tokens=64,
        )

    def test_multiturn_replay_kl_with_interleaved_abc_branches(self):
        raw_input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=2400,
            num_samples=18,
        )[:18]
        shared_prefixes, turn1_suffixes, turn2_suffixes = (
            self._build_multiturn_branch_inputs(raw_input_ids)
        )

        self._run_multiturn_branching_replay_case(
            label="test_multiturn_replay_kl_with_interleaved_abc_branches",
            shared_prefixes=shared_prefixes,
            turn1_suffixes=turn1_suffixes,
            turn2_suffixes=turn2_suffixes,
            max_new_tokens=64,
        )

    def test_prefill_replay_kl_after_competing_branch_pressure(self):
        input_ids = get_input_ids(
            tokenizer_path=self.model,
            max_prompt_tokens=2600,
            num_samples=self.max_samples + 4,
        )[: self.max_samples + 4]

        common_prefix = input_ids[0][:960]
        target_suffix = input_ids[0][960:1680]
        if len(target_suffix) < 256:
            target_suffix = input_ids[1][:256]
        target_full = common_prefix + target_suffix

        pressure_prompts = []
        for i, ids in enumerate(input_ids[1:3]):
            suffix = ids[960 : 960 + 448 + i * 32]
            if len(suffix) < 256:
                suffix = ids[:256]
            pressure_prompts.append(common_prefix + suffix)

        self.flush_cache()
        _generate(self.base_url, [common_prefix], max_new_tokens=0)

        hot_result = _generate(
            self.base_url,
            [target_full],
            max_new_tokens=self.prefill_max_new_tokens,
            return_logprob=True,
        )[0]
        hot_expected_cached_tokens = self._expected_prefill_cached_tokens(
            len(common_prefix)
        )
        self._assert_exact_cached_tokens(
            hot_result,
            hot_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[hot]",
        )
        self._assert_cache_hit(
            hot_result,
            hot_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[hot]",
        )

        for pressure_prompt in pressure_prompts:
            _generate(
                self.base_url,
                [pressure_prompt],
                max_new_tokens=24,
                return_logprob=False,
            )

        revisit_result = _generate(
            self.base_url,
            [target_full],
            max_new_tokens=self.prefill_max_new_tokens,
            return_logprob=True,
        )[0]
        revisit_expected_cached_tokens = self._expected_prefill_cached_tokens(
            len(target_full)
        )
        self._assert_exact_cached_tokens(
            revisit_result,
            revisit_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[revisit]",
        )
        self._assert_cache_hit(
            revisit_result,
            revisit_expected_cached_tokens,
            "test_prefill_replay_kl_after_competing_branch_pressure[revisit]",
        )

        replay_input_ids, output_logprobs = zip(
            self._build_replay_item(target_full, revisit_result)
        )
        self._compare_replay_kl(
            label="test_prefill_replay_kl_after_competing_branch_pressure",
            replay_input_ids=list(replay_input_ids),
            output_logprobs=list(output_logprobs),
        )


if __name__ == "__main__":
    unittest.main()
