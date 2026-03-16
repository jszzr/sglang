from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")

import unittest

from sglang.lang.api import gen
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams


class TestSamplingSeedInDsl(unittest.TestCase):
    def test_gen_accepts_sampling_seed(self):
        expr = gen("answer", sampling_seed=123, temperature=0.7)
        self.assertEqual(expr.sampling_params.sampling_seed, 123)
        self.assertEqual(expr.sampling_params.temperature, 0.7)

    def test_to_srt_kwargs_includes_sampling_seed(self):
        sampling_params = SglSamplingParams(sampling_seed=99)
        self.assertIn("sampling_seed", sampling_params.to_srt_kwargs())
        self.assertEqual(sampling_params.to_srt_kwargs()["sampling_seed"], 99)

    def test_clone_preserves_sampling_seed(self):
        sampling_params = SglSamplingParams(
            max_new_tokens=16,
            sampling_seed=7,
            temperature=0.2,
        )
        cloned = sampling_params.clone()
        self.assertEqual(cloned.sampling_seed, 7)
        self.assertEqual(cloned.max_new_tokens, 16)
        self.assertEqual(cloned.temperature, 0.2)

    def test_resolve_sampling_params_overrides_sampling_seed(self):
        default_sampling = SglSamplingParams(sampling_seed=11, temperature=0.1)
        override_sampling = SglSamplingParams(sampling_seed=42)
        dummy_executor = SimpleNamespace(
            default_sampling_para=default_sampling,
            chat_template=SimpleNamespace(stop_str=[]),
        )

        resolved = StreamExecutor._resolve_sampling_params(
            dummy_executor, override_sampling
        )

        self.assertEqual(resolved.sampling_seed, 42)
        self.assertEqual(dummy_executor.default_sampling_para.sampling_seed, 11)


if __name__ == "__main__":
    unittest.main()
