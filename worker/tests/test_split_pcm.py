import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr_worker import split_pcm  # noqa: E402

RATE = 16000


def noisy(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, int(RATE * seconds)).astype(np.float32)


class SplitPcmTest(unittest.TestCase):
    def test_short_recording_is_not_split(self):
        samples = noisy(20.0)
        chunks = split_pcm(samples, RATE)
        self.assertEqual(len(chunks), 1)
        self.assertIs(chunks[0], samples)

    def test_long_recording_keeps_every_sample_in_order(self):
        samples = noisy(60.0)
        chunks = split_pcm(samples, RATE)
        self.assertGreater(len(chunks), 1)
        np.testing.assert_array_equal(np.concatenate(chunks), samples)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), int(RATE * 22.0))

    def test_boundary_prefers_silence(self):
        samples = noisy(40.0)
        quiet_start, quiet_end = int(RATE * 19.0), int(RATE * 19.5)
        samples[quiet_start:quiet_end] = 0.0
        chunks = split_pcm(samples, RATE)
        first_boundary = len(chunks[0])
        self.assertGreaterEqual(first_boundary, quiet_start)
        self.assertLessEqual(first_boundary, quiet_end)

    def test_first_boundary_not_before_minimum(self):
        samples = noisy(40.0)
        samples[: int(RATE * 5.0)] = 0.0  # тишина в начале не должна стать границей
        chunks = split_pcm(samples, RATE)
        self.assertGreaterEqual(len(chunks[0]), int(RATE * 8.0))


if __name__ == "__main__":
    unittest.main()
