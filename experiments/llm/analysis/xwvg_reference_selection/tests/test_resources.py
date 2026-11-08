"""Resource accounting distinguishes reclaimable cache from resident workload."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bca_reference_selection.pipeline import resources


class ResourceChecks(unittest.TestCase):
    def inspect(self, limit, current, stats=None, host_available=500, quota='350000 100000'):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cpu.max').write_text(quota)
            (root / 'memory.max').write_text(str(limit))
            (root / 'memory.current').write_text(str(current))
            if stats is not None:
                (root / 'memory.stat').write_text(stats)
            with patch('bca_reference_selection.pipeline.os.sched_getaffinity', return_value=set(range(8))), \
                 patch('bca_reference_selection.pipeline.psutil.virtual_memory',
                       return_value=SimpleNamespace(available=host_available)):
                return resources(root)

    def test_nearly_full_page_cache_does_not_block_cpu_workers(self):
        result = self.inspect(1000, 980, 'anon 80\nfile 900\ninactive_file 300\nactive_file 600\n')
        self.assertEqual(result['available_cpus'], 3)
        self.assertEqual(result['cgroup_memory']['immediate_headroom_bytes'], 20)
        self.assertEqual(result['available_memory_bytes'], 320)
        # With the pilot's existing 50% margin, a 100-unit worker now fits.
        self.assertGreaterEqual(result['available_memory_bytes'] // 2, 100)

    def test_resident_memory_is_not_treated_as_reclaimable(self):
        result = self.inspect(1000, 980, 'anon 980\ninactive_file 0\n')
        self.assertEqual(result['available_memory_bytes'], 20)
        self.assertLess(result['available_memory_bytes'] // 2, 100)

    def test_missing_stats_falls_back_to_immediate_headroom(self):
        self.assertEqual(self.inspect(1000, 980)['available_memory_bytes'], 20)

    def test_host_limit_and_racing_cache_counter_are_bounded(self):
        result = self.inspect(1000, 980, 'inactive_file 2000\n', host_available=400)
        self.assertEqual(result['cgroup_memory']['estimated_available_bytes'], 1000)
        self.assertEqual(result['available_memory_bytes'], 400)
        self.assertEqual(self.inspect(1000, 1010, 'inactive_file 0\n')['available_memory_bytes'], 0)

    def test_unlimited_cgroup_uses_host_availability(self):
        result = self.inspect('max', 5000, 'inactive_file 50\n', quota='max 100000')
        self.assertEqual(result['available_cpus'], 8)
        self.assertEqual(result['available_memory_bytes'], 500)
        self.assertIsNone(result['cgroup_memory'])


if __name__ == '__main__':
    unittest.main()
