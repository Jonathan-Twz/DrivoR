"""CPU-only tests; no Slurm jobs or training processes are started."""
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from check_allocation import duration, inspect, main

JOB = ('JobId=123 UserId=alice(1000) Account=project JobState=RUNNING '
       'NumNodes=1 NumCPUs=16 NodeList=node1 AllocTRES=cpu=16,gres/gpu=4 '
       'TimeLimit=7-00:00:00 RunTime=2-00:00:00')


class AllocationGuardTest(unittest.TestCase):
    def check(self, job=JOB, steps='123.interactive\n123.extern', **kwargs):
        args = dict(owner='alice', num_gpus=4, requested_cpus=0, expected_hours=40,
                    reserve_minutes=5, expected_account='project')
        args.update(kwargs)
        return inspect(job, steps, **args)

    def test_resources_and_remaining_time(self):
        self.assertEqual(self.check(), (4, 7195, 'project', 'node1'))

    def test_durations(self):
        self.assertEqual(duration('01:02:03'), 3723)
        self.assertEqual(duration('02:03'), 123)

    def test_rejected_allocations(self):
        cases = [dict(owner='bob'), dict(expected_account='other'), dict(num_gpus=8),
                 dict(requested_cpus=8), dict(expected_hours=121), dict(expected_hours=float('nan')),
                 dict(job=JOB.replace('RUNNING','PENDING')), dict(job=JOB.replace('NumNodes=1','NumNodes=2')),
                 dict(steps='123.34\n123.interactive')]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.check(**case)

    def test_own_interactive_step(self):
        self.check(steps='123.0\n123.extern', current_step='123.0')

    def test_read_only_cli(self):
        output = StringIO()
        with patch('sys.argv', ['check_allocation.py','--job','123','--gpus','4','--expected-hours','40']), \
             patch('getpass.getuser', return_value='alice'), \
             patch('subprocess.check_output', side_effect=[JOB, '123.interactive\n123.extern']) as query, \
             redirect_stdout(output), redirect_stderr(StringIO()):
            main()
        self.assertEqual(output.getvalue(), '4 7195\n')
        self.assertEqual(query.call_args_list[0].args[0], ['scontrol','show','job','123','-o'])
        self.assertEqual(query.call_args_list[1].args[0], ['squeue','--steps','-j','123','-h','-o','%i'])


if __name__ == '__main__':
    unittest.main()
