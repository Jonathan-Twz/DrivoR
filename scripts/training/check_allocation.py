"""Read-only guard for training steps inside an existing, single-node allocation."""
import argparse
import getpass
import math
import os
import re
import subprocess
import sys


def duration(value):
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    parts = [int(x) for x in clock.split(':')]
    if len(parts) == 2:
        parts.insert(0, 0)
    if len(parts) != 3:
        raise ValueError(f'Unsupported Slurm duration: {value}')
    return int(days)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]


def inspect(job, steps, owner, num_gpus, requested_cpus, expected_hours, reserve_minutes,
            expected_account='', current_step=''):
    fields = dict(re.findall(r'(\S+?)=(\S+)', job))
    if fields.get('JobState') != 'RUNNING':
        raise ValueError('Allocation is not RUNNING')
    if fields.get('UserId', '').split('(')[0] != owner:
        raise ValueError('Allocation belongs to a different user')
    if expected_account and fields.get('Account') != expected_account:
        raise ValueError(f"Account mismatch: allocated {fields.get('Account')}, expected {expected_account}")
    if int(fields.get('NumNodes', '0')) != 1:
        raise ValueError('Only single-node training allocations are supported')
    match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', fields.get('AllocTRES', ''))
    if not match or int(match[1]) < num_gpus:
        raise ValueError(f'Allocation does not contain {num_gpus} GPUs')
    total_cpus = int(fields['NumCPUs'])
    cpus = requested_cpus or total_cpus // num_gpus
    if cpus < 1 or cpus*num_gpus > total_cpus:
        raise ValueError(f'Insufficient CPUs: {num_gpus} ranks x {cpus} CPUs, allocated {total_cpus}')
    if not math.isfinite(expected_hours) or expected_hours <= 0 or reserve_minutes < 1:
        raise ValueError('Supply a positive EXPECTED_RUN_HOURS and positive time reserve')
    seconds = duration(fields['TimeLimit']) - duration(fields['RunTime']) - reserve_minutes*60
    if seconds < expected_hours*3600:
        raise ValueError(f'Insufficient time: {seconds/3600:.2f} usable hours, expected {expected_hours:g}')
    busy = [s.strip() for s in steps.splitlines()
            if s.strip() and s.strip().rsplit('.', 1)[-1] not in ('interactive', 'extern')
            and s.strip() != current_step]
    if busy:
        raise ValueError('Allocation already has active workload steps: '+', '.join(busy))
    return cpus, seconds//60, fields['Account'], fields['NodeList']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--cpus', type=int, default=0)
    parser.add_argument('--expected-hours', type=float, required=True)
    parser.add_argument('--reserve-minutes', type=int, default=5)
    parser.add_argument('--account', default='')
    args = parser.parse_args()
    if not args.job.isdecimal() or args.gpus < 1 or args.cpus < 0:
        parser.error('Use a numeric job ID, positive GPU count and nonnegative CPU count')
    job = subprocess.check_output(['scontrol','show','job',args.job,'-o'], text=True)
    steps = subprocess.check_output(['squeue','--steps','-j',args.job,'-h','-o','%i'], text=True)
    current = ''
    if os.environ.get('SLURM_JOB_ID') == args.job and os.environ.get('SLURM_STEP_ID'):
        current = f"{args.job}.{os.environ['SLURM_STEP_ID']}"
    cpus, minutes, account, node = inspect(job, steps, getpass.getuser(), args.gpus,
                                         args.cpus, args.expected_hours, args.reserve_minutes,
                                         args.account, current)
    print(f'Allocation {args.job}: account={account}, node={node}, {args.gpus} GPUs, '
          f'{cpus} CPUs/rank, {minutes/60:.2f} usable hours', file=sys.stderr)
    print(cpus, minutes)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f'Allocation check failed: {error}')
