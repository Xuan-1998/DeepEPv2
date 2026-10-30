"""Background EFA stressor: incast all-to-all bursts + uniform all-to-all + concurrent all-reduces
over all ranks, forever (or --duration-s). Each node's local rank 0 prints the node's EFA
retransmission delta every --report-s seconds so the run log shows whether the fabric actually
retransmitted while the kernel under test was running.

torchrun --nproc_per_node=8 --nnodes=N ... bg_a2a.py [--incast-mb 32] [--uniform-mb 8] [--ar-mb 64]
"""
import argparse
import glob
import os
import time

import torch
import torch.distributed as dist


def hw(name):
    tot = 0
    for d in glob.glob('/sys/class/infiniband/rdmap*/ports/1/hw_counters'):
        try:
            with open(os.path.join(d, name)) as f:
                tot += int(f.read())
        except OSError:
            pass
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration-s', type=float, default=1e9)
    ap.add_argument('--report-s', type=float, default=30)
    ap.add_argument('--incast-mb', type=int, default=32, help='MiB every rank sends to one target per burst')
    ap.add_argument('--uniform-mb', type=int, default=8, help='MiB per peer in the uniform all-to-all')
    ap.add_argument('--ar-mb', type=int, default=64, help='MiB per concurrent all-reduce (4 in flight)')
    ap.add_argument('--burst', type=int, default=8, help='incast rounds per burst')
    ap.add_argument('--pause-ms', type=float, default=20, help='idle gap between bursts')
    args = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    dev = torch.device('cuda', local)
    mb = 1024 * 1024 // 2  # bf16 elements per MiB

    incast_in = torch.ones(args.incast_mb * mb, dtype=torch.bfloat16, device=dev)
    incast_out = torch.empty(args.incast_mb * mb * world, dtype=torch.bfloat16, device=dev)
    uni = torch.ones(args.uniform_mb * mb * world, dtype=torch.bfloat16, device=dev)
    uni_out = torch.empty_like(uni)
    ar = [torch.ones(args.ar_mb * mb, dtype=torch.bfloat16, device=dev) for _ in range(4)]

    t0 = time.time()
    last = t0
    tx0, re0, to0 = hw('tx_pkts'), hw('retrans_pkts'), hw('retrans_timeout_events')
    rounds = 0
    while time.time() - t0 < args.duration_s:
        for b in range(args.burst):
            tgt = (rounds + b) % world
            if (rounds + b) % 2 == 0:
                # incast: every rank sends --incast-mb to one target
                in_splits = [incast_in.numel() if p == tgt else 0 for p in range(world)]
                out_splits = [incast_in.numel() if rank == tgt else 0 for _ in range(world)]
                out = incast_out if rank == tgt else incast_out[:0]
                works = [dist.all_to_all_single(out, incast_in, out_splits, in_splits, async_op=True)]
            else:
                # outcast: one rank sends --incast-mb to every peer (saturates its own egress)
                in_splits = [incast_in.numel() if rank == tgt else 0 for _ in range(world)]
                out_splits = [incast_in.numel() if p == tgt else 0 for p in range(world)]
                inp = incast_out if rank == tgt else incast_out[:0]
                works = [dist.all_to_all_single(incast_in, inp, out_splits, in_splits, async_op=True)]
            works += [dist.all_reduce(a, async_op=True) for a in ar]
            for w in works:
                w.wait()
        dist.all_to_all_single(uni_out, uni)
        torch.cuda.synchronize()
        rounds += 1
        time.sleep(args.pause_ms / 1000)
        if local == 0 and time.time() - last >= args.report_s:
            tx1, re1, to1 = hw('tx_pkts'), hw('retrans_pkts'), hw('retrans_timeout_events')
            dt = time.time() - last
            print(f'[bg_a2a] {os.uname().nodename.split(".")[0]} t={time.time() - t0:7.0f}s rounds={rounds} '
                  f'tx_pkts={tx1 - tx0} retrans_pkts={re1 - re0} ({(re1 - re0) * 1e6 / max(tx1 - tx0, 1):.0f}/M) '
                  f'timeout_events={to1 - to0} in last {dt:.0f}s', flush=True)
            tx0, re0, to0, last = tx1, re1, to1, time.time()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
