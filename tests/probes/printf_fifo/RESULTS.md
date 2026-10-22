# Device printf FIFO: records that survive per timeout line format

Measured with `fifo_probe_all.cu` (one format per process, 28 blocks x 32 threads x 50 lines = 44,800 lines, the host
counts the lines it prints) in a CUDA 13.0 container. `fifo_probe.cu` is the earlier single-format probe with the
trap test. Record size = FIFO bytes / lines printed.

| line | FIFO 8,650,752 B (default here) | FIFO 1,048,576 B (documented default) | bytes per record |
|---|---|---|---|
| GIN queue state (EFA GDA line, 2 strings and 7 ints) | 16,896 | 2,048 | 512 |
| GIN queue state (backend unavailable line, 1 string and 6 ints) | 33,792 | 4,096 | 256 |
| Gin flush timeout (2 ints) | 33,792 | 4,096 | 256 |
| grid sync timeout (1 string and 5 ints) | 33,792 | 4,096 | 256 |
| NVLink barrier timeout (8 ints) | 33,792 | 4,096 | 256 |
| Gin barrier timeout (6 ints) | 33,792 | 4,096 | 256 |
| notify, GPU reduction timeout (8 ints) | 33,792 | 4,096 | 256 |
| notify, scale-out reduction timeout (1 string and 5 ints) | 33,792 | 4,096 | 256 |
| notify, scale-up reduction timeout (8 ints) | 33,792 | 4,096 | 256 |
| dispatch forwarding timeout (14 ints) | 16,896 | 2,048 | 512 |
| dispatch epilogue drain timeout (6 ints) | 33,792 | 4,096 | 256 |
| combine proxy ring full timeout (8 ints) | 33,792 | 4,096 | 256 |
| combine scale-up wait timeout (9 ints) | 33,792 | 4,096 | 256 |
| combine scale-out wait all timeout (7 ints) | 33,792 | 4,096 | 256 |

Trap test (`fifo_probe 26 28 1 0 trap`): 728 lines then `trap`, 728 lines printed, then `unspecified launch failure`.
44,800 lines into the 1 MiB FIFO: the kernel ends normally, the oldest lines are dropped.

Raw output:

```
fifo_probe fifo_bytes=8650752 records=44800 format=fwd10  printed=16896
fifo_probe fifo_bytes=8650752 records=44800 format=fwd14  printed=16896
fifo_probe fifo_bytes=8650752 records=44800 format=scaleup9  printed=33792
fifo_probe fifo_bytes=8650752 records=44800 format=qstate8  printed=10953
fifo_probe fifo_bytes=8650752 records=44800 format=short3  printed=33792
fifo_probe fifo_bytes=1048576 records=44800 format=fwd10  printed=2048
fifo_probe fifo_bytes=1048576 records=44800 format=fwd14  printed=2048
fifo_probe fifo_bytes=1048576 records=44800 format=scaleup9  printed=4096
fifo_probe fifo_bytes=1048576 records=44800 format=qstate8  printed=2048
fifo_probe fifo_bytes=1048576 records=44800 format=short3  printed=4096
fifo_probe trap printed=728
fifo_bytes=8650752 records=728 format=trap
after trap: unspecified launch failure
fifo_probe_all GIN_queue_state_s fifo_bytes=8650752 records=44800 printed=16896
fifo_probe_all GIN_queue_state_s fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all Gin_flush_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all grid_sync_timeout_s fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all NVLink_barrier_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all Gin_barrier_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all hybrid_notify_GPU_reduction_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all hybrid_notify_scale_out_s_reduction_time fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all hybrid_notify_scale_up_reduction_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all hybrid_dispatch_forwarding_timeout fifo_bytes=8650752 records=44800 printed=16896
fifo_probe_all hybrid_dispatch_epilogue_shadow_drain_ti fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all combine_proxy_ring_full_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all combine_scale_up_wait_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all combine_scale_out_wait_all_timeout fifo_bytes=8650752 records=44800 printed=33792
fifo_probe_all GIN_queue_state_s fifo_bytes=1048576 records=44800 printed=2048
fifo_probe_all GIN_queue_state_s fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all Gin_flush_timeout fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all grid_sync_timeout_s fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all NVLink_barrier_timeout fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all Gin_barrier_timeout fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all hybrid_notify_GPU_reduction_timeout fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all hybrid_notify_scale_out_s_reduction_time fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all hybrid_notify_scale_up_reduction_timeout fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all hybrid_dispatch_forwarding_timeout fifo_bytes=1048576 records=44800 printed=2048
fifo_probe_all hybrid_dispatch_epilogue_shadow_drain_ti fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all combine_proxy_ring_full_timeout fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all combine_scale_up_wait_timeout fifo_bytes=1048576 records=44800 printed=4096
fifo_probe_all combine_scale_out_wait_all_timeout fifo_bytes=1048576 records=44800 printed=4096
```
