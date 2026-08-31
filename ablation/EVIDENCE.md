# PR #9 factorial ablation — separating channel / QP / forward-pair effects

Base for every run is **reordering (remote-first put scheduling)**, which is
always on (it is PR #8 = commit `cdec521`). On top of that base the three
remaining changes in PR #9 are toggled independently:

- **C** = channel unpin (remove the `min(num_channels_per_sm, 4)` cap on the
  unordered path -> 8 channels/SM at 12 SM instead of 4)
- **Q** = default GIN contexts 11 -> 13 (`kDefaultGinContextCnt`)
- **F** = cooperative forward warp pairs

All 8 corners of the (C,Q,F) cube were built as separate git trees (verified
per-tree: gin value, cap occurrence count, coop refs) and run on p5en,
12 SM, 8k tokens, topk 8, 2 nodes, `--test-first-only`. Each config was run
2-3 times on confirmed-healthy node pairs; the table reports the mean.

## Results (plain combine, SO GB/s)

| config | C | Q | F | combine mean | runs | cached dispatch | dispatch |
|---|---|---|---|---|---|---|---|
| base    |   |   |   | 69.4 | 69.8/69.5/68.9 | 76.6 | 81 |
| +C      | Y |   |   | 73.7 | 74.1/74.1/72.9 | 75.8 | 81 |
| +Q      |   | Y |   | 68.3 | 67.1/69.2/68.5 | 81.0 | 81 |
| +F      |   |   | Y | 69.9 | 70.1/69.8       | 76.5 | 81 |
| +C+Q    | Y | Y |   | 73.0 | 73.6/73.2/72.2 | 81.0 | 81 |
| +C+F    | Y |   | Y | 76.1 | 75.8/75.6/77.0 | 75.6 | 81 |
| +Q+F    |   | Y | Y | 70.0 | 69.8/69.1/71.1 | 81.0 | 81 |
| +C+Q+F (PR) | Y | Y | Y | 77.3 | 77.2/77.4 | 81.0 | 81 |

Runs at 8k are the bistable-stagger regime, so combine carries +/-1-2 GB/s
run-to-run noise; that is why each cell is averaged and why single-run
inversions (rep1 showed 111>101 by 1.4, rep3 showed them equal) should
not be over-read.

## Decomposition (delta combine vs base 69.4)

Separate effects:
- **C alone: +4.3** - the dominant combine lever.
- **Q alone: -1.1** - QP 13 slightly hurts combine.
- **F alone: +0.5** - coop pairs do almost nothing by themselves.

Pairwise:
- C+Q: +3.6  (Q drags C down ~0.7)
- **C+F: +6.7**  (best pair)
- Q+F: +0.6

All three:
- C+Q+F: +7.9

## Findings

1. **C is the main combine lever and F depends on C.** F alone is +0.5, but
   C+F is +6.7 - larger than C(+4.3)+F(+0.5)=+4.8, a +1.9 synergy. Coop
   pairs only pay off once channel-unpin has created the extra forward warps
   to pair. Matches the earlier "coop without channels ~= baseline" result.

2. **Q is a cached-dispatch lever, not a combine lever.** Cached dispatch is a
   clean binary on Q: 75.6-76.6 with Q off, 81.0 with Q on (+~4.4),
   regardless of C/F. The cost is ~1 GB/s of combine. dispatch and expanded
   dispatch stay at 81 in every config - none of the three changes touches
   them.

3. **Full-stack combine (77.3) ~= C+F (76.1) within noise.** Q adds its
   cached-dispatch win on top without materially changing combine. So the PR's
   combine gain is essentially "C then F"; Q is carried for cached dispatch.

## Reproduce

Trees: branches `abl-000-base` ... `abl-111-CQF` (abl-000-base = cdec521;
abl-111-CQF == 3c737dc). Runner:
NODES=2 GIN_TYPE=5 KERNEL=unordered TEST_ARGS="--test-first-only
--num-tokens=8192 --num-topk=8 --num-sms=12" ./ep <tree>. Raw logs for one
representative run of each config are in the sibling directories here.
