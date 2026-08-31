import subprocess, os
def run(c): 
    r = subprocess.run(c, shell=True, cwd='/home/xuanj/deepep-kernel-select', capture_output=True, text=True)
    if r.returncode: print("ERR", c, r.stderr[:300])
    return r.stdout

GIN='deep_ep/include/deep_ep/common/gin_resource_alloc.cuh'
BUF='csrc/elastic/buffer.hpp'
CAP_ANCHOR='''                           "dispatch TMA pool exceeds the shared-memory budget");
'''
CAP_LINES=CAP_ANCHOR+'''            if (not prefer_overlap_with_compute)
                num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
'''

def set_gin(val):
    p=GIN; s=open(p).read()
    import re
    s=re.sub(r'(static constexpr int kDefaultGinContextCnt\s+= )\d+;', rf'\g<1>{val};', s)
    open(p,'w').write(s)

def remove_unordered_cap():
    # remove ONLY the cap that follows the dispatch TMA assert (unordered path)
    p=BUF; s=open(p).read()
    if CAP_LINES not in s: return
    s=s.replace(CAP_LINES, CAP_ANCHOR, 1)
    open(p,'w').write(s)

def add_unordered_cap():
    p=BUF; s=open(p).read()
    if CAP_LINES in s: return  # already there
    assert CAP_ANCHOR in s, "anchor missing for cap add"
    s=s.replace(CAP_ANCHOR, CAP_LINES, 1)
    open(p,'w').write(s)

variants = {
  # name : (base_commit, gin_value, want_unordered_cap)
  'abl-000-base'   : ('cdec521', 11, True),
  'abl-100-C'      : ('cdec521', 11, False),
  'abl-010-Q'      : ('cdec521', 13, True),
  'abl-110-CQ'     : ('cdec521', 13, False),
  'abl-001-F'      : ('3c737dc', 11, True),
  'abl-011-QF'     : ('3c737dc', 13, True),
  'abl-101-CF'     : ('3c737dc', 11, False),
  'abl-111-CQF'    : ('3c737dc', 13, False),
}
for name,(base,gin,cap) in variants.items():
    run(f'git checkout -q {base}')
    run(f'git branch -qD {name} 2>/dev/null; git checkout -q -b {name}')
    set_gin(gin)
    if cap: add_unordered_cap()
    else: remove_unordered_cap()
    # verify
    ng = open(GIN).read()
    import re
    gv = re.search(r'kDefaultGinContextCnt\s+= (\d+)', ng).group(1)
    ncap = open(BUF).read().count('std::min<int>(num_channels_per_sm, 4)')
    run("git add -A && git commit -qam ablation-"+name+" --allow-empty")
    print(f'{name}: base={base} gin={gv} unordered_cap={"yes" if cap else "no"} total_cap_occurrences={ncap}')
run('git checkout -q perf/combine-12sm-pr8')
