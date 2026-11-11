# Retargeting all of BONES-SEED to the Unitree R1

Three SLURM stages turn the raw BONES-SEED corpus into a training set for
`02_train_any2any.slurm`. Everything is configured in `env.sh`; the job scripts
read paths from there and nothing else.

| stage | job | GPU | output | rough cost |
|---|---|---|---|---|
| 1 | `10_extract_bvh.slurm` | no | `$BVH_ROOT` on scratch | ~295 GB, a few hours |
| 2 | `11_retarget_r1.slurm` | yes, array | `$R1_CSV` in **home** | ~35 GPU-hours total, ~45 GB |
| 3 | `12_convert_motion_lib.slurm` | no | `$R1_PKL` + `$FILTERED` in **home** | ~30 GB compressed, ~1 h |

Scratch holds the bulky intermediate (BVH); home holds everything a training
job needs, so the dataset is readable from any node rather than only `zac`.

**Check your home quota before you start.** The CSVs and the PKLs together are
about 75 GB, and `$FILTERED` adds another ~27 GB because the filter copies
rather than links. Budget **~105 GB in home**. If that is too much, delete
`$R1_CSV` after stage 3 — nothing downstream reads it — which brings the
steady state to ~57 GB.

---

## Before: four things to set up once

### 1. Get the R1 target onto the HPC — mind Git LFS

The `unitree_r1` target is not in upstream soma-retargeter; it lives on the
`unitree-r1-target` branch of the DIRECTLab fork. soma-retargeter tracks
`*.STL` and `*.urdf` through **Git LFS**, so a clone without git-lfs active
gives you 130-byte pointer files instead of the 22 MB of geometry — and that
fails ~40 minutes into a job, not at checkout.

git-lfs must be installed *and* initialised. On a shared cluster you usually
cannot write to the admin-owned conda `base`, so install it into an env you own
(`conda create --prefix ~/envs/tools -c conda-forge git-lfs`), via `module`, or
as the standalone binary.

```bash
git lfs install                                   # once per user
cd ~/repos/soma-retargeter
git checkout unitree-r1-target
git lfs pull --include="assets/robotics/unitree/r1/**"   # 22 MB, 43 files
```

Installing git-lfs after cloning does not retroactively fix the working tree;
`git lfs pull` does. Then confirm the geometry really arrived, rather than
trusting the file list:

```bash
du -sh ~/repos/soma-retargeter/assets/robotics/unitree/r1   # expect ~22M, not ~50K
head -1 ~/repos/soma-retargeter/assets/robotics/unitree/r1/desc/r1_26dof_retarget.urdf
# must be XML. "version https://git-lfs.github.com/spec/v1" means you moved pointers.
```

### 2. soma-retargeter itself

Stage 2 runs soma-retargeter, which is a different repo **and a different
interpreter**. It needs python >= 3.12 and `env_isaaclab` is 3.11, so it cannot
share the SONIC environment — it has to have one of its own.

The upstream way is `uv`, which installs per-user with no root and will fetch
its own 3.12 if the system has none:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # lands in ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"                # env.sh also adds this
cd ~/repos/soma-retargeter && uv sync
```

If you would rather not add uv, any python 3.12 environment works — point
`SOMA_PYTHON` in `env.sh` at its interpreter and nothing else changes:

```bash
conda create -y --prefix ~/envs/soma python=3.12
conda run -p ~/envs/soma pip install -e ~/repos/soma-retargeter
# in env.sh:
export SOMA_PYTHON=$HOME/envs/soma/bin/python
```

`uv sync` honours `uv.lock`, so it reproduces the exact dependency set;
`pip install -e` resolves afresh against the pins in `pyproject.toml`. Both
work — the lockfile is just stricter.

Job 11 pre-flights this by importing `newton`, `warp` and `soma_retargeter`
through `$SOMA_PYTHON` and refuses to start if that fails, so a missing or
wrong environment costs seconds rather than being discovered mid-array.

Smoke-test it on an interactive GPU before committing 35 GPU-hours. This
exercises the robot registry, the URDF load and the meshes — the three things
an LFS-pointer mistake breaks:

```bash
srun --nodelist=zac --gres=gpu:1 --pty bash
cd ~/repos/soma-retargeter
uv run python -c "
from soma_retargeter.robotics import robot_registry
robot_registry.register_robots_path('assets/robotics')
print(robot_registry.list_available_targets())"   # must include unitree_r1
```

### 3. Point `env.sh` at your paths

Four variables decide everything:

```bash
CLUSTER_NODE=bowser                  # the node that holds the data
RETARGET_SHARDS=2                    # = GPUs to use; drives --array
SEED_ROOT=$HOME/seed                 # where soma_uniform.tar.gz is
BVH_ROOT=/scratch/$USER/seed/soma_uniform/bvh
R1_CSV=$HOME/seed/r1/csv             # retargeted CSVs
R1_PKL=$HOME/seed/r1/motion_lib      # motion_lib PKLs
FILTERED=$HOME/seed/r1/filtered      # what training reads
```

**`SEED_ROOT` does not have to be on the same filesystem as `BVH_ROOT`.** The
normal case after moving the corpus between nodes is a tarball in home and the
extracted tree on scratch: 45 GB crosses the network, the 295 GB it unpacks to
never does.

`CLUSTER_NODE` and `RETARGET_SHARDS` are the only places the node and the array
width are written down. The job files still carry `#SBATCH --nodelist=zac` and
`#SBATCH --array=0-1`, but `submit_retarget.sh` passes both on the command
line, and a command-line value overrides a directive. Editing the job files is
not necessary and will only create a second thing to keep in sync.

`FILTERED` **changed** from `/scratch/$USER/seed/r1_filtered` to home. That is
what makes the dataset visible off whichever node built it, but it also means
`01_prepare_data.slurm` now targets a different directory than it used to.

### 4. Pick the shard count

Size it by the **time limit**, not the GPU count. Job 11 asks for 12 h, and the
corpus is ~35 GPU-hours on a 4090 or ~23 on a Blackwell, so one task has to
cover at most a twelfth of the work:

| shards | per task, 4090 | per task, Blackwell | |
|---|---|---|---|
| 1 | 34.8 h | 23.2 h | over the limit |
| 2 | 17.4 h | 11.6 h | over / no margin |
| 4 | 8.7 h | 5.8 h | ok |
| **8** | **4.3 h** | **2.9 h** | **default** |

Shards are not capped by GPUs. Slurm runs as many concurrently as there are
free GPUs and queues the rest, so an array wider than the node adapts by
itself — it picks up a second GPU the moment another job releases one, and a
shard that does time out loses only its own remainder, because
`make_retarget_shard.py` resumes. Oversharding is the safe direction.

See what the node has and what is already taken:

```bash
scontrol show node $CLUSTER_NODE | grep -E 'Gres|AllocTRES|State'
squeue -w $CLUSTER_NODE -o '%.10i %.8u %.2t %.10M %.10L %.20b %R'
```

`%L` is time left on someone else's job — the limit minus elapsed, so an upper
bound rather than a prediction.

---

## Run

One command submits the whole chain and returns:

```bash
cd ~/repos/GR00T-WholeBodyControl
./gear_sonic/slurm_scripts/submit_retarget.sh
```

| flag | effect |
|---|---|
| *(none)* | extract → retarget → motion_lib |
| `--skip-extract` | BVH already unpacked at `$BVH_ROOT` |
| `--retarget-only` | stop after the CSVs |
| `--with-training` | also chain `02_train_any2any.slurm` |

Every link is `afterok`, so a failed stage stops the chain rather than letting
the next one run on a partial dataset. Submit it and log out; `squeue -u $USER`
shows where it has got to.

There is deliberately no single "master job". A job that held an allocation
while waiting for its own children would occupy a GPU for the hours stages 1
and 3 spend on CPU work, and a crash in it would strand everything. Slurm
dependencies sequence it for free and each stage asks only for what it needs.

To run a stage by hand, pass the node yourself, since the directives still say
`zac`:

```bash
sbatch --nodelist=$CLUSTER_NODE gear_sonic/slurm_scripts/12_convert_motion_lib.slurm
```

### If a shard dies

Re-submit the array. `make_retarget_shard.py` links only the clips whose CSV is
missing or zero-length, so a re-run picks up exactly where it stopped and
recomputes nothing. A timeout on one shard costs that shard's remainder, not
the run.

`afterok` will not have fired, so stage 3 was never submitted. Re-running the
submitter with `--skip-extract` re-queues both, and the retargeting array
finishes the stragglers before conversion starts:

```bash
./gear_sonic/slurm_scripts/submit_retarget.sh --skip-extract
```

---

## After: train

Stage 3 ends by printing the clip count in `$FILTERED`. Then:

```bash
./gear_sonic/slurm_scripts/submit_all.sh --train-only
```

`02_train_any2any.slurm` reads `$FILTERED` and needs no changes. Consider
bumping `EXPERIMENT_NAME` in `env.sh` first — `auto_load_latest=true` means an
unchanged name resumes the previous run's checkpoint instead of starting a new
one on the new data.

---

## What the stages actually do

**Sharding without touching the converter.** `bvh_to_csv_converter.py` takes one
`import_folder`, rglobs it, and names each output from the path relative to that
folder. It has no notion of a shard and no notion of resuming. Both come from
giving each array task a directory of **symlinks**: clip *i* goes to task
`i % count`, and a clip that already has a CSV is simply not linked. Because the
symlinks keep the `<session>/<clip>.bvh` layout, the CSVs land as
`<session>/<clip>.csv` — a few hundred directories instead of one with 142k
entries in it. Clips are ordered by size before striding, so no shard gets all
the long motions.

**All of BONES-SEED, then filtered.** Stage 2 retargets the whole corpus, and
stage 3 applies the same 40-keyword exclusion list G1 SONIC was trained with
(shared code, not a reimplementation — `filter_and_copy_bones_data.py`). About
9% is dropped. The unfiltered `$R1_PKL` stays on disk, so the motions the G1
could not perform are still there if you want them later.

**Frame rate.** PKLs are stored at the CSVs' native 120 fps and motion_lib
resamples to its own `target_fps` (50) at load. One canonical interpolation
rather than two, and 120 → 50 is not an integer stride.

**Compression.** `PKL_COMPRESS=3` takes the corpus from ~86 GB to ~30 GB, mostly
by squeezing the mandatory all-zero `smpl_joints` field. It costs decompression
time at every motion_lib load; set it to 0 in `env.sh` to trade the disk back.

**`metadata.pkl`.** The filter skips it, so stage 3 copies the unfiltered one
into `$FILTERED`. Metadata is allowed to be a superset of what is on disk
(`motion_lib_base.py:411`). Without it, the motion library opens all ~130k clips
at startup just to read their frame counts.
