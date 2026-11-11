#!/bin/bash
#SBATCH --job-name=r1-extract-bvh
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --nodelist=zac
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=logs/r1-extract-%j.out
# Pinned to zac: /scratch is node-local and only zac holds BONES-SEED.

# Stage 1 of 3. Puts the soma_uniform BVH corpus on scratch as $BVH_ROOT.
#
# BONES-SEED ships soma_uniform as a single 45 GB .tar.gz that unpacks to about
# 295 GB -- BVH is plain text, so it compresses roughly 6.5x. That is why this
# lands on scratch and not in home: only the retargeted CSVs need to be shared
# across nodes, and they are a seventh of the size.
#
# Nothing here is R1-specific; if someone else already extracted the corpus,
# point BVH_ROOT at it in env.sh and skip this job entirely.
#
# Safe to re-run: it refuses to extract over a populated $BVH_ROOT rather than
# spending six hours rewriting files that are already there.

set -euo pipefail
REQUIRE_RAW=0 source "$HOME/repos/GR00T-WholeBodyControl/gear_sonic/slurm_scripts/env.sh"

TARBALL=$SEED_ROOT/soma_uniform.tar.gz
# The tarball's own top-level directory, so BVH_ROOT must end in soma_uniform/bvh.
DEST=$(dirname "$(dirname "$BVH_ROOT")")

echo "node:    $(hostname)"
echo "source:  $SEED_ROOT"
echo "dest:    $BVH_ROOT"
echo

# --- already extracted, here or at the source? ------------------------------
if [ -d "$BVH_ROOT" ] && [ -n "$(ls -A "$BVH_ROOT" 2>/dev/null)" ]; then
    n=$(find "$BVH_ROOT" -name '*.bvh' | wc -l)
    echo "$BVH_ROOT already holds $n BVH files -- nothing to do."
    echo "Delete it to re-extract."
    exit 0
fi

if [ -d "$SEED_ROOT/soma_uniform/bvh" ]; then
    n=$(find "$SEED_ROOT/soma_uniform/bvh" -name '*.bvh' | wc -l)
    echo "The corpus is already extracted at the source ($n BVH files)."
    echo "Set BVH_ROOT=$SEED_ROOT/soma_uniform/bvh in env.sh and skip this job."
    exit 0
fi

[ -f "$TARBALL" ] || { echo "FATAL: neither $BVH_ROOT, $SEED_ROOT/soma_uniform/bvh nor $TARBALL exists"; exit 1; }

# --- disk check -------------------------------------------------------------
# Checked up front because tar fills the filesystem and then fails, leaving a
# half-extracted tree that looks complete to anything counting directories.
# $DEST is created first so df has something to stat: on a fresh node
# /scratch/$USER may not exist yet, and df on a missing path returns nothing,
# which would make the comparison below a bash syntax error under `set -e`.
mkdir -p "$DEST"
need_gb=320
avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
echo "free on $DEST: ${avail_gb}G, need ~${need_gb}G"
if [ "$avail_gb" -lt "$need_gb" ]; then
    echo "FATAL: not enough space to extract soma_uniform."
    echo "       Free space, or extract a subset and point BVH_ROOT at it."
    exit 1
fi

# --- extract ----------------------------------------------------------------
# The tarball does not have to live next to the extracted tree. SEED_ROOT in
# home and BVH_ROOT on scratch is the normal case when the corpus was copied
# between nodes: 45 GB crosses the network, the 295 GB it unpacks to does not.
echo "extracting $(du -h "$TARBALL" | cut -f1) ... (expect a few hours)"
# Only the bvh/ tree: the archive also carries per-clip metadata the retargeter
# never reads. --wildcards is explicit rather than relying on the extraction
# default, and --no-same-owner because the archive was built elsewhere.
tar -xzf "$TARBALL" -C "$DEST" --no-same-owner --wildcards 'soma_uniform/bvh/*'

n=$(find "$BVH_ROOT" -name '*.bvh' | wc -l)
echo
echo "=== extracted $n BVH files to $BVH_ROOT ==="
du -sh "$BVH_ROOT"
[ "$n" -gt 0 ] || { echo "FATAL: extraction produced no BVH"; exit 1; }
