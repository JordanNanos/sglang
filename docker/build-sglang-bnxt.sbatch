#!/bin/bash
#SBATCH --job-name=sgl-build-bnxt
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/nfsdata/sa/.j9s/sglang/docker/build-bnxt-%j.log
#SBATCH --error=/nfsdata/sa/.j9s/sglang/docker/build-bnxt-%j.log

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
IMAGE="ghcr.io/jordannanos/sgl-mi325x-mori:v0.5.9-bnxt"
CACHE_FROM="${IMAGE}"
DOCKERFILE="/nfsdata/sa/.j9s/sglang/docker/rocm.Dockerfile"
BUILD_CONTEXT="/nfsdata/sa/.j9s/sglang/docker/"
ENV_FILE="/nfsdata/sa/.j9s/.env.local"

# ── Load PAT ────────────────────────────────────────────────────────────
set -a
source "${ENV_FILE}"
set +a

if [ -z "${GH_PAT:-}" ]; then
    echo "ERROR: GH_PAT not set in ${ENV_FILE}"
    exit 1
fi

# ── GHCR Login ──────────────────────────────────────────────────────────
echo "Logging in to ghcr.io..."
echo "${GH_PAT}" | docker login ghcr.io -u jordannanos --password-stdin

# ── Build ───────────────────────────────────────────────────────────────
echo "Starting build at $(date)"
echo "Image: ${IMAGE}"

docker build \
    --cache-from "${CACHE_FROM}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg SGL_BRANCH=v0.5.9 \
    --build-arg GPU_ARCH=gfx942 \
    --build-arg ENABLE_MORI=1 \
    --build-arg NIC_BACKEND=ibgda \
    -t "${IMAGE}" \
    -f "${DOCKERFILE}" \
    "${BUILD_CONTEXT}"

echo "Build finished at $(date)"

# ── Push ────────────────────────────────────────────────────────────────
echo "Pushing ${IMAGE}..."
docker push "${IMAGE}"

echo "Done. Image pushed: ${IMAGE}"
