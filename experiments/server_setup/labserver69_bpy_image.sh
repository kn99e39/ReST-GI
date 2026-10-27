#!/usr/bin/env bash
# LabServer69 (CentOS 8.4, glibc 2.28): Ubuntu 22.04 runtime image for the RNA venv's bpy 3.5.1 (built for glibc 2.35).
# User-space only (nd is in the docker group); no host configuration is changed.
#   - base: the official ubuntu-base 22.04.5 rootfs (sha256 242cd889...), relayed from LabServer63 because the host's
#     resolver (a stale Tailscale /etc/resolv.conf) cannot resolve registry names, so `docker pull` fails
#   - apt runs inside the build container with an explicit resolver (--dns, the campus DNS of the host's NetworkManager)
#   - packages: the non-bundled shared libraries of bpy/__init__.so (ldd on LabServer63, dpkg -S)
# Usage on LabServer69:  bash labserver69_bpy_image.sh ~/docker_src/ubuntu-base-22.04.5-base-amd64.tar.gz
# Run a script with the image (same /home/nd paths as LabServer63, uid 1013, CPU Cycles - no GPU needed):
#   docker run --rm -u 1013:1013 -v /home/nd:/home/nd -e HOME=/home/nd -w "$PWD" restgi/bpy-runtime:22.04 \
#       /home/nd/NeuralShaderDemo/external/relightable-neural-assets/.venv/bin/python <script.py> -- <args>
set -euo pipefail
TAR=${1:?rootfs tarball}
DNS=${DNS:-210.107.239.131}
echo "242cd8898b33ea806ef5f13b1076ed7c76f9f989d18384452f7166692438ff1a  $TAR" | sha256sum -c -
docker import --change 'CMD ["/bin/bash"]' "$TAR" restgi/ubuntu-base:22.04.5
docker rm -f restgi-bpy-build 2>/dev/null || true
docker run --dns "$DNS" --name restgi-bpy-build restgi/ubuntu-base:22.04.5 bash -c '
  set -e; export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q --no-install-recommends ca-certificates libx11-6 libxext6 libxfixes3 libxi6 libxxf86vm1 libxrender1 \
      libsm6 libice6 libxcb1 libgl1 libglx0 libglvnd0 libegl1 libepoxy0 libfreetype6 libpng16-16 libjpeg-turbo8 libbrotli1 \
      libbz2-1.0 libgomp1 libuuid1 libbsd0 libxkbcommon0 zlib1g
  rm -rf /var/lib/apt/lists/*'
docker commit --change 'CMD ["/bin/bash"]' restgi-bpy-build restgi/bpy-runtime:22.04
docker rm restgi-bpy-build
docker images restgi/bpy-runtime:22.04
