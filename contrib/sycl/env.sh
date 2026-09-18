# Not required for `crane-serve` or `chat_cli` — they set this up themselves
# (binaries bake an rpath to the kernel libraries, and
# `crane_core::utils::sycl_env::ensure_sycl_runtime_env` re-execs once with the
# oneAPI runtime on LD_LIBRARY_PATH). Kept for other example binaries, for a
# shell where you want `icpx`/`sycl-ls` on PATH, and as the escape hatch.
#
#   source contrib/sycl/env.sh                    # release .so (default)
#   CRANE_SYCL_PROFILE=debug source contrib/sycl/env.sh
#
# Sets:
#  - LD_LIBRARY_PATH for the out-of-tree libcandle_sycl.so / libcrane_gdn_sycl.so
#  - UR_LOADER_USE_LEVEL_ZERO_V2=0 — oneAPI 2026.x defaults to the Level-Zero V2
#    adapter, which fails on kernel launch on Battlemage (B70); see README.md.
#
# LD_LIBRARY_PATH beats the binary's RUNPATH, so pick one profile's OUT_DIRs,
# newest first, rather than whatever `find` returns first — otherwise a release
# binary can silently run a stale debug-profile kernel library.

# Via the environment, not $1: `source` with no arguments passes the calling
# script's positional parameters through.
_crane_sycl_profile="${CRANE_SYCL_PROFILE:-release}"
case "$_crane_sycl_profile" in
  debug | release) ;;
  *)
    echo "contrib/sycl/env.sh: CRANE_SYCL_PROFILE must be debug or release," \
         "got '$_crane_sycl_profile'; using release" >&2
    _crane_sycl_profile=release
    ;;
esac
_crane_sycl_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

export UR_LOADER_USE_LEVEL_ZERO_V2=0

_crane_sycl_libdirs="$(
  find "$_crane_sycl_repo/target/$_crane_sycl_profile/build" \
       \( -name 'libcandle_sycl.so' -o -name 'libcrane_gdn_sycl.so' \) 2>/dev/null |
    xargs -rn1 ls -td 2>/dev/null |
    xargs -rn1 dirname |
    awk '!seen[$0]++' |
    paste -sd: -
)"
if [ -z "$_crane_sycl_libdirs" ]; then
  echo "contrib/sycl/env.sh: no libcandle_sycl.so / libcrane_gdn_sycl.so under" \
       "$_crane_sycl_repo/target/$_crane_sycl_profile — build with --features sycl first" >&2
else
  export LD_LIBRARY_PATH="${_crane_sycl_libdirs}:${LD_LIBRARY_PATH}"
  echo "contrib/sycl/env.sh: SYCL kernels from target/$_crane_sycl_profile" >&2
fi
unset _crane_sycl_libdirs _crane_sycl_profile _crane_sycl_repo
