#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local}"
WORKDIR="${WORKDIR:-$HOME/.cache/vcdd/vina-gpu-2.1}"
BOOST_ROOT="${BOOST_ROOT:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PLATFORM="NVIDIA"
BOX="SMALL"
OPENCL_VERSION="2_0"
VARIANTS=()
REPO="https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git"

declare -A DIRS=(
  [vina]="AutoDock-Vina-GPU-2.1"
  [quickvina2]="QuickVina2-GPU-2.1"
  [quickvinaw]="QuickVina-W-GPU-2.1"
)

declare -A BINS=(
  [vina]="AutoDock-Vina-GPU-2-1"
  [quickvina2]="QuickVina2-GPU-2-1"
  [quickvinaw]="QuickVina-W-GPU-2-1"
)

usage() {
  cat <<'USAGE'
Usage:
  ./build_vina_gpu.sh [options]

Options:
  --variant NAME        one of: vina | quickvina2 | quickvinaw
  --all                 build all three variants
  --boost PATH          path to boost source tree
  --cuda PATH           CUDA root, default: /usr/local/cuda
  --prefix PATH         install prefix, default: ~/.local
  --workdir PATH        local source tree or clone/build dir
  --amd                 build for AMD platform
  --large-box           compile with LARGE_BOX
  --opencl-1-2          compile with OPENCL_1_2
  -h, --help            show help
USAGE
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)    VARIANTS+=("$2"); shift 2 ;;
    --all)        VARIANTS=(vina quickvina2 quickvinaw); shift ;;
    --boost)      BOOST_ROOT="$2"; shift 2 ;;
    --cuda)       CUDA_HOME="$2"; shift 2 ;;
    --prefix)     PREFIX="$2"; shift 2 ;;
    --workdir)    WORKDIR="$2"; shift 2 ;;
    --amd)        PLATFORM="AMD"; shift ;;
    --large-box)  BOX="LARGE"; shift ;;
    --opencl-1-2) OPENCL_VERSION="1_2"; shift ;;
    -h|--help)    usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(vina)

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mxx  %s\033[0m\n' "$*" >&2; exit 1; }

case "$OPENCL_VERSION" in
  1_2) CL_TARGET_OPENCL_VERSION=120 ;;
  2_0) CL_TARGET_OPENCL_VERSION=200 ;;
  3_0) CL_TARGET_OPENCL_VERSION=300 ;;
  *) die "unsupported OPENCL_VERSION=$OPENCL_VERSION" ;;
esac

say "preflight"

command -v make >/dev/null || die "make not found"
command -v python3 >/dev/null || die "python3 not found"

if [[ -x /usr/bin/g++ ]]; then
  GXX_BIN=/usr/bin/g++
else
  GXX_BIN="$(command -v g++)"
fi
[[ -n "${GXX_BIN:-}" && -x "$GXX_BIN" ]] || die "g++ not found"

say "C++ compiler: $GXX_BIN"

if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L | sed 's/^/    /'
else
  warn "no GPU visible right now; runtime still needs a working GPU/OpenCL"
fi

# Prefer system OpenCL on Ubuntu, then CUDA, then ROCm
OPENCL_LIB_DIR=""
OPENCL_INC_DIR=""
for pair in \
  "/usr/lib/x86_64-linux-gnu:/usr/include" \
  "/usr/lib64:/usr/include" \
  "$CUDA_HOME/lib64:$CUDA_HOME/include" \
  "$CUDA_HOME/lib:$CUDA_HOME/include" \
  "/opt/rocm/lib:/opt/rocm/include" \
  "/opt/rocm/opencl/lib:/opt/rocm/opencl/include"
do
  libdir="${pair%%:*}"
  incdir="${pair##*:}"
  if ls "$libdir"/libOpenCL.so* >/dev/null 2>&1; then
    OPENCL_LIB_DIR="$libdir"
    OPENCL_INC_DIR="$incdir"
    break
  fi
done

[[ -n "$OPENCL_LIB_DIR" ]] || die "libOpenCL.so not found"
[[ -f "$OPENCL_INC_DIR/CL/opencl.h" || -f "$OPENCL_INC_DIR/CL/cl.h" ]] || die "OpenCL headers not found under $OPENCL_INC_DIR"

say "OpenCL include: $OPENCL_INC_DIR"
say "OpenCL lib:     $OPENCL_LIB_DIR"

if [[ -z "$BOOST_ROOT" ]]; then
  for cand in \
    "$PWD/boost_1_77_0" "$PWD/boost_1_84_0" \
    "$HOME/boost_1_77_0" "$HOME/boost_1_84_0" \
    "$WORKDIR/boost_1_77_0" "$WORKDIR/boost_1_84_0" \
    "/usr/local/boost_1_77_0" "/usr/local/boost_1_84_0" \
    "/opt/boost_1_77_0" "/opt/boost_1_84_0"
  do
    [[ -d "$cand" ]] && BOOST_ROOT="$cand" && break
  done
fi

[[ -n "$BOOST_ROOT" && -d "$BOOST_ROOT" ]] || die "Boost source tree not found; pass --boost /path/to/boost_1_xx_0"
[[ -f "$BOOST_ROOT/boost/version.hpp" ]] || die "invalid Boost tree: $BOOST_ROOT"

for lib in program_options system filesystem; do
  ls "$BOOST_ROOT"/stage/lib/libboost_"$lib"* >/dev/null 2>&1 || die "missing $BOOST_ROOT/stage/lib/libboost_$lib*"
done

say "Boost:          $BOOST_ROOT"

# Add compatibility header for newer Boost
if [[ ! -f "$BOOST_ROOT/boost/filesystem/convenience.hpp" ]]; then
  say "adding Boost compatibility header: boost/filesystem/convenience.hpp"
  mkdir -p "$BOOST_ROOT/boost/filesystem"
  cat > "$BOOST_ROOT/boost/filesystem/convenience.hpp" <<'HPP'
#ifndef BOOST_FILESYSTEM_CONVENIENCE_HPP
#define BOOST_FILESYSTEM_CONVENIENCE_HPP

#include <string>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace boost {
namespace filesystem {

inline std::string basename(const path& p) {
    return p.stem().string();
}

inline std::string extension(const path& p) {
    return p.extension().string();
}

inline path change_extension(const path& p, const std::string& new_ext) {
    path q = p;
    q.replace_extension(new_ext);
    return q;
}

} // namespace filesystem
} // namespace boost

#endif
HPP
fi

# Reuse existing local source tree if present; otherwise clone
if [[ -d "$WORKDIR/AutoDock-Vina-GPU-2.1" || -d "$WORKDIR/QuickVina2-GPU-2.1" || -d "$WORKDIR/QuickVina-W-GPU-2.1" ]]; then
  say "reusing local source tree at $WORKDIR"
elif [[ -d "$WORKDIR/.git" ]]; then
  say "reusing clone at $WORKDIR"
else
  command -v git >/dev/null || die "git not found and local source tree not present"
  mkdir -p "$(dirname "$WORKDIR")"
  say "cloning $REPO -> $WORKDIR"
  git clone --depth 1 "$REPO" "$WORKDIR"
fi

install -d "$PREFIX/bin" "$PREFIX/libexec/vcdd" "$PREFIX/share/vcdd/vina-gpu-kernels"

for v in "${VARIANTS[@]}"; do
  dir="${DIRS[$v]:-}"
  bin="${BINS[$v]:-}"
  [[ -n "$dir" && -n "$bin" ]] || die "unknown variant '$v'"

  src="$WORKDIR/$dir"
  mk="$src/Makefile"
  [[ -d "$src" ]] || die "expected source dir missing: $src"
  [[ -f "$mk" ]] || die "expected Makefile missing: $mk"

  say "patching Makefile for $dir"

  python3 - "$mk" "$src" "$BOOST_ROOT" "$OPENCL_INC_DIR" "$OPENCL_LIB_DIR" "$PLATFORM" "$BOX" "$OPENCL_VERSION" "$CL_TARGET_OPENCL_VERSION" <<'PY'
import re
import sys
from pathlib import Path

mk, work, boost, ocl_inc, ocl_lib, platform, box, oclver, cl_target = sys.argv[1:10]
p = Path(mk)
text = p.read_text()

def set_line(txt, name, value):
    pat = rf'^{re.escape(name)}\s*=.*$'
    rep = f'{name}={value}'
    txt2, n = re.subn(pat, rep, txt, flags=re.M)
    if n == 0:
        if not txt2.endswith("\n"):
            txt2 += "\n"
        txt2 += rep + "\n"
    return txt2

text = set_line(text, 'WORK_DIR', work)
text = set_line(text, 'BOOST_LIB_PATH', boost)
text = set_line(text, 'OPENCL_LIB_PATH', ocl_lib.rsplit('/lib', 1)[0] if '/lib' in ocl_lib else ocl_lib)
text = set_line(text, 'OPENCL_VERSION', f'-DOPENCL_{oclver}')
text = set_line(text, 'GPU_PLATFORM', f'-D{platform}_PLATFORM')
text = set_line(text, 'DOCKING_BOX_SIZE', f'-D{box}_BOX')
text = set_line(text, 'BOOST_INC_PATH', '-I$(BOOST_LIB_PATH) -I$(BOOST_LIB_PATH)/boost')
text = set_line(text, 'VINA_GPU_INC_PATH', '-I$(WORK_DIR)/lib -I$(WORK_DIR)/OpenCL/inc')
text = set_line(text, 'OPENCL_INC_PATH', f'-I{ocl_inc}')
text = set_line(text, 'LIB_PATH', f'-L$(BOOST_LIB_PATH)/stage/lib -L{ocl_lib}')
text = set_line(text, 'MACRO', f'$(OPENCL_VERSION) $(GPU_PLATFORM) $(DOCKING_BOX_SIZE) -DBOOST_TIMER_ENABLE_DEPRECATED -DCL_TARGET_OPENCL_VERSION={cl_target}')
text = set_line(text, 'OPTION', f'-Wl,-rpath,{boost}/stage/lib -Wl,-rpath,{ocl_lib}')

p.write_text(text)
PY

  python3 - "$mk" "$GXX_BIN" <<'PY_FIX_MAKE'
from pathlib import Path
import sys

mk, gxx = sys.argv[1], sys.argv[2]
p = Path(mk)
lines = p.read_text().splitlines()
out = []

for line in lines:
    s = line.lstrip()
    if s.startswith("gcc -o "):
        out.append("\t" + s.replace("gcc -o ", f"{gxx} -o ", 1))
    elif s.startswith("g++ -o "):
        out.append("\t" + s.replace("g++ -o ", f"{gxx} -o ", 1))
    elif s.startswith("rm "):
        out.append("\t" + s)
    else:
        out.append(line)

p.write_text("\n".join(out) + "\n")
PY_FIX_MAKE

  say "building source-mode binary for $bin"
  (
    cd "$src"
    ulimit -s 8192 || true
    make clean >/dev/null 2>&1 || true
    make source -j1
  )
  [[ -x "$src/$bin" ]] || die "source build failed: $src/$bin missing"
  install -m 0755 "$src/$bin" "$PREFIX/libexec/vcdd/$bin.source"

  say "building normal binary for $bin"
  (
    cd "$src"
    ulimit -s 8192 || true
    make clean >/dev/null 2>&1 || true
    make -j1
  )
  [[ -x "$src/$bin" ]] || die "normal build failed: $src/$bin missing"
  install -m 0755 "$src/$bin" "$PREFIX/libexec/vcdd/$bin"

  kdir="$PREFIX/share/vcdd/vina-gpu-kernels/$bin"
  install -d "$kdir"
  for k in Kernel1_Opt.bin Kernel2_Opt.bin; do
    [[ -f "$src/$k" ]] && install -m 0644 "$src/$k" "$kdir/$k"
  done

  cat > "$PREFIX/bin/$bin" <<WRAP
#!/usr/bin/env bash
set -euo pipefail
ulimit -s 8192 >/dev/null 2>&1 || true

REAL="$PREFIX/libexec/vcdd/$bin"
SOURCE="$PREFIX/libexec/vcdd/$bin.source"
KDIR="$PREFIX/share/vcdd/vina-gpu-kernels/$bin"

mkdir -p "\$KDIR"

args=( "\$@" )
has_opencl_path=0
for ((i=0; i<\${#args[@]}; i++)); do
  if [[ "\${args[i]}" == "--opencl_binary_path" ]]; then
    has_opencl_path=1
    break
  fi
done

if [[ \$has_opencl_path -eq 0 ]]; then
  args+=(--opencl_binary_path "\$KDIR")
fi

if [[ -s "\$KDIR/Kernel1_Opt.bin" && -s "\$KDIR/Kernel2_Opt.bin" ]]; then
  exec "\$REAL" "\${args[@]}"
else
  echo "[vina-gpu] no cached OpenCL kernels in \$KDIR; using source-build once to generate them." >&2
  exec "\$SOURCE" "\${args[@]}"
fi
WRAP
  chmod 0755 "$PREFIX/bin/$bin"

  say "installed wrapper: $PREFIX/bin/$bin"
done

cat <<DONE

==> done

Add to PATH if needed:
    export PATH="$PREFIX/bin:\$PATH"

Installed:
  wrappers: $PREFIX/bin
  binaries: $PREFIX/libexec/vcdd
  kernels:  $PREFIX/share/vcdd/vina-gpu-kernels
DONE
