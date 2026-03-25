#!/usr/bin/env bash
set -euo pipefail

if [ -f /usr/local/include/gtest/gtest.h ] && \
   [ -f /usr/local/lib/libgtest.a ] && \
   [ -f /usr/local/lib/libgtest_main.a ]; then
    echo "Google Test already installed — skipping."
    exit 0
fi

echo "Installing Google Test v1.14.0..."
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -sL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz \
    | tar xz -C "$TMPDIR"

cmake -S "$TMPDIR/googletest-1.14.0" -B "$TMPDIR/build" \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF

cmake --build "$TMPDIR/build" -j"$(nproc)"
cmake --install "$TMPDIR/build"

echo "Google Test installed successfully."
