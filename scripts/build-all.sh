#!/bin/bash
# Build libhindsight_core.so for all supported targets
# Usage: ./scripts/build-all.sh [release|debug]
#
# Prerequisites:
#   - rustup targets: aarch64-unknown-linux-ohos, aarch64-linux-android, aarch64-unknown-linux-gnu
#   - gcc (used as universal linker for cross-compilation)
#   - OHOS stub libs at ~/ohos-sdk/sysroot/usr/lib/aarch64-linux-ohos/

set -e

PROFILE="${1:-release}"
BUILD_FLAG="--release"
if [ "$PROFILE" = "debug" ]; then
    BUILD_FLAG=""
fi

cd "$(dirname "$0")/.."

echo "Building hindsight-core for all targets ($PROFILE)..."

# 1. Linux ARM64 (native or cross)
echo ""
echo "=== [1/3] aarch64-unknown-linux-gnu (Linux ARM64) ==="
cargo build --target aarch64-unknown-linux-gnu $BUILD_FLAG

# 2. HarmonyOS NEXT
echo ""
echo "=== [2/3] aarch64-unknown-linux-ohos (HarmonyOS NEXT) ==="
cargo build --target aarch64-unknown-linux-ohos $BUILD_FLAG

# 3. Android
echo ""
echo "=== [3/3] aarch64-linux-android (Android ARM64) ==="
CC_aarch64_linux_android=gcc \
CXX_aarch64_linux_android=g++ \
AR_aarch64_linux_android=gcc-ar \
CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=gcc \
cargo build --target aarch64-linux-android $BUILD_FLAG

# Summary
echo ""
echo "=== Build Results ==="
for target in aarch64-unknown-linux-gnu aarch64-unknown-linux-ohos aarch64-linux-android; do
    if [ -f "target/$target/release/libhindsight_core.so" ]; then
        SIZE=$(ls -lh "target/$target/release/libhindsight_core.so" | awk '{print $5}')
        echo "  $target: $SIZE ✓"
    elif [ -f "target/$target/debug/libhindsight_core.so" ]; then
        SIZE=$(ls -lh "target/$target/debug/libhindsight_core.so" | awk '{print $5}')
        echo "  $target: $SIZE ✓ (debug)"
    else
        echo "  $target: FAILED ✗"
    fi
done

echo ""
echo "Done."
