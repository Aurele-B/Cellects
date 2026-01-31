#!/bin/bash
set -e

echo "Creating Mac DMG installer..."

# Always run relative to the repository root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Working directory: $(pwd)"

echo "Preparing temporary directory..."
mkdir -p dist/dmg
rm -rf dist/dmg/*

echo "Copying Cellects.app..."
cp -R dist/Cellects.app dist/dmg/

echo "Removing old DMG if exists..."
rm -f dist/Cellects.dmg

# Create DMG
echo "Building DMG with create-dmg..."
create-dmg \
  --volname "Cellects" \
  --volicon "src/cellects/icons/cellects_icon.icns" \
  "dist/Cellects.dmg" \
  "dist/dmg"

# Verify DMG was created
if [ -f "dist/Cellects.dmg" ]; then
    echo "DMG created successfully: dist/Cellects.dmg"
    ls -lh dist/Cellects.dmg
else
    echo "Error: DMG was not created"
    exit 1
fi
