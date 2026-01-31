#!/bin/bash
set -e

echo "Creating Mac DMG installer..."

cd "$GITHUB_WORKSPACE"

echo "Working directory: $(pwd)"

mkdir -p dist/dmg
rm -rf dist/dmg/*

cp -R dist/Cellects.app dist/dmg/

rm -f dist/Cellects.dmg

create-dmg \
  --volname "Cellects" \
  --volicon "src/cellects/icons/cellects_icon.icns" \
  "dist/Cellects.dmg" \
  "dist/dmg"

ls -lh dist/Cellects.dmg
