#!/bin/bash
set -e

echo "Creating Mac DMG installer..."
echo "Working directory: $(pwd)"

echo "Removing old DMG files..."
rm -f *.dmg
rm -f dist/*.dmg

echo "Building DMG with create-dmg..."
create-dmg dist/Cellects.app || true

echo "Finding created DMG..."
DMG_FILE=$(find . -maxdepth 1 -name "*.dmg" -type f 2>/dev/null | head -1)

if [ -n "$DMG_FILE" ]; then
    mv "$DMG_FILE" dist/Cellects.dmg
    echo "DMG created successfully: dist/Cellects.dmg"
    ls -lh dist/Cellects.dmg
else
    echo "Error: DMG was not created"
    ls -la
    ls -la dist/ || true
    exit 1
fi