#!/bin/bash
set -e  # Exit on error

echo "Creating Mac DMG installer..."

# Clean and create temporary directory
echo "Preparing temporary directory..."
mkdir -p dist/dmg
rm -rf dist/dmg/*

# Copy app bundle
echo "Copying Cellects.app..."
cp -r "dist/Cellects.app" dist/dmg/

# Remove old DMG if exists
if [ -f "dist/Cellects.dmg" ]; then
    echo "Removing old DMG..."
    rm "dist/Cellects.dmg"
fi

# Create DMG
echo "Building DMG with create-dmg..."
create-dmg \
  --volname "Cellects" \
  --volicon "src/cellects/icons/cellects_icon.icns" \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "Cellects.app" 200 190 \
  --hide-extension "Cellects.app" \
  --app-drop-link 600 185 \
  "dist/Cellects.dmg" \
  "dist/dmg" \
  || true  # create-dmg returns non-zero even on success sometimes

# Verify DMG was created
if [ -f "dist/Cellects.dmg" ]; then
    echo "DMG created successfully: dist/Cellects.dmg"
    ls -lh dist/Cellects.dmg
else
    echo "Error: DMG was not created"
    exit 1
fi
