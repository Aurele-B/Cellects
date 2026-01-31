#!/bin/bash
set -e  # Exit on error

echo "Creating Mac DMG installer..."

echo "Preparing temporary directory..."
mkdir -p dist/dmg
rm -rf dist/dmg/*

echo "Copying Cellects.app..."
cp -r "dist/Cellects.app" dist/dmg/

echo "Removing old DMG if exists..."
rm -f "Cellects.dmg"

echo "Building DMG with create-dmg..."
create-dmg dist/dmg || true

# The tool creates a DMG in the current directory, find and move it
DMG_FILE=$(ls *.dmg 2>/dev/null | head -1)
if [ -n "$DMG_FILE" ]; then
    mv "$DMG_FILE" dist/Cellects.dmg
    echo "DMG created successfully: dist/Cellects.dmg"
    ls -lh dist/Cellects.dmg
else
    echo "Error: DMG was not created"
    exit 1
fi

# Verify DMG was created
if [ -f "dist/Cellects.dmg" ]; then
    echo "DMG created successfully: dist/Cellects.dmg"
    ls -lh dist/Cellects.dmg
else
    echo "Error: DMG was not created"
    exit 1
fi
