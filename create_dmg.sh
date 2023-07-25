#!/bin/sh

mkdir -p dist/dmg

rm -r dist/dmg/*

cp -r "dist/Cellects.app" dist/dmg

test -f "dist/Cellects.dmg" && rm "dist/Cellects.dmg"

create-dmg \
  --volname "Cellects" \
  --volicon "myicon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "Cellects.app" 200 190 \
  --hide-extension "Cellects.app" \
  --app-drop-link 600 185 \
  "dist/Cellects.dmg" \
  "dist/dmg"