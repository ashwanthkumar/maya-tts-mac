#!/bin/bash
# Build "dist/Maya TTS.app": a double-clickable bundle that runs this checkout's app with its .venv.
# The bundle points at this directory, so keep the project where it is (re-run after moving it).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/Maya TTS.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Maya TTS</string>
  <key>CFBundleDisplayName</key><string>Maya TTS</string>
  <key>CFBundleIdentifier</key><string>local.maya-tts</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>maya-tts</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>Not used; Maya TTS only plays audio.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/maya-tts" <<LAUNCH
#!/bin/bash
cd "$ROOT"
# app/logcap.py writes the log and keeps it under 50 MB the whole time (cut to its last 1 MB when it passes the cap)
exec "$ROOT/.venv/bin/python" -m app.launch --log "\$HOME/Library/Logs/maya-tts.log" >/dev/null 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/maya-tts"

# icon: a speech-bubble glyph rendered with Qt (optional; skipped if anything fails)
"$ROOT/.venv/bin/python" "$ROOT/scripts/make_icon.py" "$APP/Contents/Resources/AppIcon.icns" || true
echo "Built $APP"
