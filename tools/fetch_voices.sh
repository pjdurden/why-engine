#!/usr/bin/env bash
# Fetch the Piper TTS voice models. They are 60-115MB each and deliberately not
# committed: the larger one exceeds GitHub's hard 100MB per-file limit and would
# block every push.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets/voices"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"
mkdir -p "$DEST"

fetch() {  # fetch <speaker>/<quality> <name>
  local path="$1" name="$2"
  [ -s "$DEST/$name.onnx" ] && { echo "  have $name"; return; }
  echo "  fetching $name ..."
  curl -fsSL -o "$DEST/$name.onnx"      "$BASE/$path/$name.onnx"
  curl -fsSL -o "$DEST/$name.onnx.json" "$BASE/$path/$name.onnx.json"
}

fetch "ryan/high"    "en_US-ryan-high"      # narration voice: male
fetch "lessac/medium" "en_US-lessac-medium" # fallback
echo "voices ready in $DEST"

# Indian English. Community model, not part of rhasspy/piper-voices, which ships
# only en_GB and en_US -- Piper has no official Indian English voice.
# Licence: GPL-2.0. The model is used locally and never redistributed.
IN_URL="https://huggingface.co/navgurukul-ai/Indian_English_piper/resolve/main"
if [ ! -s "$DEST/en_IN-navgurukul-accent.onnx" ]; then
  echo "  fetching en_IN-navgurukul-accent ..."
  curl -fsSL -o "$DEST/en_IN-navgurukul-accent.onnx"      "$IN_URL/hindi_accent.onnx"
  curl -fsSL -o "$DEST/en_IN-navgurukul-accent.onnx.json" "$IN_URL/hindi_accent.json"
else
  echo "  have en_IN-navgurukul-accent"
fi
