#!/usr/bin/env bash
# Download the public CirCor DigiScope dataset (v1.0.3, zip of ~470 MB) next to the repository.
#
#   bash scripts/download_data.sh [destination]
#
# The download resumes where it stopped if it is interrupted (just run it again).
set -euo pipefail

DEST="${1:-../circor-dataset}"   # the path used by configs/base.yaml (data.dir)
URL="https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/"
ZIP="$DEST/circor-1.0.3.zip"

mkdir -p "$DEST"
echo "downloading to $ZIP ..."
curl -L --fail --retry 5 -C - -o "$ZIP" "$URL"

echo "extracting ..."
TMP="$(mktemp -d "$DEST/extract.XXXX")"
unzip -q "$ZIP" -d "$TMP"
SRC="$(dirname "$(find "$TMP" -name training_data.csv | head -n 1)")"
rsync -a "$SRC"/ "$DEST"/
rm -rf "$TMP"

echo "patients: $(($(wc -l < "$DEST/training_data.csv") - 1)), recordings: $(ls "$DEST"/training_data/*.wav | wc -l)"
echo "done (you can delete $ZIP)"
