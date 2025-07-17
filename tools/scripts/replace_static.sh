#!/bin/bash
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Usage:
#   ./replace_static_functions.sh <source_file> [--replace-vars] [--verbose]
#
# - Replaces all 'static' function definitions with non-static.
# - Replaces all 'static inline' with 'inline'.
# - If --replace-vars is given, also replaces 'static' at variable definitions.
# - If --verbose is given, shows a diff of the changes.

set -e

SOURCE_FILE="$1"
REPLACE_VARS=0
VERBOSE=0

for arg in "$@"; do
  if [[ "$arg" == "--replace-vars" ]]; then
    REPLACE_VARS=1
  fi
  if [[ "$arg" == "--verbose" ]]; then
    VERBOSE=1
  fi
done

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "Source file '$SOURCE_FILE' not found!"
  exit 1
fi

TMP_FILE="${SOURCE_FILE}.tmp.$$"

# Regex explanation:
# \b : Word boundary, ensures 'static' and 'inline' are matched as whole words.
# static : Matches the literal word 'static'.
# [[:space:]]+ : Matches one or more whitespace characters (spaces, tabs, etc.) between 'static' and 'inline'.
# inline : Matches the literal word 'inline'.
# \b : Word boundary after 'inline'.
echo "[INFO] Replacing 'static inline' with 'inline' in $SOURCE_FILE"
sed -E 's/\bstatic[[:space:]]+inline\b/inline/g' "$SOURCE_FILE" > "$TMP_FILE"

# Regex explanation:
# ^ : Start of the line.
# ([[:space:]]*) : Captures any leading whitespace at the start of the line (indentation).
# (inline[[:space:]]+|__device__[[:space:]]+|__forceinline__[[:space:]]+|__host__[[:space:]]+|__global__[[:space:]]+|)* :
#   Matches zero or more occurrences of common C/C++/CUDA qualifiers (each followed by whitespace).
# ([[:space:]]*(...|)*) : The outer group allows for any combination/order of these qualifiers.
# static[[:space:]]+ : Matches the literal word 'static' followed by one or more spaces/tabs.
# \1 : In the replacement, refers to the leading whitespace and any qualifiers (without 'static').
#
# Removes 'static' after any qualifiers before the function name
echo "[INFO] Replacing 'static' in function qualifiers in $SOURCE_FILE"
sed -E -i 's/^([[:space:]]*(inline[[:space:]]+|__device__[[:space:]]+|__forceinline__[[:space:]]+|__host__[[:space:]]+|__global__[[:space:]]+|)*)static[[:space:]]+/\1/g' "$TMP_FILE"


# Regex explanation:
# ^ : Start of the line.
# ([[:space:]]*) : Captures any leading whitespace at the start of the line.
# static : Matches the literal word 'static'.
# ([[:space:]]+) : Captures one or more spaces after 'static'.
if [[ "$REPLACE_VARS" == "1" ]]; then
  echo "[INFO] Replacing 'static' at variable definitions in $SOURCE_FILE"
  # This matches 'static' at the start of a line (possibly with spaces), followed by a type and a variable name
  sed -E -i 's/^([[:space:]]*)static([[:space:]]+)/\1/g' "$TMP_FILE"
fi

if [[ "$VERBOSE" == "1" ]]; then
  echo "[INFO] Showing diff for changes:"
  diff -u "$SOURCE_FILE" "$TMP_FILE" || true
fi

mv "$TMP_FILE" "$SOURCE_FILE"
echo "Static function replacement complete for $SOURCE_FILE"
if [[ "$REPLACE_VARS" == "1" ]]; then
  echo "Static variable replacement also performed."
fi

