#!/bin/bash
# taskset -c 0 "$0" "$@"
TARGET_GUID="0c6e4cfffe7117d7"

CURRENT_GUID=$(ibv_devices | awk '/bnxt_re_bond0/ {print $2}')

if [ -z "$CURRENT_GUID" ]; then
    echo "Error: Could not find bnxt_re_bond0 in ibv_devices output."
    exit 1
fi

if [ "$CURRENT_GUID" == "$TARGET_GUID" ]; then
    echo "I am Prefill instance.............................."
    bash proxy.sh
    bash prefill.sh

else
    echo "I am Decode instance.............................."
    bash decode.sh
fi