#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo "$SCRIPTPATH"
docker build -t uls23 "$SCRIPTPATH"
