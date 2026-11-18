#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "$0")"; pwd)
cd "$SCRIPT_PATH"

if [ ! -d "build" ]; then
  mkdir build
  echo "build create successfully."
else
  echo "build does not exists."
fi

export ASCEND_HOME_PATH=$1

rm -rf ./build/*
cd build
cmake ..
make -j12
cd ..
chmod 550 -R build
