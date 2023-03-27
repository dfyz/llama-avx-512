#!/bin/sh

set -eu

sudo cpupower frequency-set --governor performance >/dev/null
./main --benchmark_time_unit=ms
sudo cpupower frequency-set --governor powersave >/dev/null