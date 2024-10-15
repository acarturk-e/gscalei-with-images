#!/bin/bash -i
for i in {1..5}; do
  ./run_once_2step2_cpu.sh "data$i" 64 || exit
done
