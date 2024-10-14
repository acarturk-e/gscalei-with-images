#!/bin/bash -i
for i in {1..5}; do
  ./run_once_cpu.sh "data$i" || exit
done

