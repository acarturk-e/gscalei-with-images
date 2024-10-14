#!/bin/bash -i
for i in {1..5}; do
  ./run_once_2step_cpu.sh "new_data$i" || exit
done
