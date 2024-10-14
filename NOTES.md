- `run_once_cci.sh` may throw errors when we include the `-i`
    interactive option to the `#!/bin/bash -i` shebang line.

## TODO

- make `y` dim 64 since `y` to `z` is linear
- learn score diff on `y`s
- make enc1 100 epochs for convergence
- make enc2 10 epochs for convergence
- In linear autoenc, reconstr loss still dominates.. figure smth out
