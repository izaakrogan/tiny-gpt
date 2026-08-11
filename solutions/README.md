# Solutions

Reference implementations for steps 2, 3 and 4. Have a proper attempt at
the TODOs before reading these.

Nothing in here runs on its own. `train.py` always loads `models/`, so
these files are reference only. To use one, copy the code across into the
matching file in `models/`: the body of `forward` for step 2, the three
assignments in `MultiHeadAttention` for step 3, the two lines of
`Block.forward` for step 4. The imports at the top of each `models/` file
already point at your own work, so a pasted step 3 uses your step 2 head,
not ours.

Steps 3 and 4 build on the step before, so an unfinished step blocks the
rest. If you're stuck on step 2, pasting its solution lets you carry on
writing steps 3 and 4 yourself.
