# High-Level Overview

*From [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)*

## High-Level Overview

All the techniques we'll cover in this book tackle one or several of the following three key challenges, which we'll bump into repeatedly:

1. **Memory usage:** This is a hard limitation - if a training step doesn't fit in memory, training cannot proceed.
2. **Compute efficiency:** We want our hardware to spend most time computing, so we need to reduce time spent on data transfers or waiting for other GPUs to perform work.
3. **Communication overhead:** We want to minimize communication overhead, as it keeps GPUs idle. To achieve this, we will try to make the best use of intra-node (fast) and inter-node (slower) bandwidths and to overlap communication with compute as much as possible.

In many places, we'll see that we can trade one of these (computation, communication, memory) off against another (e.g., through recomputation or tensor parallelism). Finding the right balance is key to scaling training.

As this book covers a lot of ground, we've made a [cheatsheet](assets/images/ultra-cheatsheet.svg) to help you navigate it and get the general takeaways. Keep it close by as you navigate these stormy waters!

![Cheatsheet](assets/images/ultra-cheatsheet.svg)
