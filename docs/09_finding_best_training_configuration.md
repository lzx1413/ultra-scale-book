# Finding the Best Training Configuration

*From [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)*

## Finding the Best Training Configuration

We’ve now covered all the parallelism techniques that are actually used to distribute and train larger models, as well as how and why they can be combined together. The general question remains, though: Which ones should we choose, and how do we decide on a specific combination?

We touched on this a little bit in the previous section, but let's now walk in more detail through a possible decision process, step by step (keeping in mind that you'll always have to run a few experiments to find the definitive optimal setup for your compute cluster given its various physical properties, network bandwidth, GPUs per node, memory per GPU, etc.).

### Step 1: Fitting a training step in memory

First, we need to figure out how we can fit a full model instance on our GPUs. There are two general cases:

**1. GPU-rich case 🤑** - when you have plenty of GPUs available:
        
For models under 10B parameters, you can use a single parallelism technique, e.g. tensor parallelism or ZeRO-3/DP with full recompute across 8 GPUs.
For models between 10B-100B parameters requiring more than 8 GPUs, you have several options:

Combining tensor parallelism (TP=8) with pipeline parallelism
Combining tensor parallelism (TP=8) with data parallelism (ZeRO-3)
Using only ZeRO-3 (i.e., pure data parallelism) 

At 512+ GPU scale, pure data parallelism/ZeRO-3 will start to becomes inefficient due to communication cost - it can be better to then combine DP with either tensor or pipeline parallelism.
At 1024+ GPU scale, a recommended setup may be tensor parallelism (TP=8) with data parallelism (ZeRO-2) and pipeline parallelism.

We focus on fitting a single instance for now - even though we may use DP for ZeRO to achieve this goal - we're only interested here in the model parameters memory savings that it provide when used with ZeRO-3.

*Special considerations:*

For very long sequences, you will probably want to add context parallelism across nodes.
For Mixture of Experts architectures, it will be advantageous to use expert parallelism across nodes.

**2. GPU-poor case 😭** - when you might be low on GPU resources:
        
You can enable full activation recomputation to trade some compute for memory (and train a bit more slowly).
You can increase gradient accumulation to process larger batches with limited memory.

Now that we have a first model instance training, we need to make sure we have the right batch size.

### Step 2: Achieving the target global batch size

Depending on where step 1 left us in terms of micro-batch size and DP, our current batch size might be too small or too big. It's now time to hit our target batch size.

To increase our current global batch size:

- We can scale up data parallelism or gradient accumulation steps.
- For long sequences, we can leverage context parallelism.

To decrease our current global batch size:

- We can reduce data parallelism in favor of other parallelization strategies.
- For long sequences, we can reduce context parallelism.

OK, now we have the model running in the general configuration we want in terms of model size and batch size - but are we training it the fastest way? The final step is to work on optimizing throughput.

### Step 3: Optimizing training throughput

We want to make sure the training is running as fast as possible so all our precious GPUs are well utilized at all times. As long as memory and communication aren't bottlenecks, we can try the following:

- Scale up tensor parallelism (using the fast intra-node bandwidth) until we reach a degree close to the node size, so that we can reduce other forms of parallelism.
- Increase data parallelism with ZeRO-3 while keeping the target batch size.
- When data parallelism communication starts to become a bottleneck, transition to using pipeline parallelism.
- Try scaling up different parallelisms one by one.
- Experiment with micro-batch sizes ($mbs$) to aim for an optimal balance between max global batch size, model size, compute, and communication.

### Benchmarking thousands of configurations

Now that we've covered the step-by-step, let's implement this search process in real life.

In the [Nanotron](https://github.com/huggingface/nanotron) repository, you'll find several scripts you can use to run all the experiments discussed previously and benchmark your own model and cluster.

We actually ran benchmarks ourselves on **several thousand distributed configurations**, covering every model size we've discussed here as well as a very large number of cluster configurations (namely, 1-64 nodes of 8xH100s) in order to produce the results we've covered up to now in this book.

> We want to take this opportunity to apologize to our coworkers for blocking most of the science cluster, and in turn forgive any threats that may have been whispered.

Now let's take a step back to gather and analyze the results of all our benchmarks and see if, beyond theory, we can actually discover using real-world data how various configurations fare against each other.

All the following benchmarks were conducted with a sequence length of 4,096 and a global batch size of 1M tokens. We gathered all the top configurations for each model and cluster size and plotted them in the following heatmaps:

Heatmap visualization showing the optimal training configurations across different model sizes and compute node counts (we have 8 GPUs per node). For each combination, the configuration details include data parallelism (DP), tensor parallelism (TP), pipeline parallelism (PP), gradient accumulation steps (GAS), micro-batch size (MBS), and ZeRO optimization stage. The color intensity indicates the model FLOPs utilization (MFU), with brighter colors representing higher efficiency.

From this high-level visualization, we can draw several important insights:

- First, as we increase the number of nodes (higher parallelism), we observe a decrease in efficiency. This effect is particularly pronounced for smaller models, which have a lower compute to model size ratio. While we might typically compensate for small model size by increasing the batch size, we're constrained by our global batch size limit of 1M.
- Second, larger models present a different challenge. As model size increases, memory requirements grow substantially. This creates two scenarios with fewer nodes: either the model doesn't fit at all, or it fits but runs inefficiently due to operating near the GPU memory limits (see for instance the 80B parameter model training on 4 nodes).
- Finally, our benchmarks show how performance heavily depends on implementation quality. When we first implemented both parallelism strategies, tensor parallelism outperformed pipeline parallelism. After optimizing our PP code, it became the faster option. Now that we're improving the communication overlap in our TP implementation, we expect it to regain the performance lead.

### Lessons learned on benchmarking

Our goal for this book was not only to discuss theory and implementations, but to provide actual data points as well. So, the plan was simple: let's run every possible distributed configuration for every model and a number of cluster sizes. Even after excluding impossible configurations, we still needed to run thousands of experiments.

On paper, this sounds easy enough: we can easily launch big arrays of jobs on our cluster. However, as soon as we launched the first batches of experiments, our troubles began:

- PyTorch processes would sometimes fail to clean up properly.
- The Slurm job manager would forcefully terminate jobs, leading to node failures.
- Simple benchmarks that should have taken minutes would stretch into hours.
- Some jobs would hang indefinitely.

Running all the experiments in a finite amount of time required additional engineering, and we ended up spending a significant amount of time on things like:

- Minimizing cluster restart times and optimizing idle time
- Analyzing detailed NCCL debug logs
- Understanding memory usage patterns and CUDA memory allocator behaviors
- Improving pipeline parallelism performance on multi-node setups

These challenges taught us valuable lessons about the complexities of distributed training infrastructure. What looks simple in theory often requires careful attention to many moving parts in practice.

Reproducing theoretical results in real life is challenging, especially given the limited availability of production training code. Through open source projects like [Nanotron](https://github.com/huggingface/nanotron) and [Picotron](https://github.com/huggingface/picotron), we hope we can help making distributed training techniques more accessible, as well as collaborating on simple and efficient codebases that help researchers and practitioners get the most out of their hardware resources.

This concludes our very deep dive into the distribution methods of 5D parallelism.

Taking a step back, our discussion so far has often relied on a critical assumption: that computation and communication can be efficiently overlapped on GPUs without any impact on the computation throughput. The reality is more nuanced. When using common communication primitives like NCCL send/recv, we face hidden contention between computation and communication resources as communication kernels will usually make use of the same GPU streaming multiprocessors (discussed in the following section) that are used for computation. This leads to decreased throughput when communication is overlapped with computation. To truly optimize our distributed training, we need to dive deeper into the GPU architecture itself.

> Additionally, the synchronization patterns when overlapping computation and communication may not always be optimal for our parallel strategies. You can find an example in [this blog post](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) by the PyTorch team.

Time to turn the lights off and activate CUDA mode!
