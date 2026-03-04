# Expert Parallelism

*From [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)*

## Expert Parallelism

This is the last parallelism method we're going to discuss. Before tackling it, if you don't have any exposure to Mixture of Experts (MoE) models, you might want to take some time to read about them in [this much shorter blog post](https://huggingface.co/blog/moe) we published some time ago, which should help you better understand the MoE architecture in general.

The Mixture of Experts paradigm has recently gained traction and visibility with models such as GPT-4, Mixtral[^ref], and DeepSeek-V3/R1. The basic idea is that instead of having a single feedforward module per layer, we can have several parallel modules and route tokens through them to be processed differently.

Illustration of an MoE layer taken from the Switch Transformers paper[^ref]

The design of MoE layers makes it easy to implement parallelism across the experts dimension, for what we call *expert parallelism (EP)*. Since the feedforward layers are fully independent, we can simply put each expert's feedforward layer on a different worker. Compared to TP, this approach is much more lightweight, since we don't need to split the matrix multiplication; we just need to route the hidden states of a token to the right expert.

In practice, EP is typically used in conjunction with other forms of parallelism, such as data parallelism. This is because EP only affects the MoE layers and doesn't shard the input tokens (unlike context parallelism, which shards tokens along the sequence length dimension). This means our GPUs would be doing redundant computation for all the non-MoE blocks if we only used EP. By combining EP with DP, we can efficiently shard both the experts and the input batches across our GPUs, as you can see in the simplified diagram below:

Source: "A Survey on Mixture of Experts"[^ref]

But let's not get ahead of ourselves - we'll talk about all the interactions between different parallelism strategies in the following section, so don't worry if you don't understand this last diagram yet.

In practice, there are a few tricks to make EP work efficiently, and they are closely tied to model design. For instance, DeepSeek-V3 enforces a constraint in the router, ensuring that each token is sent to at most $M$ nodes (in their case, 4) to keep the tokens on a single node and reduce communication overhead. While expert parallelism has been around for a while[^ref], it is just now gaining new traction with the MoE architecture gaining popularity.

We plan to add a more complete example of EP in Picotron/Nanotron soon, so stay tuned for more!
