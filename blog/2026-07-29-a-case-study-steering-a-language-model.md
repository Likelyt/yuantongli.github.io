---
title: A case study - Steering a Language Model: Activation Vectors, SAE Features, and LoRA style
date: 2026-07-29
---

# A case study - Steering a Language Model: Activation Vectors, SAE Features, and LoRA style

The most useful interventions are the ones we can explain. In recommender-system fairness, this means closing a performance gap between cohorts—say, men and women—without hiding the change inside an opaque reweighting or post-processing step. In language models, it means steering a trained model toward a desired behavior without retraining billions of weights into a configuration no one can inspect. The two domains share a common question: what is the least destructive way to turn a specific behavioral dial, and how do we know what is actually being turned?

The interpretability literature around language models has proposed several answers. One can add a vector to activations, discovered from contrastive examples. One can clamp a sparse-autoencoder feature to a higher value, treating the model as a dictionary of labeled concepts. Or one can learn a low-rank subspace edit, training the intervention directly on the target behavior. The methods look different—[activation vectors](https://arxiv.org/abs/2308.10248), [sparse-autoencoder features](https://transformer-circuits.pub/2023/monosemantic-features/index.html), [learned subspace edits](https://arxiv.org/pdf/2404.03592)—but they all promise the same thing: a small, reversible intervention that changes behavior in a controlled way.

That promise is hard to evaluate without a common metric. What matters is not whether a method can steer, but what it costs. Any intervention that pushes a model toward a target behavior also pushes it away from its training distribution. The question is how much.

To make that cost concrete, I compared the three methods on a single behavior in GPT-2 small, measuring both steering efficacy and distribution damage on the same held-out prompts. The goal was not to crown a winner. It was to see where each method lives on the same Pareto frontier.

What I found was a reminder that the field is still in the early stages of learning how to control these systems. The two least interpretable-looking methods turned out to be Pareto-equivalent. The method that felt most principled, ReFT, was not better on this task—it was worse.

## Contents

- [The steering problem](#the-steering-problem)
- [Three ways to turn the dial](#three-ways-to-turn-the-dial)
- [The frontier](#the-frontier)
- [What is the SAE buying you here?](#what-is-the-sae-buying-you-here)
- [Why ReFT underperformed](#why-reft-underperformed)
- [The failures that matter](#the-failures-that-matter)
- [What I would do next](#what-i-would-do-next)
- [A small lesson](#a-small-lesson)
- [Reproducibility](#reproducibility)
- [References and further reading](#references-and-further-reading)
- [Contrast pairs](#contrast-pairs)

## The steering problem

Suppose you want to make a language model produce more positive sentiment when it continues a neutral review sentence. There are many ways to do this. You can fine-tune the weights, but that changes the model everywhere and is expensive. You can prompt it, but prompts are brittle and affect context length. Or you can intervene directly on the internal activations.

Activation interventions are appealing because they are small, reversible, and local. But "small" does not mean "harmless." Any intervention that pushes the model toward a target behavior also pushes it away from its training distribution. The question is how much.

I measured two things on held-out neutral prompts:

- **Efficacy:** did the model shift toward the desired continuation? I used `logP(" good") - logP(" bad")` at the final position.
- **Distribution damage:** how much did the next-token distribution change overall? I used `KL(p_steered || p_base)` averaged over the next-token distribution.

The ideal intervention sits in the top-left of this plot: high efficacy, low damage. The bad ones sit in the bottom-right: high damage, low steering. Everything else lies on a frontier.

## Three ways to turn the dial

I compared three families of activation-level interventions at the same layer of GPT-2 small:

**Activation addition** is the simplest. You collect a set of positive and negative continuation prompts, take the mean difference of residual activations between the two classes, and add that vector to every position at test time, scaled by a coefficient `α`. It is a blunt instrument: a single direction in activation space, applied everywhere. But it is cheap and requires no training. ([Turner et al. 2023 / "Activation Addition: Steering Language Models Without Optimization"](https://arxiv.org/abs/2308.10248))

Not every layer is equally steerable. The mean-diff steering vector grows with depth: its L2 norm was 2.0 at layer 6, 6.3 at layer 8, and 13.0 at layer 10. That deeper-layer amplification shows up in the sweep. At `α = 2.0`:

| Layer | Efficacy | KL damage |
|---|---|---|
| 6 | +1.54 | 0.0035 |
| 8 | +3.14 | 0.0818 |
| 10 | +4.39 | 0.1412 |

Layer 10 gives the strongest sentiment signal for a still-reasonable damage level, so I used it as the fixed layer for all three methods. The comparison is not between "best layer for each method"—it is between the three methods at the same depth where the behavior is most controllable.

**SAE feature clamping** is the interpretable one. I used a public sparse autoencoder trained on GPT-2 small residual activations (`gpt2-small-res-jb` via [SAELens](https://github.com/jbloomAus/SAELens)). For the sentiment behavior, I identified feature 15438 by comparing mean feature activations on the positive and negative prompt pairs, after checking the feature's dashboard on [Neuronpedia](https://www.neuronpedia.org/). Then I clamped that feature to `α` times its maximum observed activation. The hope is that an SAE gives you a meaningful, human-interpretable knob: instead of pushing a random direction, you are turning a feature that *means* positive sentiment. ([Bricken et al. 2023 / "Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html); [Templeton et al. 2024 / "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html))

**LoReFT (low-rank subspace intervention)** is the learned one. I trained a small orthogonal projection and edited linear map on the same contrastive pairs using [pyvene](https://github.com/stanfordnlp/pyvene). ReFT does not assume a single feature or a single direction. It learns a low-rank subspace that maps base activations to a target behavior. In principle, that should make it more flexible and more generalizable across phrasings than a fixed clamp. ([Wu et al. 2024 / ReFT](https://arxiv.org/pdf/2404.03592))

## The frontier

I swept each method across strengths and plotted efficacy against KL damage. The result was not what my prior intuition predicted.

Activation addition and SAE feature clamping landed almost exactly on top of each other. At comparable strength, they achieved the same efficacy and the same low damage. For example, at `α = 2.0`:

| Method | Efficacy | KL damage |
|---|---|---|
| Activation addition, `α = 2.0` | +4.39 | 0.141 |
| SAE feature clamp, feature 15438, `α = 2.0` | +4.39 | 0.141 |
| LoReFT, rank 8 | +4.41 | 1.180 |

The first two are not just close. They are nearly identical on both metrics. The SAE clamp gives me a feature I can name and reason about, but it does not buy a better trade-off than the crude mean-diff vector.

ReFT, meanwhile, reached similar raw efficacy but at roughly eight times the KL damage. Lower-rank configurations (rank 4 and rank 2) performed worse across random seeds. So on this small-data sentiment task, the learned subspace method was not the most efficient—it was the most damaging.

![Activation-addition sweep](/fig/llm_model_steering_layers_analysis.png)
*Efficacy versus steering strength for activation addition. The method works, and it works monotonically, but the high-strength end is where damage accumulates.*

![Three-arm steering comparison](/fig/llm_model_steering_pareto_frontier.png)
*The Pareto frontier across the three methods. Activation addition and SAE feature clamping are essentially on the same curve. LoReFT sits to the right: similar efficacy, much higher damage.*

## What is the SAE buying you here?

This is the part of the result that makes me pause. One of the selling points of SAEs is that they give us *interpretable* levers. Feature 15438 is not just a direction; it is a sparse, identified unit whose decoder logits can be inspected. That interpretability is real and valuable if you are trying to understand why a model behaves a certain way.

But interpretability is not the same as control. On this task, the SAE clamp did not give me a better controller than the mean-diff vector. The feature I selected was genuinely causal—clamping it did steer sentiment—but it was not *more efficient* than the black-box direction. The SAE told me *which* knob to turn, but it did not turn it more cleanly than a steering vector discovered by contrastive averaging.

That distinction matters. If we want SAEs to guide safe interventions, we will need evidence that they improve the control frontier, not just that they provide a story about what is being changed.

## Why ReFT underperformed

There are several plausible explanations, and the honest answer is that this is a small pilot with several caveats.

First, ReFT was trained on only fourteen prompts (positive and neutral, all labeled toward the positive continuation). Low-rank subspace methods are data-hungry relative to feature clamps or contrastive vectors. With so few examples, ReFT may have memorized the training phrasing rather than learning a general sentiment subspace. That would explain the high variance across seeds and the higher distribution damage on held-out prompts.

Second, the ReFT intervention was applied only at the last token position, whereas activation addition and the SAE clamp were applied at all positions. That is a design choice, not a fundamental limitation of the method, but it makes the comparison less symmetric. A multi-position ReFT might behave differently.

Third, the behavior is narrow: a single next-token valence contrast. ReFT might shine on more complex, multi-token behaviors where a learned subspace is genuinely needed to generalize across phrasings. This experiment does not rule that out. It only says that for the simplest possible steering task, ReFT is not the best tool among the three.

## The failures that matter

Most of this project's value, in my view, is in the things that did not work cleanly.

Feature selection was harder than the final result suggests. I started by looking at the top features correlated with the positive/negative contrast. Most of them did not steer. Correlation is not causation, and this is the central lesson of mechanistic interpretability. I kept track of how many top-correlated features had a real causal effect, and the hit rate was low. Feature 15438 happened to work, but it was not the obvious choice from correlation alone.

The top five features by mean activation difference were:

| Feature idx | mean_pos | mean_neg | diff |
|---|---:|---:|---:|
| 8553 | 2.572 | 0.758 | +1.814 |
| 3109 | 1.656 | 0.638 | +1.019 |
| 15438 | 0.912 | 0.000 | +0.912 |
| 16917 | 0.779 | 0.000 | +0.779 |
| 19213 | 1.046 | 0.269 | +0.777 |

Feature 15438 ranked third by this contrastive score. It was the one that happened to be causally steerable, while the higher-ranked features did not reliably produce the sentiment shift. That is why I think the most valuable habit in this work is to keep a "causal hit rate" column next to every correlation score.

ReFT also overfit. On some seeds the training efficacy looked excellent while the held-out damage looked bad. That is a familiar pattern from production work: a method can look good on the metric you optimized and bad on the distribution you actually care about. The KL damage metric kept that honest.

I also noticed the usual tokenizer artifacts. At high steering strengths, the model's behavior becomes strange not because the concept is broken, but because the probability mass is being pushed onto specific high-valence tokens. That is why I reported KL alongside perplexity and inspected generations by eye.

## What I would do next

If I had ten times the compute, I would run this comparison on Gemma-2-2B with [Gemma Scope](https://arxiv.org/abs/2408.05147) SAEs at different widths. The GPT-2 SAE I used is a useful starting point, but modern SAEs may have finer-grained feature splitting. One of the key questions is whether feature granularity changes the control frontier. If SAEs become much better at low-damage steering as the dictionary becomes larger, that would be a real argument for using them as control interfaces.

I would also test cross-behavior transfer. Does a method that works well for sentiment also work well for language switching or factual completion? A method that dominates on one narrow task but fails on others is not a general control mechanism. That question is more important than the absolute ranking on any single task.

Finally, I would revisit ReFT with a larger training budget and a paraphrased evaluation set. Its promise is generalization across phrasing. I would also keep an eye on newer dictionary-learning methods such as [cross-layer transcoders (CLTs)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), which may change the feature-granularity story again. If ReFT still loses to activation addition and SAE clamps on that test, that would be a strong result. If it wins, that would clarify where learned subspace methods belong in the toolkit.

## A small lesson

There is a temptation in interpretability to treat interpretability and control as the same problem. If we can name a feature, we assume we can control it cleanly. If we can learn a subspace, we assume it generalizes better than a hand-crafted direction. This experiment suggests that those assumptions deserve independent testing.

On a toy sentiment task in GPT-2 small, the two methods that look least like principled control mechanisms—a mean-diff vector and a single clamped SAE feature—were the most efficient. The learned method was not. That may change at larger scales, with richer features, or with more training data. But the point of a small controlled experiment is to surface the assumptions before scaling up, not to prove them.

The broader question remains: when we want to steer a powerful model safely, what is the least destructive intervention? I do not know the answer yet. But I think the right way to find it is to keep these three approaches on the same plot, measured by the same damage metric, and let the frontier tell the story.

## Reproducibility

The experiments were run on GPT-2 small with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [SAELens](https://github.com/jbloomAus/SAELens), and [pyvene](https://github.com/stanfordnlp/pyvene). I used layer 10 residual activations, fixed seeds, and saved all training configs. The exercises in [ARENA Chapter 1](https://arena-ch1-transformers.streamlit.app/) were a useful warmup before the implementation. The metrics, figure paths, and the full notebook skeleton are in the project notes. If you want to replicate or extend the comparison, the methodology is deliberately small enough to run on a single GPU in a day or two.

---

*This post is a reflection on a small independent interpretability project. The results are preliminary, and the failures are at least as informative as the successes. If you are working on similar comparisons, I would love to hear what you are finding.*

## References and further reading

1. Anthropic, *So You Want to Work in Mechanistic Interpretability?* (April 2025 update). https://transformer-circuits.pub/2025/april-update/index.html
2. Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." *Transformer Circuits* (2023). https://transformer-circuits.pub/2023/monosemantic-features/index.html
3. Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2023). *Activation Addition: Steering Language Models Without Optimization* (later retitled "Steering Language Models With Activation Engineering"). https://arxiv.org/abs/2308.10248
4. Templeton et al., "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Transformer Circuits* (2024). https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
5. Wu et al., "ReFT: Representation Finetuning for Language Models." https://arxiv.org/pdf/2404.03592
6. Stanford NLP, *pyvene* library. https://github.com/stanfordnlp/pyvene
7. Elhage et al., "Toy Models of Superposition." *Transformer Circuits* (2022). https://transformer-circuits.pub/2022/toy_model/index.html
8. ARENA Chapter 1, *TransformerLens & SAE tutorials*. https://arena-ch1-transformers.streamlit.app/
9. Nanda, N. et al., *TransformerLens*. https://github.com/TransformerLensOrg/TransformerLens
10. Bloom, *SAELens*. https://github.com/jbloomAus/SAELens
11. *Neuronpedia* feature dashboards. https://www.neuronpedia.org/
12. Lieberum et al., "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." https://arxiv.org/abs/2408.05147
13. Mechanistic Interpretability Discord. https://discord.com/invite/7ZzQZzZuVY
14. Anthropic, "Attribution Graphs: Methods and Tools." *Transformer Circuits* (2025). https://transformer-circuits.pub/2025/attribution-graphs/methods.html
15. Olshausen & Field, "Sparse coding with an overcomplete basis set: A strategy employed by V1?" *Vision Research* (1997); and the earlier 1996 technical report that introduced sparse coding as a principle.
16. Spielman, Wang & Wright, "Exact Recovery of Sparsely-Used Dictionaries." *COLT* (2012).

## Contrast pairs

The behavior was trained and evaluated with paired sentence stems that share topic and grammar but differ only in sentiment. These are the exact prompts used to build the mean-diff steering vector, to rank SAE features, and to train the ReFT intervention.

### Positive prompts (used for contrastive mean and ReFT labels)

- I absolutely loved this movie, it was
- This restaurant is wonderful, the food is
- What a fantastic book, truly
- The concert last night was amazing and
- I'm so happy with this purchase, it's
- The new policy is a great success, it
- She gave a brilliant performance, it was
- Our vacation was perfect, every day was
- The team did an excellent job, the result is
- This is the best phone I've owned, it

### Negative prompts (contrastive counterpart)

- I absolutely hated this movie, it was
- This restaurant is terrible, the food is
- What a boring book, truly
- The concert last night was awful and
- I'm so disappointed with this purchase, it's
- The new policy is a total failure, it
- She gave a dreadful performance, it was
- Our vacation was miserable, every day was
- The team did a sloppy job, the result is
- This is the worst phone I've owned, it

### Neutral evaluation prompts (held-out, for KL damage and efficacy)

- The film we watched yesterday was
- The meal at the new place downtown was
- The presentation this morning was
- The weather during our trip was
- The software update they released is
- The apartment we looked at was
- The lecture on Tuesday was
- The service at the hotel was

The positive/negative pairs were designed so that the mean-diff direction would isolate sentiment rather than topic, phrasing, or entity. The neutral prompts were chosen to be emotionally ambiguous continuations where the next token could plausibly go either way.
