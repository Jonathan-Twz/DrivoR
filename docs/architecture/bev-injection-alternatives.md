# DrivoR BEV Injection Alternatives

This note records independent BEV experiments and their Git branches. New
alternatives should remain sibling branches so benchmark comparisons do not
silently inherit another alternative's trainable path.

## Shared Baseline

- Branch: `dev-bev-score-decoder`
- Anchor commit: `a253460`
- Contains independently configurable scorer-side and decoder-side BEV
  cross-attention, BEV tokenization, checkpoint support, and evaluation tools.

## Decoder BEV Cross-Attention

- Branch: `dev-bev-score-decoder`
- Injection location: inside every trajectory decoder block.
- Flow: self-attention -> scene cross-attention -> BEV cross-attention -> MLP.
- Design: chained BEV injection with parallel low-rank side adapters.
- Canonical note: `docs/architecture/bev-trajectory-decoder.md`.

## Alternative 1: BEV Residual Proposal Refiner

- Branch: `dev-bev-residual-proposal-refiner`
- Injection location: after the original trajectory decoder and before scorer.
- Frozen path: encoder, original decoder, trajectory heads, and scorer.
- Trainable path: BEV tokenizer and residual proposal refiner.

```text
frozen base proposals
  -> frozen trajectory embedding as one query per proposal
  -> global cross-attention over BEV tokens
  -> delta trajectory head
  -> base + alpha * delta
  -> frozen scorer
```

`alpha` is one learned scalar initialized to zero. This gives exact baseline
proposals at initialization. At the first backward pass only `alpha` receives
a useful gradient; the refiner starts receiving gradients after `alpha` moves.

The scorer and losses are unchanged. Because scorer input remains detached,
the refiner is trained by the existing final-proposal trajectory/diversity loss,
while the frozen scorer is used for proposal selection.

## Reserved Alternatives

- `dev-bev-path-pooling`: sample or pool BEV features along each proposal's
  spatial path before residual prediction.
- `dev-bev-decoder-residual-refiner`: combine decoder-internal BEV attention
  with the post-decoder residual proposal refiner.
