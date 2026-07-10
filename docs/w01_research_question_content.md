# ML-02: Research Question and Provisional Lane

## 1. My lane and why

**My lane: Refresh / Content Opportunity Scoring**

I'm choosing Lane 2 because I already have direct evidence from the Week 1/2 starter
pipeline that this is a workable, well-supported problem on this data:

- The starter pipeline's Precision@50 for `is_declining_label` shows a clear gap
  between a hand-written rule (0.240) and a random forest (0.740) — meaning a
  learned ranking genuinely outperforms a fixed threshold rule on this task, not
  just marginally.
- In my own Week 1 discovery, I found that CTR varies meaningfully by content_type
  even within the same position tier (comparison articles consistently underperform
  other types across every tier) — suggesting there are real, discoverable content-level
  signals beyond position/volume alone that a refresh-scoring model could learn from.

This lane also has the most direct mentor-provided support (starter playground +
warehouse `dim_content` + `fact_content_daily_performance`), and its output — a
ranked review queue with reason codes — maps to something a real content team could
act on with limited weekly capacity. I'll confirm or revise this by end of Week 4.

## 2. The question: decision, action, cost of a wrong call

**Research question:** Which pages should a content reviewer look at first this
week, given they can only realistically review ~50 pages?

**Unit of analysis:** one row = one (client_hash_id, content_hash_id) pair —
a single piece of published content for a single client, evaluated over a 90-day window.

**Decision this informs:** given limited reviewer capacity, which content items
get reviewed first for refresh, expansion, protection, pruning, or monitoring.

**Action someone takes:** a content reviewer opens the ranked queue, reads the
reason codes attached to each row (e.g. `stale_visible_page`, `declining_with_demand`,
`low_ctr_visible_page`), and decides on one of the defined actions for that specific page.

**Cost of a wrong call:**
- **False positive** (flagged high-priority, but the page was actually fine):
  wastes limited reviewer hours that could have gone to a page that genuinely needed
  attention — a real opportunity cost, since review capacity is the actual bottleneck.
- **False negative** (a genuinely declining/high-opportunity page never surfaces):
  silent, compounding traffic loss for the client that isn't caught until it's much
  worse — harder to recover from than a wasted review.

**Why data/ML can help at all:** the starter pipeline already shows a hand-written
rule only gets ~12 of the top 50 reviewer picks right, while a random forest gets
~37 of 50 right (client-holdout validated) — meaning there's real, learnable signal
in these observable features beyond what a simple threshold rule captures. That gap
is the evidence this isn't a fixed-answer problem — a better model has room to earn
real precision improvement over the existing baseline.

## 3. Quick look at the data (code cell)

```python
import pandas as pd

df = pd.read_csv("data/raw/content_refresh_anonymized.csv")
df = df[(df["impressions_90d"] > 0) & (df["content_age_days"] >= 90)].drop_duplicates("content_id")

# Number 1: class balance of the label
label_rate = (df["trend_direction"] == "down").mean()
print(f"Share of pages labeled 'declining': {label_rate:.1%}  (n={len(df):,} pages)")

# Number 2: how much demand is actually at stake
visible_declining = df[(df["trend_direction"] == "down") & (df["impressions_90d"] >= 100)]
print(f"Declining pages with >=100 impressions/90d: {len(visible_declining):,} "
      f"({len(visible_declining) / len(df):.1%} of all pages)")

# Number 3: Precision@50 gap already established in the Week 1/2 starter pipeline
print("\nFrom outputs/model_report.md (client-holdout validated):")
print("  baseline rule   Precision@50: 0.240  (~12 of top 50 correct)")
print("  random forest   Precision@50: 0.740  (~37 of top 50 correct)")
```

**Interpretation:** These numbers show two things worth 7 weeks of work: (1) declining
pages are common enough (and enough of them have real traffic at stake) that a review
queue is worth building, and (2) there's a large, validated gap between a simple rule
and a learned model — meaning there's real headroom to improve on FlyRank's current
approach, not just noise.

## 4. Careful words: what I can and can't claim

**I can claim:**
- Which pages *historically* showed patterns associated with decline, low CTR, or
  staleness, based on observable signals in this dataset.
- That a learned ranking model beats a simple rule at *identifying* these patterns
  in past data (client-holdout validated).
- Which pages are worth prioritizing for human review first, given limited capacity.

**I cannot claim:**
- That refreshing a flagged page will *cause* it to recover — that requires an
  actual experiment (e.g. A/B test on refreshed vs non-refreshed pages), not this
  observational ranking.
- Anything about *why* Google ranks pages the way it does — I only have observable
  outcomes (impressions, clicks, position), not the algorithm itself.
- That "declining" always means something is wrong with the page — some drops are
  consolidation (a sibling page absorbing demand) or seasonality, not real decline,
  and I need minimum-volume and persistence checks to rule that out.
- That this result generalizes to the full 78.8M-row warehouse — this is evidence
  from a 30,000-row anonymized starter slice only, and needs to be re-earned with
  proper validation once I move to warehouse-scale data.

## 5. Self-check

- [x] Picked one of the four predefined lanes (Lane 2) with a clear reason grounded
      in Week 1/2 evidence, not just a guess.
- [x] Named the decision (what to review first), the action (reviewer picks one of
      refresh/expand/protect/prune/monitor), and the cost of a wrong call in both directions.
- [x] Showed 2-3 real numbers from the starter dataset (label rate, visible-declining
      count, Precision@50 gap).
- [x] Explained why this isn't just "train a model" — it's a decision-support ranking
      problem under a real capacity constraint, with reason codes a human reviewer
      can inspect and override.
- [x] Used careful, non-causal language throughout — no claims about Google's
      algorithm or that refresh causes recovery.
