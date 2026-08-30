# ML-12 — Tell the Story

Lane 3: Structured Content Archetype Clustering — FlyRank ML Internship Capstone

## Demo outline (5 minutes)

1. **Question** (30s): Does an unsupervised content archetype help predict underperformance, beyond raw position and engagement signals?
2. **Method** (90s): Random Forest vs. a transparent position-benchmark baseline, on an identical client-grouped split (`GroupShuffleSplit`), with a deliberate sealed-month leak trap to prove the validation can actually fail.
3. **One chart**: permutation-importance bar chart — `total_impressions` (0.153) and `sessions_per_impression` (0.074) dominate; `archetype_cluster` sits at 0.0013.
4. **One measured result**: baseline AUC 0.963 is near-circular (`corr(baseline_score, avg_ctr) = -1.000`) and discarded; model AUC 0.927 is the trustworthy number, built from features independent of the target.
5. **One recommendation**: route items flagged `REVIEW_TITLE_META` (CTR below their own position tier's benchmark) to human review first — the highest-confidence, most reproducible signal in this study. Never auto-apply.

## Social cut

- LinkedIn post: [LinkedIn_Post.png](https://github.com/Diwash17/FlyRank-AI-Assentment-and-Capstone/blob/main/work/outputs/LinkedIn_Post.png)
- Full carousel (PDF): [social_post.pdf](https://github.com/Diwash17/FlyRank-AI-Assentment-and-Capstone/blob/main/work/outputs/social_post.pdf)

## Employer summary (3 sentences)

Built and validated a content-performance classifier on FlyRank's anonymized search warehouse, comparing a Random Forest against a transparent baseline on a client-grouped split with an explicit leakage-trap check. Found that the baseline's higher accuracy was a validation artifact (near-perfect correlation with its own target) and that the capstone lane's own technique — unsupervised archetype clustering — contributed effectively zero predictive value, reporting both as genuine findings rather than optimizing the write-up around a flattering number. Delivered a ranked, human-reviewed content-review queue and a fully reproducible, publicly deployed research paper with leakage checks, limitations, and a data-safety pass built in from the start.
