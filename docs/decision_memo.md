# Decision Memo: Roll out the engagement policy, pilot a waiver for top spenders

**To:** VP of Product  **From:** Rishiraj Chouhan  **Date:** October 20, 2026
**Basis:** real transactions (UCI Online Retail II) with a **simulated** pricing and bundling policy. This memo demonstrates the decision workflow; the effects are not real.

## Recommendation

1. **Roll out the policy to the lower four spend quintiles (Q1 to Q4).** The effect there is clear and positive.
2. **Do not roll it out unchanged to the top spend quintile (Q5).** Run a randomized pilot of a fee waiver for Q5 accounts first.
3. **Judge the pilot on active share, not revenue.** Revenue is too noisy to detect the harm; active share is not.

## Evidence (scenario with hidden degradation, 27 weeks after launch)

| Group | Weekly spend per customer | Active share (last 90 days) |
|---|---|---|
| All treated customers | +$1.66 (95% CI -1.71 to +5.03), not significant | +2.5 pp (+0.7 to +4.4) |
| Q1 to Q4 | **+$3.05** (+1.65 to +4.46) | **+4.5 pp** (+2.4 to +6.6) |
| Q5, top spenders | -$3.26 (-17.27 to +10.75), unclear | **-3.7 pp** (-7.4 to -0.05), borderline |

The total looks mildly positive, but it blends a clear gain in Q1 to Q4 with a likely loss of active top customers. A revenue-only report would miss the second part.

## What the waiver would be worth

A 50% waiver for Q5 accounts, over 27 weeks at 40% gross margin, gives a central estimate of **+$6,955**, but the 95% range is **-$22,927 to +$36,838**. The data cannot say whether a waiver pays for itself. That uncertainty is the reason to pilot rather than decide now.

## Pilot design

- Randomize Q5 accounts: half receive the waiver, half do not.
- **Primary metric:** share of accounts with an order in the last 90 days. **Secondary:** weekly orders and weekly spend.
- Read out after 13 weeks at the earliest and ideally 27, since retention effects build up over time.
- Guardrails (configurable): amber if active share falls 0.5 pp, red at 2.0 pp. Stop the pilot if the worst case of the interval reaches red.
- Expectation: simulations with about 395 treated Q5 customers detect the retention drop about 70% of the time and the spend change about 14% of the time. A pilot that shows no spend change does not prove there is no harm.

## Risks and limits

- Results come from a simulated policy on one retailer's two years of data; real effects may differ.
- Projections beyond 27 weeks assume the effect persists.
- Gross margin (40%) and the churn level are assumptions.
- Random assignment is what makes the comparison valid. A rollout to self-selected customers would not support the same conclusions; the project's stress test shows such a design overstates the effect by about 195%.

## Next steps

1. Approve the Q1 to Q4 rollout with weekly monitoring of active share.
2. Approve the Q5 waiver pilot and its metrics.
3. Review pilot results at 13 weeks and decide on Q5 rollout.
