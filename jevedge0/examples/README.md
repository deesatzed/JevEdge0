# Example decision rows

`decisions.jsonl` holds six decision rows in the JEV-CPU row format
(`id`, `state`, `question`, `options`), usable directly with
`jevedge0 bench --input`.

`labels.jsonl` holds one expected answer per row.

## What these labels are, and are not

These labels are **reasoned expectations written while building this
project**, not a validated gold standard. They were not produced by clinical
review, adjudicated by multiple annotators, or drawn from operational outcome
data.

They exist so the benchmark's accuracy path can be exercised on real rows.
They are not sufficient evidence that any method is fit for operational use.

Before attaching operational meaning to accuracy or calibration figures in a
given domain, build a labeled set from that domain with real adjudication.
