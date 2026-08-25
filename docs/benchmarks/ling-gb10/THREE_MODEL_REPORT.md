# Ling 3.0 vs Qwen 3.8 27B vs Ornith 1.5 35B

Generated: 2026-08-25T19:33:36.847632+00:00

Eight official medium-difficulty LiveCodeBench test6 tasks, one deterministic pass@1 generation per task.
All deployments use TP1, a 262,144-token server ceiling, BF16 KV, MTP, one sequence, and no prefix cache.

## Aggregate

| Model | Passed | Median TPS | Median TTFT ms | Accepted drafts | Length cutoffs |
|---|---:|---:|---:|---:|---:|
| ling-3.0-flash-nvfp4-mtp-k2 | 7/8 | 14.17 | 25966.8 | 6428 | 1 |
| ornith-1.5-35b-nvfp4-mtp-k2 | 4/8 | 107.66 | 225.6 | 3490 | 3 |
| qwen3.8-27b-nvfp4-mtp-k4 | 5/8 | 15.77 | 660.6 | 2717 | 1 |

## Per task

| Model | Task | Pass | Tokens | TPS | TTFT ms | Accepted drafts | Finish |
|---|---|:---:|---:|---:|---:|---:|---|
| ling-3.0-flash-nvfp4-mtp-k2 | abc387_c | no | 3202 | 16.89 | 18236.5 | 1274 | stop |
| ling-3.0-flash-nvfp4-mtp-k2 | abc389_d | yes | 8192 | 17.75 | 16990.2 | 3716 | length |
| ling-3.0-flash-nvfp4-mtp-k2 | abc390_d | yes | 1568 | 15.33 | 35770.0 | 498 | stop |
| ling-3.0-flash-nvfp4-mtp-k2 | abc390_c | yes | 1151 | 14.40 | 31585.6 | 294 | stop |
| ling-3.0-flash-nvfp4-mtp-k2 | abc394_d | yes | 582 | 12.84 | 26901.1 | 75 | stop |
| ling-3.0-flash-nvfp4-mtp-k2 | abc396_d | yes | 1055 | 13.95 | 40750.1 | 233 | stop |
| ling-3.0-flash-nvfp4-mtp-k2 | abc397_c | yes | 881 | 13.86 | 25032.4 | 187 | stop |
| ling-3.0-flash-nvfp4-mtp-k2 | abc398_c | yes | 817 | 13.52 | 21011.7 | 151 | stop |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc387_c | no | 1123 | 106.32 | 194.1 | 271 | length |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc389_d | no | 2714 | 111.03 | 177.5 | 1004 | length |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc390_d | no | 2012 | 108.64 | 271.2 | 653 | length |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc390_c | yes | 1538 | 109.63 | 250.5 | 465 | stop |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc394_d | no | 1564 | 106.87 | 227.3 | 461 | stop |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc396_d | yes | 915 | 103.79 | 299.6 | 202 | stop |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc397_c | yes | 1099 | 108.46 | 223.9 | 290 | stop |
| ornith-1.5-35b-nvfp4-mtp-k2 | abc398_c | yes | 793 | 103.06 | 197.3 | 144 | stop |
| qwen3.8-27b-nvfp4-mtp-k4 | abc387_c | no | 1417 | 18.61 | 493.6 | 687 | length |
| qwen3.8-27b-nvfp4-mtp-k4 | abc389_d | no | 1236 | 18.64 | 482.3 | 497 | stop |
| qwen3.8-27b-nvfp4-mtp-k4 | abc390_d | no | 1714 | 19.40 | 883.2 | 793 | stop |
| qwen3.8-27b-nvfp4-mtp-k4 | abc390_c | yes | 753 | 15.58 | 776.2 | 168 | stop |
| qwen3.8-27b-nvfp4-mtp-k4 | abc394_d | yes | 661 | 13.13 | 667.3 | 92 | stop |
| qwen3.8-27b-nvfp4-mtp-k4 | abc396_d | yes | 790 | 15.60 | 1007.5 | 186 | stop |
| qwen3.8-27b-nvfp4-mtp-k4 | abc397_c | yes | 765 | 15.94 | 654.0 | 180 | stop |
| qwen3.8-27b-nvfp4-mtp-k4 | abc398_c | yes | 588 | 15.31 | 534.6 | 114 | stop |
