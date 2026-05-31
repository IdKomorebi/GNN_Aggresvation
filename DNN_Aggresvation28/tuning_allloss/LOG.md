# All-Loss Tuning Log

## Objective

Primary objective: minimize `artifacts/summary.json: best_test_loss` across all
configured Confidential targets. Secondary checks: mean/median per-target R2
and whether gains come at the cost of collapsing a weak target.

## Starting Baseline

User-selected baseline:
`outputs/20260524_234906_870673_dnn13_compact_allloss`

| run | best_test_loss | mean R2 | median R2 | min R2 |
| --- | ---: | ---: | ---: | ---: |
| `20260524_234906_870673` | 0.312926 | 0.678897 | 0.780035 | 0.289413 |

Baseline distinguishing settings: Raw data, 12 Confidential fields
(`net_actual_interchange_mw` dropped), `top_k=8`,
`gross_actual_interchange_mw top_k=16`, `window_size=3`,
`attention_heads=4`, `attention_temperature=0.8`, batch size 128.

Weakest baseline targets:

| target | R2 |
| --- | ---: |
| `gross_actual_interchange_mw` | 0.289413 |
| `congestion_price_da` | 0.299593 |
| `da_as_total_mw_thirty_minutes_reserve` | 0.393728 |
| `marginal_loss_price_da` | 0.434285 |

## Wave 1

Purpose: search locally around the improvement from attention temperature
`1.0` to `0.8`, and test whether giving the weakest target more incoming
edges helps without disturbing the whole graph.

| candidate | GPU | change from baseline | status |
| --- | ---: | --- | --- |
| `w1_temp060` | 0 | `attention_temperature=0.6` | completed |
| `w1_temp070` | 1 | `attention_temperature=0.7` | completed |
| `w1_temp090` | 2 | `attention_temperature=0.9` | completed |
| `w1_gross_topk24` | 3 | `gross_actual_interchange_mw top_k=24` | completed |

The DNN probe is pinned to the baseline probe settings and can be reused
while GNN-only parameters change.

### Wave 1 Results

| candidate | best_test_loss | mean R2 | median R2 | min R2 | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `temp=0.8, gross_k=16` | **0.312926** | **0.678897** | 0.780035 | **0.289413** | remains best |
| `w1_temp060` | 0.330697 | 0.657980 | 0.767525 | 0.110949 | too sharp |
| `w1_temp070` | 0.318606 | 0.675419 | 0.776843 | 0.200814 | close, not better |
| `w1_temp090` | 0.324588 | 0.663418 | **0.781380** | 0.107876 | worse aggregate |
| `w1_gross_topk24` | 0.323615 | 0.670907 | 0.778910 | 0.258213 | denser gross edges not useful |

Baseline reached its best checkpoint at epoch 97 with train loss `0.163572`
and test loss `0.312926`; the gap motivates testing stronger regularization.

## Wave 2

Purpose: retain the baseline graph and temperature while reducing
generalization gap or slowing correlation-alpha adaptation.

| candidate | GPU | change from baseline | status |
| --- | ---: | --- | --- |
| `w2_dropout020` | 0 | `dropout=0.20` | completed |
| `w2_dropout025` | 1 | `dropout=0.25` | completed |
| `w2_weightdecay001` | 2 | `weight_decay=0.001` | completed |
| `w2_alphalr6` | 3 | `alpha_lr_multiplier=6` | completed |

### Wave 2 Results

| candidate | best_test_loss | mean R2 | median R2 | min R2 | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| previous baseline `dropout=0.15` | 0.312926 | 0.678897 | 0.780035 | **0.289413** | replaced |
| `w2_dropout020` | 0.321732 | 0.669432 | 0.780368 | 0.174926 | worse |
| `w2_dropout025` | **0.309865** | **0.690209** | **0.810146** | 0.264479 | new best |
| `w2_weightdecay001` | 0.320234 | 0.674283 | 0.760467 | 0.251685 | worse |
| `w2_alphalr6` | 0.328503 | 0.656640 | 0.760323 | 0.063398 | worse |

The new best reaches its selected checkpoint at epoch 108. Its weakest
target is now `congestion_price_da`; aggregate and central tendency improved,
while the minimum R2 is slightly below the prior baseline.

## Wave 3

Purpose: refine the improved dropout value and check one nearby
attention-temperature interaction.

| candidate | GPU | change from Wave 2 best | status |
| --- | ---: | --- | --- |
| `w3_dropout023` | 0 | `dropout=0.23` | completed |
| `w3_dropout027` | 1 | `dropout=0.27` | completed |
| `w3_dropout030` | 2 | `dropout=0.30` | completed |
| `w3_dropout025_temp075` | 3 | `attention_temperature=0.75` | completed |

### Wave 3 Results

| candidate | best_test_loss | mean R2 | median R2 | min R2 | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| Wave 2 best `dropout=0.25` | 0.309865 | **0.690209** | **0.810146** | **0.264479** | strong secondary choice |
| `w3_dropout023` | **0.308745** | 0.688169 | 0.799854 | 0.262006 | new loss best |
| `w3_dropout027` | 0.319478 | 0.671678 | 0.802905 | 0.146558 | worse |
| `w3_dropout030` | 0.313073 | 0.684072 | 0.796677 | 0.216052 | worse |
| `w3_dropout025_temp075` | 0.320480 | 0.671961 | 0.773035 | 0.189524 | temperature interaction worse |

Increasing the gross override from 16 to 24 changed actual gross incoming
edges from 19 to 26 and was harmful. Lowering only the override may not remove
threshold/symmetry-retained edges, so Wave 4 tests the global threshold.

## Wave 4

| candidate | GPU | change from current loss best | status |
| --- | ---: | --- | --- |
| `w4_dropout022` | 0 | `dropout=0.22` | completed |
| `w4_dropout024` | 1 | `dropout=0.24` | completed |
| `w4_threshold020` | 2 | `threshold=0.20` | completed |
| `w4_threshold030` | 3 | `threshold=0.30` | completed |

### Wave 4 Results

| candidate | best_test_loss | conclusion |
| --- | ---: | --- |
| current best `dropout=0.23, threshold=0.25` | **0.308745** | remains best |
| `w4_dropout022` | 0.324284 | worse |
| `w4_dropout024` | 0.321978 | worse |
| `w4_threshold020` | 0.316024 | denser graph worse |
| `w4_threshold030` | 0.323604 | sparser graph worse |

The narrow dropout bracket and both threshold directions failed to improve the
current loss best.

## Wave 5

Final high-value structural check around the current best.

| candidate | GPU | change from current best | status |
| --- | ---: | --- | --- |
| `w5_heads2` | 0 | `attention_heads=2` | completed |
| `w5_heads8` | 1 | `attention_heads=8` | completed |
| `w5_encoder_mlp` | 2 | `input_encoder=mlp` | completed |
| `w5_attention_dropout010` | 3 | `attention_dropout=0.10` | completed |

### Wave 5 Results

| candidate | best_test_loss | mean R2 | median R2 | min R2 | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| current best `heads=4, linear, attention_dropout=0.05` | **0.308745** | **0.688169** | **0.799854** | **0.262006** | remains loss best |
| `w5_heads2` | 0.324232 | 0.665153 | 0.739067 | 0.246307 | worse |
| `w5_heads8` | 0.323395 | 0.668017 | 0.764195 | 0.085892 | worse |
| `w5_encoder_mlp` | 0.313275 | 0.680692 | 0.787897 | 0.166356 | closer, not better |
| `w5_attention_dropout010` | 0.323657 | 0.666285 | 0.777043 | 0.149401 | worse |

## Final Selection

Primary-objective winner:
`outputs/20260525_010155_395592_tune_w3_dropout023_allloss`

| metric | initial user-selected run | selected tuned run | change |
| --- | ---: | ---: | ---: |
| `best_test_loss` | 0.312926 | **0.308745** | -0.004181 |
| mean R2 | 0.678897 | **0.688169** | +0.009272 |
| median R2 | 0.780035 | **0.799854** | +0.019819 |
| minimum R2 | **0.289413** | 0.262006 | -0.027407 |

Selected GNN settings:

```yaml
graph:
  top_k: 8
  threshold: 0.25
  target_top_k_overrides:
    gross_actual_interchange_mw: 16
model:
  dropout: 0.23
  attention_dropout: 0.05
  attention_heads: 4
  attention_temperature: 0.8
  input_encoder: linear
  window_size: 3
training:
  batch_size: 128
  lr: 0.0005
  alpha_lr_multiplier: 12.0
  weight_decay: 0.0005
```

Tradeoff: `w2_dropout025` has slightly better mean/median R2
(`0.690209` / `0.810146`) and minimum R2 (`0.264479`), but its primary
all-loss objective is worse (`0.309865`). Use it when the visual per-target
balance matters more than the minimized aggregate MSE.

This search compares settings on the current test split and seed; it is an
engineering tuning result, not an unbiased estimate of final generalization.
