# DNN_Aggresvation69 Results

## Question

Separate two roles that were previously conflated: (1) dedicated DNN/GCN attacker upper bound, and (2) MLP/GNN universal masked oracle fidelity before and after equal-step fine-tuning. Evaluate both direct sensitivity and order-2/order-3 synergy.

## Plain-language result

- A dedicated GCN attacker is useful mainly for medium-sized fixed subsets (5-16 fields).
- For one universal random-mask estimator, MLP is closer to retraining than GNN at K=0 and after equal-step fine-tuning.
- GNN does not reveal cleaner order-2/order-3 synergy in this implementation; its global synergy ranking is consistently worse.
- Fine-tuning closes much of the amortization gap for both models, but MLP remains more accurate and substantially faster.

## Data and truth protocol

- Registry: 1437 subsets: {'d60_random': 50, 'd67_single_pair': 990, 'd68_triple': 397}.
- Same shuffled split and normalization as D60/D63/D67/D68; split seed 42.
- Full-population reference: dedicated DNN retrain. Best-of-structure reference is reported separately on all 50 D60 random subsets plus the 18-subset GCN audit sample.
- Fine-tuning comparison: same seed-0 universal checkpoint family, batches, Adam settings, and K grid.
- Missing truth files at analysis time: 0; missing estimate files: 0.

## Dedicated attacker upper bound

| cohort | n_subsets | dnn_mean_r2 | gcn_mean_r2 | gcn_minus_dnn | gcn_win_fraction | gcn_significant_wins | dnn_significant_wins |
|---|---|---|---|---|---|---|---|
| pair | 9 | 0.2195 | 0.1817 | -0.0378 | 0.1111 | 0 | 8 |
| triple_all | 9 | 0.4166 | 0.3876 | -0.0290 | 0.0000 | 0 | 5 |
| wide_1-4 | 10 | 0.2281 | 0.2167 | -0.0114 | 0.2000 | 1 | 4 |
| wide_5-8 | 10 | 0.5460 | 0.5695 | 0.0235 | 0.7000 | 6 | 1 |
| wide_9-16 | 10 | 0.7200 | 0.7330 | 0.0130 | 0.8000 | 4 | 0 |
| wide_17-32 | 10 | 0.8065 | 0.8162 | 0.0097 | 0.9000 | 1 | 0 |
| wide_33-44 | 10 | 0.8601 | 0.8612 | 0.0011 | 0.4000 | 0 | 0 |

## Universal oracle fidelity to retrain references

| reference | cohort | K | arch | mae_per_conf | bias_per_conf | subset_mean_spearman | median_per_conf_spearman | mean_time_per_subset |
|---|---|---|---|---|---|---|---|---|
| best_struct_audit | pair | 0 | gnn | 0.1591 | -0.1591 | 0.7311 | 0.7288 | 0.0000 |
| best_struct_audit | pair | 0 | mlp | 0.1360 | -0.1360 | 0.8117 | 0.7155 | 0.0000 |
| best_struct_audit | pair | 10 | gnn | 0.1067 | -0.1067 | 0.9791 | 0.8409 | 0.1777 |
| best_struct_audit | pair | 10 | mlp | 0.1104 | -0.1103 | 0.8619 | 0.8609 | 0.0380 |
| best_struct_audit | pair | 50 | gnn | 0.0652 | -0.0651 | 0.9958 | 0.8409 | 0.8165 |
| best_struct_audit | pair | 50 | mlp | 0.0512 | -0.0511 | 0.9958 | 0.9121 | 0.1203 |
| best_struct_audit | pair | 200 | gnn | 0.0608 | -0.0608 | 0.9958 | 0.8147 | 2.9978 |
| best_struct_audit | pair | 200 | mlp | 0.0243 | -0.0242 | 0.9958 | 0.9875 | 0.4200 |
| best_struct_audit | triple_all | 0 | gnn | 0.2257 | -0.2257 | 0.6167 | 0.6527 | 0.0000 |
| best_struct_audit | triple_all | 0 | mlp | 0.1798 | -0.1798 | 0.6000 | 0.7583 | 0.0000 |
| best_struct_audit | triple_all | 10 | gnn | 0.1839 | -0.1839 | 0.7333 | 0.7667 | 0.1536 |
| best_struct_audit | triple_all | 10 | mlp | 0.1538 | -0.1538 | 0.7000 | 0.7917 | 0.0219 |
| best_struct_audit | triple_all | 50 | gnn | 0.1204 | -0.1204 | 0.8833 | 0.7395 | 0.7701 |
| best_struct_audit | triple_all | 50 | mlp | 0.0929 | -0.0929 | 0.8833 | 0.8500 | 0.1008 |
| best_struct_audit | triple_all | 200 | gnn | 0.0879 | -0.0879 | 0.9667 | 0.8750 | 3.0912 |
| best_struct_audit | triple_all | 200 | mlp | 0.0516 | -0.0516 | 0.9833 | 0.9167 | 0.3906 |
| best_struct_audit | wide_all | 0 | gnn | 0.1537 | -0.1537 | 0.9792 | 0.9387 | 0.0000 |
| best_struct_audit | wide_all | 0 | mlp | 0.0940 | -0.0940 | 0.9882 | 0.9597 | 0.0000 |
| best_struct_audit | wide_all | 10 | gnn | 0.1404 | -0.1404 | 0.9786 | 0.9429 | 0.1577 |
| best_struct_audit | wide_all | 10 | mlp | 0.0774 | -0.0774 | 0.9871 | 0.9621 | 0.0213 |
| best_struct_audit | wide_all | 50 | gnn | 0.1109 | -0.1109 | 0.9818 | 0.9556 | 0.7924 |
| best_struct_audit | wide_all | 50 | mlp | 0.0514 | -0.0509 | 0.9924 | 0.9639 | 0.0998 |
| best_struct_audit | wide_all | 200 | gnn | 0.0958 | -0.0957 | 0.9826 | 0.9480 | 3.0755 |
| best_struct_audit | wide_all | 200 | mlp | 0.0350 | -0.0342 | 0.9927 | 0.9656 | 0.3921 |
| dnn_full | pair | 0 | gnn | 0.0813 | -0.0813 | 0.9623 | 0.8965 | 0.0000 |
| dnn_full | pair | 0 | mlp | 0.0495 | -0.0494 | 0.9726 | 0.9536 | 0.0000 |
| dnn_full | pair | 10 | gnn | 0.0577 | -0.0577 | 0.9806 | 0.9209 | 0.1651 |
| dnn_full | pair | 10 | mlp | 0.0364 | -0.0364 | 0.9800 | 0.9715 | 0.0253 |
| dnn_full | pair | 50 | gnn | 0.0383 | -0.0380 | 0.9864 | 0.9397 | 0.7934 |
| dnn_full | pair | 50 | mlp | 0.0220 | -0.0218 | 0.9899 | 0.9861 | 0.1052 |
| dnn_full | pair | 200 | gnn | 0.0350 | -0.0344 | 0.9884 | 0.9383 | 3.1038 |
| dnn_full | pair | 200 | mlp | 0.0150 | -0.0148 | 0.9964 | 0.9947 | 0.4019 |
| dnn_full | single | 0 | gnn | 0.0402 | -0.0402 | 0.9765 | 0.8581 | 0.0000 |
| dnn_full | single | 0 | mlp | 0.0254 | -0.0254 | 0.9727 | 0.9113 | 0.0000 |
| dnn_full | single | 10 | gnn | 0.0322 | -0.0321 | 0.9848 | 0.8480 | 0.1574 |
| dnn_full | single | 10 | mlp | 0.0194 | -0.0193 | 0.9793 | 0.9490 | 0.0216 |
| dnn_full | single | 50 | gnn | 0.0213 | -0.0212 | 0.9808 | 0.8819 | 0.7920 |
| dnn_full | single | 50 | mlp | 0.0114 | -0.0113 | 0.9798 | 0.9597 | 0.1011 |
| dnn_full | single | 200 | gnn | 0.0214 | -0.0211 | 0.9838 | 0.8799 | 3.0691 |
| dnn_full | single | 200 | mlp | 0.0077 | -0.0075 | 0.9907 | 0.9829 | 0.3931 |
| dnn_full | triple_all | 0 | gnn | 0.1381 | -0.1381 | 0.8457 | 0.8303 | 0.0000 |
| dnn_full | triple_all | 0 | mlp | 0.0912 | -0.0911 | 0.8716 | 0.8712 | 0.0000 |
| dnn_full | triple_all | 10 | gnn | 0.0942 | -0.0941 | 0.9018 | 0.9016 | 0.1597 |
| dnn_full | triple_all | 10 | mlp | 0.0672 | -0.0672 | 0.8932 | 0.9038 | 0.0216 |
| dnn_full | triple_all | 50 | gnn | 0.0578 | -0.0576 | 0.9621 | 0.9429 | 0.7968 |
| dnn_full | triple_all | 50 | mlp | 0.0391 | -0.0389 | 0.9623 | 0.9696 | 0.1005 |
| dnn_full | triple_all | 200 | gnn | 0.0436 | -0.0432 | 0.9728 | 0.9471 | 3.1089 |
| dnn_full | triple_all | 200 | mlp | 0.0213 | -0.0211 | 0.9928 | 0.9907 | 0.3942 |
| dnn_full | wide_all | 0 | gnn | 0.1362 | -0.1362 | 0.9751 | 0.9428 | 0.0000 |
| dnn_full | wide_all | 0 | mlp | 0.0767 | -0.0765 | 0.9878 | 0.9710 | 0.0000 |
| dnn_full | wide_all | 10 | gnn | 0.1229 | -0.1229 | 0.9752 | 0.9420 | 0.1577 |
| dnn_full | wide_all | 10 | mlp | 0.0604 | -0.0599 | 0.9883 | 0.9713 | 0.0213 |
| dnn_full | wide_all | 50 | gnn | 0.0934 | -0.0934 | 0.9807 | 0.9495 | 0.7924 |
| dnn_full | wide_all | 50 | mlp | 0.0352 | -0.0334 | 0.9927 | 0.9693 | 0.0998 |
| dnn_full | wide_all | 200 | gnn | 0.0784 | -0.0783 | 0.9853 | 0.9558 | 3.0755 |
| dnn_full | wide_all | 200 | mlp | 0.0194 | -0.0167 | 0.9938 | 0.9769 | 0.3921 |

## Synergy fidelity

| order | K | arch | spearman | mae | top20_overlap | precision_syn_gt_0.1 | recall_syn_gt_0.1 |
|---|---|---|---|---|---|---|---|
| 2 | 0 | gnn | 0.3213 | 0.0371 | 0.3500 | 0.8350 | 0.2729 |
| 2 | 0 | mlp | 0.6708 | 0.0218 | 0.3000 | 0.9642 | 0.4358 |
| 2 | 10 | gnn | 0.4502 | 0.0262 | 0.6000 | 0.8251 | 0.4887 |
| 2 | 10 | mlp | 0.8007 | 0.0158 | 0.4000 | 0.9481 | 0.5523 |
| 2 | 50 | gnn | 0.5196 | 0.0206 | 0.7000 | 0.9383 | 0.6235 |
| 2 | 50 | mlp | 0.8964 | 0.0106 | 0.8000 | 0.9573 | 0.6764 |
| 2 | 200 | gnn | 0.5309 | 0.0188 | 0.7500 | 0.8896 | 0.7562 |
| 2 | 200 | mlp | 0.9279 | 0.0080 | 0.8500 | 0.9652 | 0.7778 |
| 3 | 0 | gnn | 0.1418 | 0.0442 | 0.0000 | 0.4333 | 0.0390 |
| 3 | 0 | mlp | 0.4766 | 0.0282 | 0.0500 | 0.9867 | 0.2222 |
| 3 | 10 | gnn | 0.3327 | 0.0325 | 0.1000 | 0.9275 | 0.1922 |
| 3 | 10 | mlp | 0.6331 | 0.0207 | 0.1000 | 0.9867 | 0.4444 |
| 3 | 50 | gnn | 0.4608 | 0.0249 | 0.4500 | 0.9565 | 0.4625 |
| 3 | 50 | mlp | 0.8329 | 0.0140 | 0.3500 | 0.9867 | 0.6667 |
| 3 | 200 | gnn | 0.5410 | 0.0196 | 0.5000 | 0.9157 | 0.7177 |
| 3 | 200 | mlp | 0.9183 | 0.0085 | 0.7500 | 0.9931 | 0.8679 |

## K=0 three-seed robustness

| cohort | arch | mae_mean | mae_std | spearman_mean | spearman_std | n_seeds |
|---|---|---|---|---|---|---|
| pair | gnn | 0.0791 | 0.0036 | 0.9637 | 0.0028 | 3 |
| single | gnn | 0.0385 | 0.0027 | 0.9774 | 0.0015 | 3 |
| triple_all | gnn | 0.1358 | 0.0026 | 0.8498 | 0.0114 | 3 |
| wide_all | gnn | 0.1326 | 0.0037 | 0.9740 | 0.0012 | 3 |
| pair | mlp | 0.0489 | 0.0015 | 0.9733 | 0.0007 | 3 |
| single | mlp | 0.0254 | 0.0009 | 0.9735 | 0.0045 | 3 |
| triple_all | mlp | 0.0892 | 0.0022 | 0.8722 | 0.0019 | 3 |
| wide_all | mlp | 0.0750 | 0.0015 | 0.9882 | 0.0003 | 3 |

## Required caveats

- The full-population reference is one-seed dedicated-DNN retraining, not a mathematical supremum over all models.
- The separate best-of-structure audit is the per-target maximum of the tested DNN and GCN retrains only; it is still an empirical lower bound on an unrestricted attacker's optimum.
- D67/D68 dedicated truth uses one training seed; D60 noise-floor results should be used for significance interpretation.
- GCN is exhaustively available only for D60's 50 broad random subsets and a deterministic 18-subset pair/triple audit sample; full pair/triple synergy fidelity therefore uses DNN retrain truth.
- D68 `triple_top` was selected using an earlier MLP-based scan, so architecture comparison on that stratum has selection bias; the 200 random triples are the unbiased order-3 check.
- Direct sensitivity ranking and synergy ranking are separate claims because subtraction amplifies estimation error.

Figures: `outputs/overview.png`, `outputs/fidelity_by_size.png`.
