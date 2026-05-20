# DNN_Aggresvation2

Fixed-edge risk aggregation project:

1. Pretrain correlation metric weights with a correlation-gated DNN for each confidential target.
2. Average target-specific metric weights into one global metric weight vector.
3. Build a fixed weighted graph from the global weights.
4. Train only GNN model parameters on the fixed graph.

The default config uses the 2025 V2 processed PJM dataset and now uses
dual-ended General anchors: high-risk anchors plus probe-confirmed low-risk anchors.

## Run

From the repository root:

```bash
conda activate Pytorch310_MacBookAir
python DNN_Aggresvation2/scripts/run_fixed_edge_pipeline.py \
  --config DNN_Aggresvation2/configs/data2025_v2_fixed_edge.yaml
```

If correlation weights have already been pretrained, reuse them and rerun only graph/anchor/GNN stages:

```bash
python DNN_Aggresvation2/scripts/run_gnn_from_pretrained.py \
  --config DNN_Aggresvation2/configs/data2025_v2_fixed_edge.yaml \
  --source-run DNN_Aggresvation2/outputs/run_20260520_173235
```

Outputs are written to:

```text
DNN_Aggresvation2/outputs/run_YYYYMMDD_HHMMSS/
```

Important outputs:

- `pretrain/correlation_weights.json`
- `reuse_source_run.json` when running from pretrained artifacts
- `graph/edge_weights.csv`
- `labels/general_anchor_labels.csv`
- `gnn/node_scores.csv`
- `gnn/metrics.json`

## Main Config Choices

- `high_anchor_count`: number of high-risk General fields whose inference labels are measured directly.
- `low_anchor_count`: number of low graph-risk General candidates used as low-risk calibration anchors after probe confirmation.
- `low_anchor_max_graph_score` / `low_anchor_max_probe_q`: thresholds for accepting low-risk anchors into GNN supervision.
- `label_mode: sensitivity`: confidential nodes are labeled as 1; general anchor labels use measured inference ability.
- `include_initial_sensitivity: true`: node features include the manually defined initial sensitivity flag.
- `graph.learn_edge_weights: false`: edge weights are fixed after pretraining.
