# DNN_Aggresvation118 运行流水

- `2026-09-24 01:40` 冒烟测试（RTS：lin / head3 / polyS）通过
- `2026-09-24 01:40–02:16` GPU2 链（`scripts/chain118.sh g2`）：RTS、NEM、PJM 两组的全部变体；ft 在四个数据集上均因 TypeError 失败（见 WORKLOG）；CAISO 因真值未就绪跳过
- `2026-09-24 02:23–02:26` GPU1：ft 补跑（RTS、NEM、PJM 两组）
- `2026-09-24 02:31–02:48` GPU1：CAISO 全部变体 + ft
- `2026-09-24 02:29–` GPU1：timeE + seq 计时（与 CAISO 变体同卡，偏慢，待重测）
- `2026-09-24 02:50` ✔ **[ANALYZE] DONE**　`scripts/analyze118.py`（10 个目标）、`scripts/figs118.py`
- `2026-09-24 06:10–06:36` ✔ **[TIME] DONE**　GPU0 空闲时重测 timeE + seq（5 个数据集）；旧计时改名 *_contended.npz 保留
