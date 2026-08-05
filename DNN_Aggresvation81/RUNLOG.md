# DNN_Aggresvation81 运行流水

> 与 `CHANGELOG.md` 并列；机器可读事件见 `outputs/events.jsonl`。

## 2026-07-24
- `03:11:50` ▶ **[BUILD] START**　构造同K的S1与两种S2
- `03:11:52` ✔ **[BUILD] DONE**　S1与79号完全一致；新增S2-any和S2-aligned　　实耗 1.7s　rows=79464　max_s1_diff=0.0
- `03:11:59` ▶ **[SINGLE] START**　同K固定预算全参数扫描
- `03:12:31` ✔ **[SINGLE] DONE**　完成K×预算×S2参数扫描　　实耗 31.7s　rows=2316
- `03:12:37` ▶ **[STAGED] START**　两层S1+S2端到端参数扫描
- `03:17:12` ✔ **[STAGED] DONE**　完成先微调、S1+S2宽进、继续微调精排的全扫描　　实耗 275.4s　rows=14040
- `03:18:20` ▶ **[STAGED] START**　两层S1+S2端到端参数扫描
- `03:35:04` ✔ **[STAGED] DONE**　完成先微调、S1+S2宽进、继续微调精排的全扫描　　实耗 1003.5s　rows=54720
- `03:36:28` ▶ **[PROTECT] START**　扫描S2救回候选的保护名额
- `03:37:06` ✔ **[PROTECT] DONE**　保护通道扫描完成　　实耗 38.2s　rows=3584
- `03:40:01` ▶ **[SUMMARY] START**　tune选协议；test只作一次最终验证
- `03:40:04` ✔ **[SUMMARY] DONE**　主方案由tune F1选中；完成test配对bootstrap　　实耗 2.7s　selected_first_k=10　selected_first_s2=any　selected_first_parameter=0.2　selected_protect=0.03
- `03:40:22` ▶ **[FIGURES] START**　生成参数与最终方案图
- `03:40:24` ✔ **[FIGURES] DONE**　完成4张中文图　　实耗 1.6s
- `03:41:00` ▶ **[FIGURES] START**　生成参数与最终方案图
- `03:41:01` ✔ **[FIGURES] DONE**　完成4张中文图　　实耗 1.4s
- `03:41:57` ▶ **[FIGURES] START**　生成参数与最终方案图
- `03:42:51` ▶ **[FIGURES] START**　生成参数与最终方案图
- `03:42:53` ✔ **[FIGURES] DONE**　完成4张中文图　　实耗 1.6s
- `03:44:11` ▶ **[SUMMARY] START**　tune选协议；test只作一次最终验证
- `03:44:14` ✔ **[SUMMARY] DONE**　主方案由tune F1选中；完成test配对bootstrap　　实耗 2.6s　selected_first_k=10　selected_first_s2=any　selected_first_parameter=0.2　selected_protect=0.03
- `03:44:16` ▶ **[FIGURES] START**　生成参数与最终方案图
- `03:44:18` ✔ **[FIGURES] DONE**　完成4张中文图　　实耗 1.6s
