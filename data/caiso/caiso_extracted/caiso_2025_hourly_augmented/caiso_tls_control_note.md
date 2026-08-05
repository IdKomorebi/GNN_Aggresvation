# CAISO OASIS 命令行 TLS 说明

本机 Chrome 能正常从 OASIS 下载官方 ZIP，但命令行证书库对
`oasis.caiso.com` 报告 self-signed certificate in certificate chain。

为避免无证据地关闭验证，本次先对完全相同的官方 `PRC_AS` URL做双通道控制：

- 正常 Chrome C 端下载；
- 命令行关闭证书校验后的控制下载。

两份 ZIP 的字节数和 SHA-256 完全一致：`True`。详细路径、字节数和
哈希见 `source_lineage.csv`。批量下载只访问固定官方主机
`oasis.caiso.com`，并对每个 ZIP 和内部 CSV 做哈希及结构验证。

这项控制降低了本机证书链异常导致错误来源的风险，但不等价于恢复正常 TLS
证书验证；因此 `source_manifest.csv` 的 retrieval_mode 对此有显式标记。
