# Transit Router BGP 邻居中断处理手册

## 告警与现象

告警名 `TR_BGP_SESSION_DOWN`，由 Transit Router 上报。指标 `bgp_neighbor_state` 从 1（Established）跌为 0，`received_routes` 骤降为 0，跨地域业务流量中断。

期望态：BGP 邻居会话状态为 Established，`bgp_session_uptime` 持续增长，路由表中包含对端通告的动态路由。实际态：会话变为 Idle 或 Active，`bgp_session_uptime` 归零，路由撤销，流量黑洞。

## 常见根因与排查

根因一：Transit Router 路由条目数超过配额 `route_limit`（默认 10000 条），触发错误码 `QUOTA_ROUTE_EXCEEDED`，新路由无法学习。排查执行 `tr-quota describe --tr-id <TR_ID>` 查看 `route_limit` 使用率。

根因二：中间链路 MTU 不一致，BGP Update 报文分片丢弃，会话反复震荡。排查执行 `tr-diag check-bgp --tr-id <TR_ID>` 确认 `hold_timer` 与报文丢弃计数。

根因三：对端设备重启或维护，TCP 179 端口连接被重置，错误码 `BGP_HOLD_TIMER_EXPIRED`。

## 处置步骤与约束

处置步骤：若为配额问题，提交配额工单扩容 `route_limit`；若为 MTU 问题，将路径 MTU 统一为 1400 并关闭 PMTU 黑洞；若为对端维护，等待对端恢复后确认会话回到 Established。

处置约束：调整 `hold_timer` 必须对端同步修改，禁止单侧修改造成会话无法协商。变更生产 Transit Router 配置需走审批。
