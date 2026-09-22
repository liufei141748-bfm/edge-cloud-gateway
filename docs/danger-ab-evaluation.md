# 危险案例自动 A/B 评测

## 1. 一条命令起步

在项目目录执行：

```bash
.venv/bin/python -m edge_cloud_gateway.danger_eval --dry-run
```

首次只运行这一条。未传模式参数也默认 dry-run；即使传入 `--config` 指向真实配置，dry-run 也不读取它、不创建真实网络适配器。无需先启动 Gateway 服务、Ollama 或配置 Key。

每次生成独立目录 `evaluation-private/danger-ab/<UTC时间-随机编号>/`，包含：

- `details.json`：原始案例、A/B 最终回答、Working Context、来源片段、指标、判定及汇总。
- `details.csv`：72 行明细，每案例 A、B 各一行；UTF-8 BOM，便于表格软件打开。
- `summary.md`：中文总览、分类结果、失败/复核清单、丢失原文及启用建议。

可用 `--output-dir 路径` 更换报告父目录。报告包含人工编写的测试资料和回答，仅在本地保存；默认目录已被项目 `.gitignore` 忽略。报告不会自动上传。Ctrl-C 中断时会保存已经完成的分组，未完成分组保持未知，报告标明未完成；尚在进行的供应商调用可能已消耗 Token，但返回用量未知，不能当作零费用。

## 2. 如何使用已有项目

新增入口使用现有 `create_app`、HTTPX ASGI Transport 和 `Store(':memory:')`，在同进程内真实经过 `/v1/chat/completions`。它不复刻 Gateway，不占用已有服务端口，不改用户配置和服务数据库。

| 组别 | 明确开关 | 预定路径 |
| --- | --- | --- |
| A | `gateway_context.optimize=false` | 完整 Raw Context → Gateway → 配置的云端 provider |
| B | `gateway_context.optimize=true` | Raw Context → 原有规则过滤 + 配置的本地模型选择原文 ID → Working Context → 配置的云端 provider |

同一个案例的 A/B 请求仅优化开关不同。云模型、消息、资料、输出格式、temperature=0、max_tokens=512 均相同；逐例交替 AB / BA 顺序，关闭 Gateway 缓存。没有让模型决定路由。

为排除短案例被路由阈值跳过，评测使用隔离设置：

- context.enabled=true，min_input_tokens=1。
- worker_max_bytes=12000；local.num_ctx 至少 8192。
- 缓存关闭，数据库在内存中。
- 保留配置的最小净缩减阈值、日志窗口规则及其他现有保护。

这些值写入报告 metadata，只影响本次评测进程，不写回配置。数据中每例均含可筛选候选块；运行前验证预定路径。实际路由、local_model_used、local_attempt_count、route_reason 和 fallback_used 单独记录。B 因无净缩减返回全量时，原 Gateway 的 fallback_used 可能仍为 false，新增 raw_returned 可区分此情况；整体判定仍为 MANUAL_REVIEW。

## 3. 36 个案例与固定离线剧本

结构化数据在 `src/edge_cloud_gateway/data/danger_cases.json`。12 类各 3 个，共 36 个，当前有 90 条 must_keep：数字阈值、否定条件、单位、版本、跨文件、指代、长日志、多条件、优先级、代码与注释、相似实体、长文本隐藏条件。

每例包含 id、category、context、question、expected_answer、must_keep、manual_review_needed；指代案例可增加 conversation。context 使用已有 Gateway 的 blocks 格式。expected_answer 是完整 JSON 对象，问题明确约定字段/类型/固定值，资料不足时指定字段填 null。

must_keep 形如：

```json
{"block_id":"rule","text":"等于 30 时不触发"}
```

支持可选 start/end，用于锁定同块重复文字的某一次位置。标注仅供评测，不加入模型 prompt 或 constraints，也不据此将关键块强制设为不可删除。模型只能看到原资料与问题，不能看到标准答案和保留清单。

`danger_fixtures.json` 是独立固定剧本，保存每例 selected_ids、answer_a、answer_b。它不调用模型，也不会按 expected_answer 动态制造“答对”的输出。它特意包含 B 错误、A 错误、关键证据丢失及人工复核，验证判定与报告能够报告失败。

特别注意：日志案例 logs_02、logs_03 能暴露现有日志窗口规则丢失未含保护标记的租户或恢复状态；reference_01、similar_entities_03、hidden_02 演示本地选择器舍弃。报告分别列出删除原因，不能把日志规则问题都算成本地模型问题。本评测不调整生产筛选策略。

## 4. 质量判定

使用确定性规则，不调用第二个云模型评分：

1. 只有成功、完整结束的最终回答才可判定；请求失败或截断为 MANUAL_REVIEW。
2. 解析唯一完整 JSON 对象，按所有字段、值及嵌套类型严格比较。不接受重复键、多余字段或“含正确关键词但结论错误”。允许仅包裹 JSON 的代码围栏。
3. 对 must_keep 验证 block ID、source、kind、原文字符偏移和内容。只有原文中连续覆盖的片段能满足条件，不允许错误来源冒充，也不能跨删除间隙拼接。
4. B 答错或遗失关键条件，整体 FAIL，即使 Token 减少或模型猜对部分内容。
5. A 答错、快照未知、人工标记、B 没有实际本地筛选或回退全量，整体 MANUAL_REVIEW；确定的 B 错误优先 FAIL。
6. 只有 A/B 均正确、关键条件完整、实际路径符合预定且无需人工复核时才整体 PASS。

A/B 答案是否正确独立保存。只有 A 正确、B 错误且 B 确实遗失 must_keep，才标记 selection_omission_evidence；这是筛选遗漏证据，单次对照不能证明因果。关键材料完整而 B 答错时不归因于遗漏。

## 5. 指标与公式

A、B 都记录最终回答、cloud prompt_tokens、completion_tokens、reasoning_tokens、total_tokens，Raw/Working 输入估算、payload 字节数、压缩率、模型名称与实际使用标志、预定和实际路由、本地/云/总延迟、fallback。

| 指标 | 定义 |
| --- | --- |
| 云端输入 Token 降幅 | `1 - B.prompt_tokens / A.prompt_tokens`，正数表示减少 |
| 云端总 Token 变化 | `B.total_tokens - A.total_tokens`，正数表示增加 |
| 完整路径 Token 变化 | B/A 各自云总量加本地输入+输出，再相减；另存 end_to_end_total_tokens |
| 延迟变化 | `B.latency_total - A.latency_total`，毫秒，正数表示更慢 |
| 上下文压缩比 | `working_input_tokens / raw_input_tokens`，1 表示未压缩 |
| B 任务通过率 | B 答案 PASS 且无需人工复核的案例数 / 全部案例数 |
| 危险案例通过/失败率 | 整体 PASS / FAIL 数各自除以全部案例数；复核不算通过 |
| 关键约束保留率 | B 已保留约束条数 / 有可核验证据的约束条数；未知条数和案例数单列 |

reasoning_tokens 通常已经包含在 completion_tokens 中，不再叠加。total_tokens 取供应商返回值，缺失或与输入+输出不一致保持 null；reasoning 超过 completion 也保持 null。raw_input_tokens 和 working_input_tokens 沿用 `utf8_bytes_div4_v1` 估算，不能当作供应商账单 Token。缺失云 usage 不补零；A 未使用本地模型时本地 Token 为确定的零。

dry-run 根据实际经 Gateway 渲染的 payload 计算模拟输入量，usage_source=estimated，simulated=true。延迟是本机 fixture 执行耗时。live 云用量来自供应商响应，usage_source=actual 或 unknown。不同来源 Token 不混算；已知延迟独立比较同类运行，不因供应商没返回 usage 而丢弃。均值和 P50/P95 使用有效样本，报告同时列有效样本数，分位数采用排序后的线性插值。

云原生缓存、本地冷热启动和网络波动仍会影响 live 耗时；交错顺序只能减少顺序偏差，不能完全消除。每类仅 3 例，live 全过也只能建议扩大验证，不能自动建议生产启用。

## 6. 真实调用与付费确认

只有自行决定付费测试时使用：

```bash
.venv/bin/python -m edge_cloud_gateway.danger_eval --live
```

默认读取已有 `config.local.toml`，也可通过 `--config` 指定文件。要求 `mode=live`、`cloud.enabled=true`、`local.enabled=true`、云端为 HTTPS OpenAI-compatible endpoint，且指定的云端密钥环境变量已存在。本地可使用 Ollama 或 OpenAI-compatible provider。程序不安装或下载模型，不变更 Key。

在创建任何真实适配器前显示：

> 即将执行真实云 API 调用，可能产生费用。

显示 36 例 / 72 次云调用后，需要在交互终端输入 `LIVE`。取消或非交互运行不产生调用。这道确认同样在 Python 的 run_suite(live=True) 边界执行，不能从库入口绕开。无自动重试；无需第二个评分模型。费用不是固定值，72 次调用是次数上限，不是金额上限；必须自行确认额度和供应商价格。

密钥只用于已有适配器的请求头。结果中的已配置密钥值会脱敏，报告不保存配置、请求头或原始异常消息。普通错误只打印异常类别及检查提示。

## 7. 测试与现有兼容性

```bash
.venv/bin/python -m pytest -q
```

新增测试覆盖案例结构/分类数量、强制开关与路由、默认禁止网络构造、实际 72 次模拟云尝试+36 次本地尝试、严格答案和来源检查、三态判定、已知/未知 Token 与推理去重、完整会话、费用确认和密钥脱敏、中断保留、报告与分位计算。

发布前应重新运行完整 pytest；历史测试数量不能替代当前代码的最终回归。Danger dry-run 与真实 provider 评测必须分别报告，不能把 fixture 结果泛化到真实 workload。
