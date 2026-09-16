# Day 1 复习：模型基础、完整问答与易错点

先掌握 1～7 题的模型与形状，再复习核心模块，最后检查工程与测试。每题先口述，再核对答案；面试题建议能画出形状。
代码：[config](../../kvforge/config.py)、[layers](../../kvforge/layers.py)、[attention](../../kvforge/attention.py)、[model](../../kvforge/model.py)、[tests](../../tests/test_day1.py)。以下讨论完整前向，增量缓存见 Day 3。

## 1. 【必会】Decoder-only 语言模型在做什么？

答：根据当前位置和之前的 token，预测下一个 token。输入 [a,b,c]，三个位置分别学习预测 b、c、d。训练能并行计算各位置，但因果掩码禁止读取未来；生成需要选出新 token 后才能把它作为下一步输入。

易错：预测下一 token 不等于只参考前一个 token；并行训练不等于可以看未来。

## 2. 【必会】config 与可训练参数有什么区别？各维度表示什么？

答：ModelConfig 保存构建模型的超参数，不是优化器更新的权重。vocab_size=V 是词表大小，dim=C 是表示宽度，n_layers=L 是层数，n_heads=Hq 是 Q 头数，n_kv_heads=Hkv 是 KV 头数，head_dim=D=C/Hq，hidden_dim 是 SwiGLU 中间宽度，max_seq_len 是当前实现的长度/位置范围，attention_backend 选择 naive/sdpa。

易错：dim 不等于 hidden_dim；字段名要统一，不能把 n_layers 写成 num_layers。

## 3. 【必会】为什么要求 C 能被 Hq 整除、Hq 能被 Hkv 整除、D 为偶数？

答：均匀拆头需要 C/Hq 为整数；分组共享 KV 需要 Hq/Hkv 为整数；RoPE 将通道两两配对，需要偶数 D。C=256、Hq=8 得到 D=32，Hkv=2 时每个 KV 头服务 4 个 Q 头。层数等规模参数还要为正，后端名称必须合法。

易错：这些条件直接决定张量能否正确计算，不只是笼统地“影响训练”。

## 4. 【必会】ID 到 logits 的形状怎样变化？Embedding 与 Linear 做什么？

答：ID [B,T] → Embedding [B,T,C] → 多层 Block [B,T,C] → 最后 RMSNorm [B,T,C] → lm_head [B,T,V]。Embedding 按整数 ID 查 [V,C] 的可学习向量表；Linear 变换最后一维，将 C 维表示映射到 V 个候选分数。例如 B=2、T=32、V=4096，logits=[2,32,4096]。

易错：ID 大小不代表语义大小；logits 是原始分数，不是概率或最终 ID。

## 5. 【必会】shape、view、reshape 有哪些容易出错的地方？

答：x.shape 是属性，x.size() 是方法，x.size(-1) 取最后一维。重塑前后元素数必须相同，[2,32,8,32] 需要 16384 个元素，不能容纳 131072 个元素。应检查上游投影宽度；reshape 可处理部分非连续布局，但不修复错误的总元素数或语义。

易错：x.shape() 会把 torch.Size 当函数调用；不能为了消除报错随意改变头数。

## 6. 【必会】GQA 的 Q/K/V 应输出多宽，拆头后是什么形状？

答：Q 输出 Hq×D，K/V 各输出 Hkv×D。C=256、Hq=8、Hkv=2、D=32 时，Q 宽度 256，K/V 各为 64。先变成 [B,T,H,D]，再交换 T/H 得到 Q=[B,8,T,32]、K/V=[B,2,T,32]。

易错：K/V 投影不能照抄 Q 宽度，也不能把 dim 又乘一次 n_heads。

## 7. 【面试】MHA、GQA、MQA 有何区别？

答：MHA 的 Hkv=Hq；GQA 将 Q 头分组，同组共享一个 KV 头；MQA 的 Hkv=1。固定其他条件时，KV 投影和缓存规模随 Hkv 减少。计算注意力时可把紧凑 KV 按组扩展到 Q 头数，但保存缓存时应保留 Hkv 维度。

易错：GQA 没有同步减少 Q 头数；节省 KV 不代表同倍率节省总内存或加速。质量变化需要实验。

## 8. 【必会】RMSNorm 的公式、计算轴与参数是什么？

答：y=x/sqrt(mean(x²)+eps)×weight。每个 token 沿最后一维 C 独立归一化，均方值形状 [B,T,1]，广播回 [B,T,C]。weight 是 C 维可学习缩放，初始为 1；eps 防止分母过小。当前代码先转 FP32 计算平方、均值和倒平方根，再恢复输入类型。

易错：沿 T 维求均值会混合位置，可能泄漏未来；归一化不改变形状，不保证均值为 0。

## 9. 【面试】RMSNorm 与 LayerNorm 有何区别？

答：LayerNorm 通常减去均值后除以标准差，并可有缩放和偏置。RMSNorm 不减均值，用均方根调整尺度，本项目只含缩放而无偏置。二者都可以逐 token 沿特征维计算。

易错：RMSNorm 不是 softmax，不会把向量变成和为 1 的概率。

## 10. 【必会】RoPE 怎样编码位置？为什么要求偶数头维度？

答：相邻通道 (u,v) 组成二维向量，位置 p、频率 ω 给出角度 pω，旋转后为 (u cos−v sin, u sin+v cos)。不同通道对有不同频率，所以 D 要能两两配对。当前实现对 Q/K 旋转，不旋转 V，没有可学习参数；理想精确计算下保持长度，位置 0 不改变向量。

易错：RoPE 不增加维度，也不是把位置直接加到 ID 上；FP32 角度计算不代表整个模型都必须双精度。

## 11. 【面试】RoPE 的相对位置性质是什么？start_pos 有何作用？

答：设 R(p) 为旋转矩阵，(R(p)q)ᵀ(R(s)k)=qᵀR(s−p)k，点积中的旋转关系取决于位置差。start_pos=5、输入长度 3 时，使用位置 5、6、7，为增量推理保持位置连续。

易错：相对位置性质不保证任意长度外推，也不意味着删除历史后所有层输出保持不变。

## 12. 【必会】SwiGLU 的公式与门控含义是什么？

答：SwiGLU(x)=Wdown(SiLU(Wgate(x)) ⊙ Wup(x))，SiLU(z)=z×sigmoid(z)，⊙ 是逐元素乘法。gate/up 从 C 投影到 hidden_dim，一支调节另一支的特征，down 再回到 C。它逐 token 变换特征，不混合不同位置。

易错：SiLU 不是只在 0～1 的开关；门控不是选择历史 token。必须投影回 C 才能与残差相加。

## 13. 【必会】注意力的计算顺序、形状和缩放是什么？

答：KV 扩展到 Hq 后，Q=[B,Hq,T,D]，Kᵀ=[B,Hq,D,T]，QKᵀ/sqrt(D) 得到 [B,Hq,T,T]。屏蔽未来后沿最后的 key 位置维 softmax，再乘 V，得到 [B,Hq,T,D]，合并头和输出投影回 [B,T,C]。除以 sqrt(D) 控制分数尺度，避免维度变大时 softmax 过度饱和。

易错：softmax 不沿 batch；缩放因子不是随意除以 dim。

## 14. 【面试】因果掩码怎样写？naive 与 SDPA 的 True 含义相同吗？

答：允许 key_position≤query_position。完整长度 3 的允许矩阵为 [[1,0,0],[1,1,0],[1,1,1]]。naive 的 masked_fill(~allowed,-inf) 中 True 表示覆盖；SDPA 布尔 attn_mask 中 True 表示允许，因此传 allowed。

易错：因果注意力允许看自己；SDPA 不保证在所有设备和形状上使用 FlashAttention，也不是自己实现了该内核。

## 15. 【面试】为什么 RMSNorm/FFN 不泄漏未来，如何验证整个模型？

答：它们只沿每个 token 的特征维处理，Embedding/输出投影也逐位置执行；跨位置通信由因果注意力控制。测试保持前缀不变、修改后缀，前缀 logits 应在合理浮点容差内不变。

易错：形状不变不能证明无泄漏，必须检查归约维度和数据依赖。

## 16. 【必会】Pre-Norm Block 与残差如何计算？

答：x1=x+Attention(RMSNorm(x))，然后 y=x1+SwiGLU(RMSNorm(x1))。残差保留直接信息与梯度通路，子层学习需要添加的变化。两次相加保持 [B,T,C]，第二个子层使用已经更新的 x1。

易错：不能省略两次残差，也不能让第二个子层仍使用最初的 x。

## 17. 【必会】ModuleList 和 Parameter 为什么重要？

答：ModuleList 注册子模块，使参数能被 parameters、state_dict 和设备迁移找到；普通 Python 列表不会自动注册内部模块。Parameter 标记可学习张量，如 RMSNorm.weight，供优化器获取。多个 Block 应分别实例化，以免意外共享同一个对象。

易错：把普通张量放在属性里不等于注册可学习参数。

## 18. 【面试】Weight tying 与复制权重有什么区别？

答：lm_head.weight 指向 embedding.weight，共享同一个 [V,C] Parameter，输入与输出路径的梯度都贡献到它，减少独立参数数量。复制数值仅让初始值相同，之后仍是两个参数。

易错：数值相等不证明共享，应检查对象身份与更新行为。

## 19. 【必会】标签和交叉熵怎样对齐？

答：原文 [a,b,c,d,e] 形成输入 [a,b,c,d] 和目标 [b,c,d,e]。调用者移位，模型不再次移位。logits [B,T,V] 展平为 [B×T,V]，目标展平为 [B×T]，直接传给 cross_entropy，不提前 softmax。

易错：再次移位会预测隔两个位置；loss 不是简单统计选错次数。

## 20. 【面试】smoke、后端对齐、微型过拟合分别证明什么？

答：smoke 证明一次前向、反向和更新可以运行；后端应比较同权重输入下的 logits 和梯度，不只看打印 loss；固定小批次过拟合验证可学习性，不证明泛化。Day 1 还检查非法配置、边界、RMSNorm/RoPE 参考、GQA 等价、权重共享与因果性。

易错：Ran 0 tests 的 OK 不代表测试通过；四位小数相同不等于全部输出一致。

## 21. 【面试】怎样用一分钟介绍模型部分？

答：我用 PyTorch 实现小型 Decoder-only 模型，集成 Pre-Norm、RMSNorm、RoPE、SwiGLU、GQA 和权重共享，管理完整的张量形状与标签对齐。提供 naive/SDPA 因果注意力，用参考公式、前向/梯度对齐、未来隔离和微型过拟合验证。它是小模型工程实践，不代表成熟语言能力或自研 GPU 内核。

易错：应能现场画 Q/K/V 形状和掩码，而不只是列出术语。

[下一篇：Day 2](day2-review.md) · [文档导航](../README.md)
