class MCFM(nn.Module):
    """
    - 共享上下文: [avg, max, edge] -> 1x1 生成分支logits
    - 分支上下文: 每分支自己的 GAP 描述经 1x1 调制到对应logits上
    - 温度Softmax: 可学习 tau 控制“竞争温度”
    - 空间门: DWConv3x3 + PWConv1x1 -> Sigmoid
    - 可学习残差: out = fused*spatial + beta*sum
    - 全流程仅用 Conv/Pool/Norm，calflops 友好
    """
    def __init__(self, inc, dim, reduction=16, init_tau=0.7, init_beta=0.1):
        super().__init__()
        self.height = len(inc)
        self.dim = dim

        # 1) 对齐通道（仅在需要时投影）
        self.proj = nn.ModuleList([
            nn.Conv2d(i, dim, 1, bias=False) if i != dim else nn.Identity()
            for i in inc
        ])

        # 2) 分支前统一归一化（稳定统计）
        # 对 [B*height, C, H, W] 统一做 GroupNorm(=LayerNorm2d)
        self.pre_norm = nn.GroupNorm(1, dim)

        # 3) 共享上下文 -> 分支权重 (avg + max + edge)
        # 通道数: avg(C) + max(C) + edge(C) = 3C
        self.shared_fc = nn.Conv2d(3 * dim, self.height * dim, 1, bias=False)

        # 4) 分支上下文（每个分支自己的 GAP）-> 调制项
        self.branch_fc = nn.Conv2d(dim, dim, 1, bias=False)

        # 5) 温度 softmax（learnable tau，限制在[0.5, 2.5]附近）
        self.tau_param = nn.Parameter(torch.tensor(float(init_tau)))
        self.softmax = nn.Softmax(dim=1)

        # 6) 空间注意力：DWConv(3x3, groups=C) → PWConv(1x1) → Sigmoid
        self.spatial_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.spatial_pw = nn.Conv2d(dim, 1, 1, bias=False)

        # 7) 可学习残差权重 beta（小值起步，让注意力主导）
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

        # 8) 固定拉普拉斯核，作为“边缘能量”提示（卷积友好、无FFT）
        self.edge_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            k = torch.tensor([[0.,  1., 0.],
                              [1., -4., 1.],
                              [0.,  1., 0.]]).view(1, 1, 3, 3)
            self.edge_conv.weight.copy_(k)
        for p in self.edge_conv.parameters():
            p.requires_grad = False  # 固定不训练

        # 将 1x1 的 edge 标量投影到 C 维（与通道对齐）
        self.edge_fc = nn.Conv2d(1, dim, 1, bias=False)

    def _temperature(self):
        # tau >= 0.5 的软阈，防止过尖/过平
        # softplus(x) >= 0；这里 tau = 0.5 + softplus(tau_param)
        return 0.5 + F.softplus(self.tau_param)

    def forward(self, in_feats_):
        # ====== 对齐 + 归一化 ======
        aligned = [p(x) for p, x in zip(self.proj, in_feats_)]   # list of [B,C,H,W]
        B, C, H, W = aligned[0].shape
        x = torch.stack(aligned, dim=1)                          # [B, height, C, H, W]
        # 统一做GroupNorm，增强数值稳定
        x_ = x.view(B * self.height, C, H, W)
        x_ = self.pre_norm(x_)
        x = x_.view(B, self.height, C, H, W)

        # ====== 共享上下文：avg + max + edge ======
        feats_sum = torch.sum(x, dim=1)                          # [B,C,H,W]
        g_avg = F.adaptive_avg_pool2d(feats_sum, 1)              # [B,C,1,1]
        g_max = F.adaptive_max_pool2d(feats_sum, 1)              # [B,C,1,1]
        # 边缘能量（近似频域的“高频强度”）
        gray = feats_sum.mean(1, keepdim=True)                   # [B,1,H,W]
        edge = torch.abs(self.edge_conv(gray))                   # [B,1,H,W]
        edge_g = F.adaptive_avg_pool2d(edge, 1)                  # [B,1,1,1]
        edge_c = self.edge_fc(edge_g)                            # [B,C,1,1]

        shared_ctx = torch.cat([g_avg, g_max, edge_c], dim=1)    # [B,3C,1,1]
        logits_shared = self.shared_fc(shared_ctx)                # [B,height*C,1,1]
        logits_shared = logits_shared.view(B, self.height, C, 1, 1)

        # ====== 分支上下文调制（每分支自己的GAP）======
        b_avg = x.mean(dim=(3, 4), keepdim=True)                 # [B,height,C,1,1]
        b_mod = self.branch_fc(b_avg.view(B * self.height, C, 1, 1))
        b_mod = b_mod.view(B, self.height, C, 1, 1)

        logits = logits_shared + b_mod                           # [B,height,C,1,1]

        # 温度化 Softmax（在分支维度上做竞争）
        tau = self._temperature()
        attn = self.softmax(logits / tau)                        # [B,height,C,1,1]

        # ====== 空间注意力（轻量有效）======
        s = self.spatial_pw(self.spatial_dw(feats_sum))          # [B,1,H,W]
        spatial = torch.sigmoid(s)

        # ====== 融合 + 可学习残差 ======
        fused = torch.sum(x * attn, dim=1)                       # [B,C,H,W]
        out = fused * spatial + self.beta * feats_sum            # [B,C,H,W]
        return out
