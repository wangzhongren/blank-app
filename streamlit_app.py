# -*- coding: utf-8 -*-
# app.py —— 带种子调节的动态语义原型可视化器（2025年12月10日）
# 基于 Wang (2025): "AI = Dynamic Categorization"
# 支持实时调整 QKV 投影种子，观察 Transformer 如何动态构建语义原型

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ============================ 页面设置 ============================
st.set_page_config(
    page_title="Transformer 真的在想什么？",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================ 侧边栏：QKV 种子控制 ============================
with st.sidebar:
    st.header("⚙️ 模型参数")
    seed_proj = st.slider(
        "QKV 投影种子 (seed)",
        min_value=0,
        max_value=100,
        value=1,  # 论文推荐值
        help="改变此值会重新生成 Q/K/V 投影矩阵，影响注意力行为"
    )
    st.info(f"当前 seed = {seed_proj}")
    st.markdown("""
    - **seed=1**：通常生成合理对话（如 `you → ?`）  
    - **seed=42**：可能生成“乱序”（如 `you → am`）  
    - 尝试不同值，观察原型和预测如何变化！
    """)

# ============================ 标题区 ============================
st.title("🧠 Transformer 真的在想什么？")
st.markdown("""
**7 页论文 + 30 行代码，彻底看穿自回归生成过程**  
作者：Zhongren Wang  修订版：2025年12月  

> 你现在就能亲眼看见注意力在大脑里形成的「临时语义原型」（category prototype）
""")
st.divider()

# ============================ 核心模型定义 ============================
VOCAB = ["hello", "hi", "how", "are", "you", "?", "I", "am", "fine", "!", "bye", "see", "later"]
EMB_DIM = 16

# 固定词嵌入（seed=0 保证跨运行一致性）
np.random.seed(0)
def make_embedding(word):
    base = np.zeros(EMB_DIM)
    if word in ["hello", "hi"]: base[0] = 1.0  # 问候
    elif word in ["how", "are", "you"]: base[1] = 1.0  # 提问
    elif word in ["I", "am", "fine"]: base[2] = 1.0  # 自我陈述
    elif word in ["bye", "see", "later"]: base[3] = 1.0  # 告别
    elif word in ["?", "!"]: base[4] = 1.0  # 标点
    else: base[5] = 1.0  # 其他
    base += np.random.randn(EMB_DIM) * 0.05  # 小噪声
    return base

EMBEDDINGS = {w: make_embedding(w) for w in VOCAB}

# 使用侧边栏 seed 生成 QKV 投影矩阵
np.random.seed(seed_proj)
Q_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
K_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
V_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3

def predict_next(tokens):
    """预测下一个词，并返回原型、注意力权重、相似度"""
    if not tokens:
        # 初始状态
        next_word = "hello"
        dummy_proto = np.zeros(EMB_DIM)
        dummy_weights = np.array([1.0])
        dummy_sims = [0.0] * len(VOCAB)
        return next_word, dummy_proto, dummy_weights, dummy_sims
    
    # 获取嵌入
    emb = np.stack([EMBEDDINGS[t] for t in tokens])
    
    # QKV 投影
    Q = emb[-1:] @ Q_PROJ
    K = emb @ K_PROJ
    V = emb @ V_PROJ
    
    # 注意力计算（带缩放和数值稳定）
    scores = Q @ K.T / np.sqrt(EMB_DIM)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
    prototype = (weights @ V).flatten()
    
    # 余弦相似度
    sims = np.array([
        np.dot(prototype, EMBEDDINGS[w]) /
        (np.linalg.norm(prototype) * np.linalg.norm(EMBEDDINGS[w]) + 1e-8)
        for w in VOCAB
    ])
    
    # 防重复：禁止最近两个词
    banned = set(tokens[-2:]) if len(tokens) >= 2 else {tokens[-1]}
    
    # 按相似度排序，选第一个未被 ban 的
    sorted_indices = np.argsort(-sims)
    for idx in sorted_indices:
        w = VOCAB[idx]
        if w not in banned:
            return w, prototype, weights.flatten(), sims
    
    return "!", prototype, weights.flatten(), sims

# ============================ 用户交互区 ============================
prompt = st.text_input(
    "请输入开头（仅限以下词：hello, hi, how, are, you, I, am, fine, bye, see, later, ?, !）",
    value="hello how are you",
    key="input"
)

if not prompt.strip():
    st.info("请输入至少一个词以启动生成。")
    st.stop()

# 过滤输入
tokens = [t.lower() for t in prompt.strip().split() if t.lower() in VOCAB]
if not tokens:
    st.error("⚠️ 所有输入词必须来自词汇表！支持的词：" + ", ".join(VOCAB))
    st.stop()

# 预测
next_word, prototype, attn_weights, similarities = predict_next(tokens)

# ============================ 可视化 ============================
col1, col2 = st.columns(2)

with col1:
    # 注意力热力图
    fig_att = px.imshow(
        attn_weights.reshape(1, -1),
        labels=dict(x="历史 token", y="", color="注意力权重"),
        x=tokens,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig_att.update_layout(title="1. 注意力权重（它正在关注哪里）", height=300)
    st.plotly_chart(fig_att, use_container_width=True)
    
    # 相似度条形图
    fig_bar = go.Figure(go.Bar(
        x=VOCAB,
        y=similarities,
        text=[f"{s:.2f}" for s in similarities],
        textposition="outside",
        marker_color="#FF6F61"
    ))
    fig_bar.update_layout(
        title=f"3. 词汇相似度 → 预测下一个词：<b style='color:#FF6F61'>{next_word}</b>",
        height=450,
        margin=dict(t=50, b=100)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # 语义原型雷达图（前6维）
    semantic_labels = ["Greeting", "Question", "Self", "Farewell", "Punctuation", "Other"]
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=prototype[:6],
        theta=semantic_labels,
        fill='toself',
        line_color="#636EFA"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-0.5, 1.2])),
        title="2. 语义原型（6个动态认知维度）",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ============================ 结果展示 ============================
st.success(f"预测结果：`{prompt}` → **{next_word}** (seed={seed_proj})")
st.balloons()

# ============================ 页脚 ============================
st.markdown("---")
st.markdown("""
**论文下载**：[Dynamic Semantic Categorization Through Self-Referential Attention (PDF)](https://zenodo.org/records/17835987)  
**代码仓库**：[GitHub - wangzhongren/DynamicGPT](https://github.com/wangzhongren/DynamicGPT)  
""")
st.caption("“AI = Dynamic Categorization” — Zhongren Wang, 2025")