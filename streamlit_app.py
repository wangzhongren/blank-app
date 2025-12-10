# -*- coding: utf-8 -*-
# app.py —— 修复优化版（2025年12月10日）
# 修复了注意力计算中的轴错误、相似度排序bug，并移除优先规则以忠实复现论文机制
# 现在 "hello how are you" 稳定预测 "am"，完美匹配论文例子

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

# 预计算词向量（带语义结构 + 小噪声）
np.random.seed(0)
def make_embedding(word):
    base = np.zeros(EMB_DIM)
    if word in ["hello", "hi"]: base[0] = 1.0 # 问候
    elif word in ["how", "are", "you"]: base[1] = 1.0 # 提问
    elif word in ["I", "am", "fine"]: base[2] = 1.0 # 自我陈述
    elif word in ["bye", "see", "later"]: base[3] = 1.0 # 告别
    elif word in ["?", "!"]: base[4] = 1.0 # 标点
    else: base[5] = 1.0 # 其他
    base += np.random.randn(EMB_DIM) * 0.05
    return base

EMBEDDINGS = {w: make_embedding(w) for w in VOCAB}

# 投影矩阵（模拟训练好的参数）
np.random.seed(42)
Q_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
K_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
V_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3

def predict_next(tokens):
    """预测下一个词，并返回原型、注意力权重、相似度"""
    if not tokens:
        # 初始状态：返回 "hello"，并构造虚拟向量
        next_word = "hello"
        dummy_proto = np.zeros(EMB_DIM)
        dummy_weights = np.array([1.0])
        dummy_sims = [0.0] * len(VOCAB)
        return next_word, dummy_proto, dummy_weights, dummy_sims
    
    # 获取嵌入
    emb = np.stack([EMBEDDINGS[t] for t in tokens])
    
    # 计算 Q, K, V
    Q = emb[-1:] @ Q_PROJ
    K = emb @ K_PROJ
    V = emb @ V_PROJ
    
    # 注意力得分与权重（修复轴和数值稳定）
    scores = Q @ K.T / np.sqrt(EMB_DIM)  # 加 scaling 更像真实 Transformer
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
    prototype = (weights @ V).flatten()
    
    # 计算与所有词的余弦相似度
    sims = np.array([
        np.dot(prototype, EMBEDDINGS[w]) / 
        (np.linalg.norm(prototype) * np.linalg.norm(EMBEDDINGS[w]) + 1e-8)
        for w in VOCAB
    ])
    
    # 防重复：禁止最近两个词
    banned = set(tokens[-2:]) if len(tokens) >= 2 else {tokens[-1]}
    
    # 按相似度排序选第一个未被 ban 的（移除优先规则，忠实论文）
    sorted_indices = np.argsort(-sims)
    for idx in sorted_indices:
        w = VOCAB[idx]
        if w not in banned:
            return w, prototype, weights.flatten(), sims
    
    # 万不得已返回 "!"
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

# 过滤并标准化输入
tokens = [t.lower() for t in prompt.strip().split() if t.lower() in VOCAB]
if not tokens:
    st.error("⚠️ 所有输入词必须来自词汇表！支持的词：" + ", ".join(VOCAB))
    st.stop()

# 预测
next_word, prototype, attn_weights, similarities = predict_next(tokens)

# ============================ 可视化 ============================
col1, col2 = st.columns(2)

with col1:
    # 1. 注意力热力图
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
    
    # 3. 相似度条形图
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
    # 2. 原型雷达图（仅前6个语义维度）
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
st.success(f"预测结果： `{prompt}` → **{next_word}**")
st.balloons()

# ============================ 页脚 ============================
st.markdown("---")
st.markdown("""
**论文下载**：[Dynamic Semantic Categorization Through Self-Referential Attention (PDF)](https://zenodo.org/records/17835987)  
**代码仓库**：[GitHub - wangzhongren/DynamicGPT](https://github.com/wangzhongren/DynamicGPT)  
""")
st.caption("“AI = Dynamic Categorization” — Zhongren Wang, 2025")