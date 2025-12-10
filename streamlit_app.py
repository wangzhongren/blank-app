# -*- coding: utf-8 -*-
# app.py   ← 直接把下面全部内容复制进 Streamlit 的编辑器

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

# ============================ 核心代码（Fully Fixed v2 精简美化版） ============================

VOCAB = ["hello", "hi", "how", "are", "you", "?", "I", "am", "fine", "!", "bye", "see", "later"]

EMB_DIM = 16
np.random.seed(0)

def make_embedding(word):
    base = np.zeros(EMB_DIM)
    if word in ["hello", "hi"]:           base[0] = 1.0   # 问候
    elif word in ["how", "are", "you"]:   base[1] = 1.0   # 提问
    elif word in ["I", "am", "fine"]:     base[2] = 1.0   # 自我陈述
    elif word in ["bye", "see", "later"]: base[3] = 1.0   # 告别
    elif word in ["?", "!"]:              base[4] = 1.0   # 终止符
    else:                                 base[5] = 1.0
    base += np.random.randn(EMB_DIM) * 0.05
    return base

EMBEDDINGS = {w: make_embedding(w) for w in VOCAB}

# 投影矩阵（模拟学习到的参数）
np.random.seed(42)
Q_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
K_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
V_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3

def predict_next(tokens):
    if not tokens:
        return "hello"
    emb = np.stack([EMBEDDINGS[t] for t in tokens])
    Q = emb[-1:] @ Q_PROJ
    K = emb @ K_PROJ
    V = emb @ V_PROJ

    scores = Q @ K.T
    scores = scores - scores.max()                    # 数值稳定
    weights = np.exp(scores) / (np.exp(scores).sum() + 1e-8)
    prototype = (weights @ V).flatten()

    sims = []
    for w in VOCAB:
        sim = np.dot(prototype, EMBEDDINGS[w]) / (
            np.linalg.norm(prototype) * np.linalg.norm(EMBEDDINGS[w]) + 1e-8)
        sims.append(sim)

    # 简单防重复 + 鼓励结束
    banned = set(tokens[-2:]) if len(tokens) >= 2 else {tokens[-1]}
    for w in [VOCAB[i] for i in np.argsort(-np.array(sims))]:
        if w not in banned:
            # 特殊规则：你问完、我说完、告别后优先打标点
            if tokens[-1] in ["you", "fine", "later"] and w in ["?", "!"]:
                return w, prototype, weights.flatten(), sims
            return w, prototype, weights.flatten(), sims

    return "!", prototype, weights.flatten(), sims

# ============================ 交互区 ============================

prompt = st.text_input(
    "输入任意开头，看看 Transformer 下一秒在想什么",
    value="hello how are you",
    key="input"
)

if prompt.strip():
    tokens = prompt.strip().split()
    next_word, prototype, attn_weights, similarities = predict_next(tokens)

    col1, col2 = st.columns(2)

    with col1:
        # 1. 注意力热力图
        fig_att = px.imshow(
            attn_weights.reshape(1, -1),
            labels=dict(x="历史 token", y="", color="注意力权重"),
            x=tokens,
            text_auto=".3f",
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig_att.update_layout(title="1. 注意力权重（它正在看哪里）", height=400)
        st.plotly_chart(fig_att, use_container_width=True)

        # 3. 相似度排行榜
        fig_bar = go.Figure(go.Bar(
            x=VOCAB,
            y=similarities,
            text=[f"{s:.3f}" for s in similarities],
            textposition="outside",
            marker_color="#FF6F61"
        ))
        fig_bar.update_layout(
            title=f"3. 相似度排序 → 下一个词是 <b style='color:#FF6F61'>{next_word}</b>",
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # 2. 原型雷达图
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=prototype,
            theta=[f"d{i}" for i in range(16)],
            fill='toself',
            line_color="#636EFA"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
            title="2. 类别原型向量（它脑子里现在的临时概念）",
            height=580
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.success(f"预测结果：  {prompt}  →  →  **{next_word}**")
    st.balloons()

# ============================ 页脚 ============================
st.markdown("---")
st.markdown("""
**论文下载**：[Dynamic Semantic Categorization Through Self-Referential Attention (PDF)](https://zenodo.org/records/17835987)  
**代码仓库**：[GitHub - wangzhongren/DynamicGPT](https://github.com/wangzhongren/DynamicGPT)
""")
st.caption("“AI = Dynamic Categorization” — Zhongren Wang, 2025")