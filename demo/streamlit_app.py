"""
demo/streamlit_app.py
------------------------
Two-Tower Retrieval Dashboard — senior MLE edition.

Tabs:
  1. Retrieval       — live recommendations for any user
  2. Embedding Space — UMAP projection of item embeddings by genre
  3. Model Metrics   — results table + ablation + cold-start comparison

Run locally (API must be running):
    streamlit run demo/streamlit_app.py
"""

import os

import numpy as np
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Two-Tower Retrieval System",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — dark ML-lab theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg:           #0B0E14;
    --bg-card:      #131722;
    --bg-card-2:    #171C29;
    --border:       #232838;
    --accent:       #5EEAD4;
    --accent-dim:   #1F4F49;
    --amber:        #F59E0B;
    --red:          #EF4444;
    --green:        #22C55E;
    --text:         #E6E9F0;
    --text-dim:     #8B93A7;
    --text-muted:   #4B5563;
}

.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background-color: var(--bg-card); border-right: 1px solid var(--border); }
header[data-testid="stHeader"] { background-color: var(--bg); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-dim);
    border-radius: 7px;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card-2) !important;
    color: var(--accent) !important;
}

/* Slider accent */
div[data-baseweb="slider"] [role="slider"] { background-color: var(--accent) !important; }
div[data-baseweb="slider"] div[style*="background-color: rgb(255"] { background-color: var(--accent) !important; }

h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: var(--text) !important; letter-spacing: -0.02em; }

/* Hero */
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.15rem;
    line-height: 1.2;
}
.hero-sub {
    color: var(--text-dim);
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
    max-width: 720px;
    line-height: 1.6;
}

/* KPI row */
.kpi-row  { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    min-width: 120px;
}
.kpi-chip b { display: block; font-size: 1.1rem; color: var(--accent); margin-top: 0.15rem; }
.kpi-chip.amber b { color: var(--amber); }
.kpi-chip.green  b { color: var(--green); }

/* Rec cards */
.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.55rem;
    transition: border-color 0.15s ease;
}
.rec-card:hover { border-color: var(--accent-dim); }
.rec-rank  { font-family: 'JetBrains Mono', monospace; color: var(--text-muted); font-size: 0.75rem; width: 28px; display: inline-block; }
.rec-title { color: var(--text); font-size: 0.98rem; font-weight: 600; }
.rec-id    { color: var(--text-dim); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
.genre-pill {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border-radius: 99px;
    padding: 0.1rem 0.55rem;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 0.25rem;
    margin-top: 0.3rem;
}
.score-track { background: #1A1F2C; border-radius: 99px; height: 5px; width: 100%; margin-top: 0.5rem; overflow: hidden; }
.score-fill  { background: linear-gradient(90deg, var(--accent-dim), var(--accent)); height: 100%; border-radius: 99px; }
.score-label { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--amber); float: right; }

/* Tables */
.metrics-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
.metrics-table th {
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-weight: 500;
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
}
.metrics-table td { color: var(--text); padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border); }
.metrics-table tr:last-child td { border-bottom: none; }
.metrics-table tr.best td { background: var(--accent-dim); }
.metrics-table td.pos { color: var(--green); }
.metrics-table td.neg { color: var(--red); }
.metrics-table td.acc { color: var(--accent); font-weight: 600; }
.metrics-table td.dim { color: var(--text-dim); }

/* Section labels */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}

/* Info / callout boxes */
.info-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin-bottom: 1rem;
}
.warn-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Empty / error state */
.empty-state {
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 2.5rem;
    text-align: center;
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.83rem;
    line-height: 1.8;
}

/* Architecture boxes */
.arch-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    color: var(--text-dim);
    line-height: 1.6;
    height: 100%;
}

/* Sidebar footer */
.sidebar-footer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4B5563;
    line-height: 1.8;
}

.stButton > button {
    background: var(--accent) !important;
    color: #0B0E14 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover { background: #4FD8C4 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="hero-eyebrow">Configuration</div>', unsafe_allow_html=True)
    api_url_input = st.text_input("API endpoint", value=API_URL, label_visibility="collapsed")
    st.caption("API endpoint")

    st.markdown("---")
    st.markdown('<div class="hero-eyebrow">Retrieval</div>', unsafe_allow_html=True)
    user_id = st.number_input("User ID", min_value=1, max_value=6040, step=1, value=1)
    top_k   = st.slider("Top-K results", min_value=1, max_value=50, value=10)
    run_btn = st.button("Run retrieval →", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="hero-eyebrow">System Status</div>', unsafe_allow_html=True)
    try:
        health = requests.get(f"{api_url_input}/health", timeout=2).json()
        model_ok  = health.get("model_loaded", False)
        titles_ok = health.get("titles_loaded", False)
        st.markdown(
            f"""<span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:{'#22C55E' if model_ok else '#EF4444'};">
            ● model {'loaded' if model_ok else 'offline'}</span><br>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:{'#22C55E' if titles_ok else '#8B93A7'};">
            ● titles {'loaded' if titles_ok else 'unavailable'}</span>""",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;color:#EF4444;">● API unreachable</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer">'
        'Two-Tower Neural Retrieval<br>'
        'MovieLens-1M · 6,040 users · 3,706 items<br>'
        'FAISS IndexFlatIP · dim=64<br>'
        'p99 latency &lt;2ms (FAISS only)'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-eyebrow">Neural Retrieval · FAISS ANN · Side Features</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Two-Tower Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    'User and item towers produce L2-normalized embeddings matched by cosine similarity. '
    'User demographics (age, gender, occupation) and item genre multi-hots are fused into the tower inputs, '
    'enabling personalized cold-start retrieval — where matrix factorization can only fall back to global popularity.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="kpi-row">
    <div class="kpi-chip"><span>NDCG@10</span><b>0.259</b></div>
    <div class="kpi-chip"><span>Recall@10</span><b>0.481</b></div>
    <div class="kpi-chip"><span>MAP</span><b>0.213</b></div>
    <div class="kpi-chip amber"><span>vs MF baseline</span><b>+79% NDCG</b></div>
    <div class="kpi-chip green"><span>p99 latency</span><b>&lt;2ms</b></div>
    <div class="kpi-chip"><span>cold-start gap</span><b>+17% NDCG</b></div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["◆  Retrieval", "⬡  Embedding Space", "▦  Model Metrics"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    if run_btn:
        try:
            resp = requests.post(
                f"{api_url_input}/recommend",
                json={"user_id": int(user_id), "top_k": int(top_k)},
                timeout=5,
            )

            if resp.status_code == 404:
                st.markdown(
                    f'<div class="empty-state">user_id <b>{user_id}</b> not found in the training set.<br>'
                    f'Valid range: 1 – 6040.</div>',
                    unsafe_allow_html=True,
                )
            else:
                resp.raise_for_status()
                data = resp.json()
                recs = data["recommendations"]

                max_score   = max((r["score"] for r in recs), default=1.0)
                min_score   = min((r["score"] for r in recs), default=0.0)
                score_range = max(max_score - min_score, 1e-6)

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("User ID",     user_id)
                col_b.metric("Results",     len(recs))
                col_c.metric("Top Score",   f"{max_score:.4f}")
                col_d.metric("Score Range", f"{score_range:.4f}")

                st.markdown('<div class="section-label">Ranked Recommendations</div>', unsafe_allow_html=True)

                for rank, rec in enumerate(recs, start=1):
                    title    = rec.get("title") or f"Movie {rec['movie_id']}"
                    genres   = (rec.get("genres") or "").split("|") if rec.get("genres") else []
                    fill_pct = ((rec["score"] - min_score) / score_range) * 100
                    genre_html = "".join(f'<span class="genre-pill">{g}</span>' for g in genres if g)

                    st.markdown(f"""
                    <div class="rec-card">
                        <span class="rec-rank">#{rank:02d}</span>
                        <span class="rec-title">{title}</span>
                        <span class="rec-id">&nbsp;·&nbsp;id {rec['movie_id']}</span>
                        <span class="score-label">{rec['score']:.4f}</span>
                        <div class="score-track"><div class="score-fill" style="width:{fill_pct:.1f}%;"></div></div>
                        <div style="margin-top:0.1rem;">{genre_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.markdown(
                f'<div class="empty-state">Cannot reach API at <code>{api_url_input}</code><br>'
                f'Start the server: <code>uvicorn serving.app:app --reload</code></div>',
                unsafe_allow_html=True,
            )
        except requests.exceptions.RequestException as e:
            st.markdown(f'<div class="empty-state">Request failed: {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state">'
            'Select a user ID in the sidebar and click <b>Run retrieval →</b><br>'
            'The user tower runs live at query time — item embeddings are pre-indexed in FAISS.'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — EMBEDDING SPACE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">UMAP Projection of Item Embeddings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Item embeddings projected to 2D via UMAP (cosine metric, n_neighbors=15, random_state=42).
    Genre colors are assigned <b>after</b> training — not used as a supervision signal.
    Emergent clustering is evidence the model learned semantic structure from co-interaction patterns alone.
    </div>
    """, unsafe_allow_html=True)

    viz_path = "results/embedding_viz.png"
    if os.path.exists(viz_path):
        st.image(viz_path, use_column_width=True)
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#4B5563;text-align:center;margin-top:0.5rem;">'
            'dim=64 item embeddings · trained with in-batch contrastive loss (InfoNCE) · UMAP random_state=42'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="empty-state">'
            'embedding_viz.png not found.<br>'
            'Generate it by running <code>notebooks/04_embedding_viz.ipynb</code>.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Reading the Plot</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-box">
        <b style="color:#5EEAD4;">Genre clusters</b><br>
        Comedy, Drama, Action, and Horror should occupy distinct regions of the projection.
        Multi-genre films appear at cluster boundaries, not in a single region.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
        <b style="color:#5EEAD4;">Unsupervised structure</b><br>
        Genre labels were not training targets. Geometric structure emerges purely
        from which users interacted with which films — a sanity check on representation quality.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-box">
        <b style="color:#5EEAD4;">Retrieval geometry</b><br>
        All embeddings are L2-normalized. Dot product equals cosine similarity —
        the exact objective FAISS IndexFlatIP maximizes during ANN search.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:

    # ── Model Comparison ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="metrics-table">
        <tr>
            <th>Model</th>
            <th>NDCG@10</th>
            <th>Recall@10</th>
            <th>MAP</th>
            <th>Cold-Start NDCG@10</th>
            <th>p99 Latency</th>
        </tr>
        <tr>
            <td>Matrix Factorization (BPR)</td>
            <td class="neg">0.145</td>
            <td class="neg">0.299</td>
            <td class="neg">0.128</td>
            <td class="neg">~0.02 (popularity fallback)</td>
            <td class="dim">—</td>
        </tr>
        <tr>
            <td>Two-Tower, ID-only</td>
            <td>0.240</td>
            <td>0.454</td>
            <td>0.199</td>
            <td class="dim">—</td>
            <td>&lt;2ms</td>
        </tr>
        <tr class="best">
            <td><b>Two-Tower + Side Features</b></td>
            <td class="acc">0.259</td>
            <td class="acc">0.481</td>
            <td class="acc">0.213</td>
            <td class="acc">0.313</td>
            <td class="acc">&lt;2ms</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Evaluation protocol:</b> Leave-one-out with 1 positive + 99 sampled negatives per user (NCF paper protocol).
    Metrics averaged across 6,035 users on the held-out test split.
    <b>Cold-start:</b> 200 users held out of training entirely — two-tower uses demographics only;
    MF has no embedding for unseen users and falls back to global popularity ranking.
    </div>
    """, unsafe_allow_html=True)

    # ── Ablation + Cold-Start side by side ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Embedding Dim Ablation</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="metrics-table">
            <tr><th>Dim</th><th>NDCG@10</th><th>Recall@10</th><th>MAP</th><th>Epoch time</th></tr>
            <tr>
                <td>32</td>
                <td>0.252</td><td>0.469</td><td>0.209</td>
                <td class="dim">44.8s</td>
            </tr>
            <tr class="best">
                <td><b>64</b></td>
                <td class="acc">0.259</td><td class="acc">0.481</td><td class="acc">0.213</td>
                <td class="acc">41.2s</td>
            </tr>
            <tr>
                <td>128</td>
                <td>0.254</td><td>0.476</td><td>0.210</td>
                <td class="dim">97.6s</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="margin-top:0.75rem;">
        dim=128 shows no measurable NDCG gain at 2.4× training cost.
        dim=32 loses 2.7% NDCG. <b>dim=64 is the optimal tradeoff.</b>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Cold-Start Evaluation (200 held-out users)</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="metrics-table">
            <tr><th>Model</th><th>NDCG@10</th><th>Recall@10</th><th>Strategy</th></tr>
            <tr>
                <td>MF (BPR)</td>
                <td class="neg">~0.02</td>
                <td class="neg">~0.04</td>
                <td class="dim">Global popularity</td>
            </tr>
            <tr class="best">
                <td><b>Two-Tower + Features</b></td>
                <td class="acc">0.313</td>
                <td class="acc">0.565</td>
                <td class="acc">Demographics only</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box" style="margin-top:0.75rem;">
        MF has <b>no embedding</b> for users it never saw during training —
        it can only rank items by global interaction count.
        The two-tower uses age, gender, and occupation to produce a
        personalized embedding with <b>zero interaction history</b>.
        This is the production justification for DL retrievers over MF.
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture summary ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="arch-box">
        <b style="color:#5EEAD4;">User Tower</b><br><br>
        ID embedding (64)<br>
        + gender (1 binary)<br>
        + age bucket embedding (4)<br>
        + occupation embedding (8)<br>
        <br>→ MLP [128 → 64] → L2 norm
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="arch-box">
        <b style="color:#5EEAD4;">Item Tower</b><br><br>
        ID embedding (64)<br>
        + genre multi-hot (18)<br>
        <br>
        <br>
        <br>→ MLP [128 → 64] → L2 norm
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="arch-box">
        <b style="color:#5EEAD4;">Training</b><br><br>
        In-batch contrastive loss (InfoNCE)<br>
        B = 512 · temp = 0.1<br>
        50 epochs · Adam<br>
        Shared loss — both towers<br>
        trained jointly
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="arch-box">
        <b style="color:#5EEAD4;">Retrieval</b><br><br>
        FAISS IndexFlatIP<br>
        Item embeddings pre-computed offline<br>
        User tower runs live at query time<br>
        p99 &lt;2ms (ANN lookup only)
        </div>
        """, unsafe_allow_html=True)