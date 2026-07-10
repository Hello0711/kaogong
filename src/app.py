"""
公务员考试职位竞争分析 · Streamlit 交互看板
==========================================
运行:
    streamlit run src/app.py
数据依赖: data/clean/positions.parquet（先执行 python src/build_dataset.py 生成）
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
st.set_page_config(page_title="公务员考试职位竞争分析", page_icon="📊", layout="wide")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(PROJECT_DIR, "data", "clean", "positions.parquet")

CAT_FEATURES = ["考试类型", "省份", "部门系统", "最低学历"]
NUM_FEATURES = ["年份", "招录人数", "是否不限专业", "专业数量", "学历及以上", "面试比例"]
特征映射 = {
    "exam_type": "考试类型", "province": "省份", "dep_sys": "部门系统",
    "min_degree_name": "最低学历", "year": "年份", "employ_count": "招录人数",
    "major_unlimited": "是否不限专业", "major_term_count": "专业数量",
    "degree_open_upward": "学历及以上", "interview_ratio_num": "面试比例",
}


# ----------------------------------------------------------------------------
# 数据与模型缓存
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="正在加载数据 ...")
def load_data():
    if not os.path.exists(CLEAN):
        return None
    return pd.read_parquet(CLEAN)


@st.cache_resource(show_spinner="正在训练竞争强度预测模型 ...")
def train_model(df):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = df[df["report_ratio"].notna() & (df["employ_count"] >= 1)].copy()
    cap = m["report_ratio"].quantile(0.99)
    m["y"] = np.log1p(m["report_ratio"].clip(upper=cap))
    top_sys = m["department_system"].value_counts().head(30).index
    m["dep_sys"] = np.where(m["department_system"].isin(top_sys),
                            m["department_system"], "其他")

    X = m[list(特征映射.keys())].rename(columns=特征映射).copy()
    for c in CAT_FEATURES:
        X[c] = X[c].fillna("未知").astype("category")
    X["面试比例"] = X["面试比例"].fillna(X["面试比例"].median())
    X["学历及以上"] = X["学历及以上"].fillna(0)
    y = m["y"]

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=8,
        categorical_features=CAT_FEATURES, random_state=42)
    model.fit(X, y)
    return model, list(top_sys), cap


def apply_font(ax=None):
    pass


# ----------------------------------------------------------------------------
# 页面 1 · 项目概览
# ----------------------------------------------------------------------------
def page_overview(df):
    st.title("📊 公务员考试职位竞争分析")
    st.caption("数据范围：2019–2026 年国考 + 31 省省考 · 端到端数据分析项目")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("职位记录数", f"{len(df):,}")
    c2.metric("覆盖年份", f"{int(df['year'].min())}–{int(df['year'].max())}")
    c3.metric("覆盖省份", f"{df['province'].nunique()} 个")
    c4.metric("报录比中位数", f"{df['report_ratio'].median():.1f} : 1")

    st.markdown("---")
    st.subheader("核心指标：报录比 = 报名人数 / 招录人数")
    st.markdown(
        "报录比衡量岗位「上岸难度」。下方为各竞争档位的岗位数量分布。")

    heat = (df["heat_level"].value_counts()
            .reindex(["低竞争(<20)", "中竞争(20-50)", "高竞争(50-100)", "极热(>=100)"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=heat.index, y=heat.values, palette="rocket_r", ax=ax)
    for i, v in enumerate(heat.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    ax.set_ylabel("岗位数"); ax.set_xlabel("")
    ax.set_title("竞争激烈度分档分布")
    st.pyplot(fig)

    st.subheader("数据样例")
    st.dataframe(df.head(20), use_container_width=True)


# ----------------------------------------------------------------------------
# 页面 2 · 多维竞争分析
# ----------------------------------------------------------------------------
def page_analysis(df):
    st.title("🔍 多维竞争格局分析")

    with st.sidebar:
        st.header("筛选条件")
        exam = st.multiselect("考试类型", sorted(df["exam_type"].dropna().unique()),
                              default=list(df["exam_type"].dropna().unique()))
        yrange = st.slider("年份范围", int(df["year"].min()), int(df["year"].max()),
                           (int(df["year"].min()), int(df["year"].max())))
        provs = st.multiselect("省份（留空为全部）",
                               sorted(df["province"].dropna().unique()))

    d = df[df["exam_type"].isin(exam) &
           df["year"].between(yrange[0], yrange[1])]
    if provs:
        d = d[d["province"].isin(provs)]
    st.caption(f"当前筛选样本：{len(d):,} 条")

    tab1, tab2, tab3, tab4 = st.tabs(["时间趋势", "地域对比", "学历维度", "专业限制"])

    with tab1:
        trend = d.groupby(["year", "exam_type"]).agg(
            招录总人数=("employ_count", "sum"),
            报录比中位数=("report_ratio", "median")).reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.lineplot(data=trend, x="year", y="招录总人数", hue="exam_type",
                         marker="o", ax=ax)
            ax.set_title("招录规模逐年变化"); ax.set_xlabel("年份")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.lineplot(data=trend, x="year", y="报录比中位数", hue="exam_type",
                         marker="o", ax=ax)
            ax.set_title("竞争强度逐年变化"); ax.set_xlabel("年份")
            st.pyplot(fig)

    with tab2:
        prov = (d[d["report_ratio"].notna()].groupby("province")
                .agg(职位数=("position_code", "size"),
                     报录比中位数=("report_ratio", "median"))
                .sort_values("报录比中位数", ascending=False))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=prov.reset_index(), y="province", x="报录比中位数",
                    palette="rocket_r", ax=ax)
        ax.set_title("各省份竞争强度排名（报录比中位数）")
        ax.set_ylabel(""); ax.set_xlabel("报录比中位数")
        st.pyplot(fig)
        st.dataframe(prov.rename_axis("省份"), use_container_width=True)

    with tab3:
        deg = (d[d["report_ratio"].notna() & d["min_degree_name"].notna()]
               .groupby("min_degree_name")
               .agg(职位数=("position_code", "size"),
                    报录比中位数=("report_ratio", "median")))
        deg = deg.reindex(["专科", "本科", "硕士", "博士"]).dropna(how="all")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=deg.reset_index(), x="min_degree_name", y="职位数",
                        palette="crest", ax=ax)
            ax.set_title("不同学历门槛职位数量"); ax.set_xlabel("最低学历")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=deg.reset_index(), x="min_degree_name", y="报录比中位数",
                        palette="flare", ax=ax)
            ax.set_title("不同学历门槛竞争强度"); ax.set_xlabel("最低学历")
            st.pyplot(fig)

    with tab4:
        grp = (d[d["report_ratio"].notna()]
               .assign(专业要求=lambda x: np.where(x["major_unlimited"] == 1,
                                                 "不限专业", "限定专业"))
               .groupby("专业要求")
               .agg(职位数=("position_code", "size"),
                    报录比中位数=("report_ratio", "median")))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=grp.reset_index(), x="专业要求", y="报录比中位数",
                    palette="Set2", ax=ax)
        for i, v in enumerate(grp["报录比中位数"]):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
        ax.set_title("「不限专业」vs 限定专业 竞争强度")
        st.pyplot(fig)
        st.info("「不限专业」岗位报名门槛最低，通常竞争最激烈。")


# ----------------------------------------------------------------------------
# 页面 3 · 专业要求文本挖掘
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="正在统计专业词频 ...")
def compute_major_freq(df):
    texts = df.loc[df["major_unlimited"] == 0, "major_requirement"].dropna()
    counter = Counter()
    for t in texts:
        for term in re.split(r"[、,，;；/]", t):
            term = term.strip()
            if len(term) >= 2:
                counter[term] += 1
    return pd.DataFrame(counter.most_common(30), columns=["专业/专业类", "岗位数"])


def page_nlp(df):
    st.title("📝 专业要求文本挖掘（NLP）")
    st.markdown("对自由文本 `专业要求` 结构化，量化各专业/专业类的**岗位机会度**。")

    topn = st.slider("展示 TOP N 专业", 10, 30, 20)
    top_major = compute_major_freq(df).head(topn)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, topn * 0.35 + 1))
        sns.barplot(data=top_major, y="专业/专业类", x="岗位数",
                    palette="viridis", ax=ax)
        ax.set_title(f"岗位需求最多的专业 / 专业类 TOP{topn}")
        st.pyplot(fig)
    with col2:
        st.dataframe(top_major, use_container_width=True, height=topn * 35 + 40)

    st.markdown("---")
    st.subheader("「不限专业」岗位占比")
    col1, col2 = st.columns(2)
    with col1:
        year_unlim = (df.groupby("year")["major_unlimited"].mean() * 100).round(1)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(x=year_unlim.index, y=year_unlim.values, marker="o",
                     color="#c0392b", ax=ax)
        ax.set_title("逐年变化"); ax.set_ylabel("不限专业占比 %"); ax.set_xlabel("年份")
        st.pyplot(fig)
    with col2:
        prov_unlim = (df.groupby("province")["major_unlimited"].mean() * 100
                      ).round(1).sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=prov_unlim.values, y=prov_unlim.index, palette="mako", ax=ax)
        ax.set_title("各省占比 TOP12"); ax.set_xlabel("占比 %")
        st.pyplot(fig)


# ----------------------------------------------------------------------------
# 页面 4 · 竞争预测与性价比推荐器
# ----------------------------------------------------------------------------
def page_recommender(df, model, top_sys, cap):
    st.title("🎯 竞争预测与「性价比」职位推荐器")
    st.markdown(
        "输入考生画像，系统在符合报考条件的岗位中，按**模型预测竞争强度从低到高**推荐。")

    level_map = {"专科": 1, "本科": 2, "硕士": 3, "博士": 4}
    years = sorted(df["year"].dropna().unique())

    with st.sidebar:
        st.header("考生画像")
        degree = st.selectbox("你的学历", ["专科", "本科", "硕士", "博士"], index=1)
        major_kw = st.text_input("专业关键词（如 计算机 / 会计 / 法学）", "计算机")
        prov = st.selectbox("目标省份（全部）",
                            ["全部"] + sorted(df["province"].dropna().unique()))
        exam = st.selectbox("考试类型", ["全部", "国考", "省考"])
        year = st.selectbox("年份", years, index=len(years) - 1)
        topn = st.slider("推荐数量", 5, 30, 10)

    cand = df[df["year"] == year].copy()
    if prov != "全部":
        cand = cand[cand["province"] == prov]
    if exam != "全部":
        cand = cand[cand["exam_type"] == exam]
    my_level = level_map[degree]
    cand = cand[cand["min_degree_level"].fillna(my_level) <= my_level]
    if major_kw.strip():
        mask = ((cand["major_unlimited"] == 1) |
                cand["major_requirement"].fillna("").str.contains(major_kw.strip()))
        cand = cand[mask]

    if cand.empty:
        st.warning("没有匹配的岗位，请放宽筛选条件。")
        return

    feat = pd.DataFrame({
        "考试类型": cand["exam_type"],
        "省份": cand["province"],
        "部门系统": np.where(cand["department_system"].isin(top_sys),
                          cand["department_system"], "其他"),
        "最低学历": cand["min_degree_name"].fillna("未知"),
        "年份": cand["year"],
        "招录人数": cand["employ_count"].fillna(1),
        "是否不限专业": cand["major_unlimited"],
        "专业数量": cand["major_term_count"],
        "学历及以上": cand["degree_open_upward"].fillna(0),
        "面试比例": cand["interview_ratio_num"].fillna(3),
    })
    for c in CAT_FEATURES:
        feat[c] = feat[c].astype("category")
    cand = cand.assign(预测报录比=np.expm1(model.predict(feat[CAT_FEATURES + NUM_FEATURES])))

    out = (cand.sort_values("预测报录比")
           .loc[:, ["province", "city", "department", "position_name",
                    "major_requirement", "min_degree_name", "employ_count", "预测报录比"]]
           .head(topn).reset_index(drop=True)
           .rename(columns={"province": "省份", "city": "城市", "department": "招录部门",
                            "position_name": "职位名称", "major_requirement": "专业要求",
                            "min_degree_name": "最低学历", "employ_count": "招录人数"}))
    out["预测报录比"] = out["预测报录比"].round(1)

    c1, c2, c3 = st.columns(3)
    c1.metric("匹配岗位总数", f"{len(cand):,}")
    c2.metric("推荐岗位预测报录比区间",
              f"{out['预测报录比'].min():.0f} – {out['预测报录比'].max():.0f}")
    c3.metric("全部匹配岗位预测中位", f"{cand['预测报录比'].median():.0f} : 1")

    st.subheader(f"性价比岗位推荐 TOP{topn}")
    st.dataframe(out, use_container_width=True)
    st.caption("预测报录比越低，代表竞争越小、上岸相对越容易。")


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------
def main():
    df = load_data()
    if df is None:
        st.error(f"未找到数据文件：{CLEAN}\n\n请先运行： python src/build_dataset.py")
        st.stop()

    page = st.sidebar.radio(
        "导航",
        ["项目概览", "多维竞争分析", "专业文本挖掘", "竞争预测与推荐器"])
    st.sidebar.markdown("---")

    if page == "项目概览":
        page_overview(df)
    elif page == "多维竞争分析":
        page_analysis(df)
    elif page == "专业文本挖掘":
        page_nlp(df)
    elif page == "竞争预测与推荐器":
        model, top_sys, cap = train_model(df)
        page_recommender(df, model, top_sys, cap)


if __name__ == "__main__":
    main()
