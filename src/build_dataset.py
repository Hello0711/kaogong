"""
模块 1 - 数据工程与清洗
=======================
将 gongkao.7z 中的 32 个原始 CSV（1 个国考 + 31 个省考）合并、清洗并做特征工程，
输出统一的分析级数据集 data/clean/positions.parquet。

运行方式:
    python src/build_dataset.py
"""
from __future__ import annotations

import os
import re
import sys
import glob
import shutil

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
ARCHIVE = os.path.join(os.path.dirname(PROJECT_DIR), "gongkao.7z")

RAW_COLUMNS = [
    "year", "province", "area_name", "working_address", "city", "district",
    "department_code", "position_code", "department_system", "department",
    "sub_department", "position_name", "major_requirement", "degree_requirement",
    "employ_count", "employ_count_desc", "interview_ratio", "remark",
    "competition_ratio", "system_ratio", "department_ratio", "enrollment_count",
]


# ----------------------------------------------------------------------------
# 0. 解压原始数据（幂等：已解压则跳过）
# ----------------------------------------------------------------------------
def ensure_raw_extracted() -> str:
    """返回解压后的 gongkao 目录路径。"""
    target = os.path.join(RAW_DIR, "gongkao")
    if os.path.isdir(target) and glob.glob(os.path.join(target, "*.csv")):
        print(f"[extract] 已存在解压数据: {target}")
        return target

    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE):
        raise FileNotFoundError(
            f"找不到压缩包 {ARCHIVE}，请把 gongkao.7z 放到项目上一级目录，或手动解压到 {RAW_DIR}")
    try:
        import py7zr
    except ImportError:
        raise ImportError("需要 py7zr 解压 7z 文件: pip install py7zr")

    print(f"[extract] 正在解压 {ARCHIVE} ...")
    with py7zr.SevenZipFile(ARCHIVE, "r") as z:
        z.extractall(RAW_DIR)
    print(f"[extract] 解压完成 -> {target}")
    return target


# ----------------------------------------------------------------------------
# 1. 读取并纵向合并所有 CSV
# ----------------------------------------------------------------------------
def load_raw(gongkao_dir: str) -> pd.DataFrame:
    frames = []

    # 国考文件：顶层目录下的 CSV
    for fp in glob.glob(os.path.join(gongkao_dir, "*.csv")):
        d = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
        d["exam_type"] = "国考"
        d["source_file"] = os.path.basename(fp)
        frames.append(d)

    # 省考文件：子目录（省考/）下每省一个 CSV
    for sub in os.listdir(gongkao_dir):
        subdir = os.path.join(gongkao_dir, sub)
        if not os.path.isdir(subdir):
            continue
        for fp in glob.glob(os.path.join(subdir, "*.csv")):
            d = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
            d["exam_type"] = "省考"
            d["source_file"] = os.path.basename(fp)
            frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    print(f"[load] 合并 {len(frames)} 个文件, 共 {len(df):,} 行")
    return df


# ----------------------------------------------------------------------------
# 2. 清洗辅助函数
# ----------------------------------------------------------------------------
def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def parse_interview_ratio(val: str):
    """'3:1' -> 3.0 ; '' / NaN -> NaN"""
    if not isinstance(val, str):
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*:\s*1\s*$", val)
    if m:
        return float(m.group(1))
    return np.nan


# 学历等级映射：取最低门槛，数值越小门槛越低
DEGREE_LEVEL = {"专科": 1, "大专": 1, "本科": 2, "硕士": 3, "研究生": 3, "博士": 4}


def normalize_degree(val: str):
    """返回 (min_degree_level, degree_open_upward)。
    min_degree_level: 1 专科 / 2 本科 / 3 硕士 / 4 博士
    degree_open_upward: 是否 '及以上'（门槛更友好）。
    """
    if not isinstance(val, str) or not val.strip():
        return (np.nan, np.nan)
    text = val.strip()
    # 找出文本中出现的所有学历，取最低门槛
    levels = []
    for kw, lv in DEGREE_LEVEL.items():
        if kw in text:
            levels.append(lv)
    min_level = min(levels) if levels else np.nan
    open_upward = 1 if "及以上" in text else 0
    return (min_level, open_upward)


# 专业不限的判定
MAJOR_UNLIMITED_PAT = re.compile(r"不限|无要求|不限专业")


def is_major_unlimited(val: str) -> int:
    if not isinstance(val, str) or not val.strip():
        return 1  # 空 => 视为不限专业
    return 1 if MAJOR_UNLIMITED_PAT.search(val) else 0


def count_major_terms(val: str) -> int:
    """粗略统计专业要求中列出的专业/类别数量（按顿号、逗号、分号切分）。"""
    if not isinstance(val, str) or not val.strip():
        return 0
    if MAJOR_UNLIMITED_PAT.search(val):
        return 0
    parts = re.split(r"[、,，;；/]", val)
    return len([p for p in parts if p.strip()])


# ----------------------------------------------------------------------------
# 3. 主清洗流程
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 3.1 去除完全重复行
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in RAW_COLUMNS if c in df.columns])
    print(f"[clean] 去重: {before:,} -> {len(df):,} (删除 {before - len(df):,})")

    # 3.2 文本字段清洗
    text_cols = ["province", "area_name", "working_address", "city", "district",
                 "department_system", "department", "sub_department",
                 "position_name", "major_requirement", "degree_requirement"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan": np.nan, "": np.nan, "None": np.nan})

    # 3.3 数值字段
    df["year"] = to_num(df["year"]).astype("Int64")
    df["employ_count"] = to_num(df["employ_count"])
    df["enrollment_count"] = to_num(df["enrollment_count"])
    df["competition_ratio_official"] = to_num(df["competition_ratio"])
    df["system_ratio"] = to_num(df["system_ratio"])
    df["department_ratio"] = to_num(df["department_ratio"])
    df["interview_ratio_num"] = df["interview_ratio"].apply(parse_interview_ratio)

    # 3.4 学历特征
    deg = df["degree_requirement"].apply(normalize_degree)
    df["min_degree_level"] = deg.apply(lambda x: x[0])
    df["degree_open_upward"] = deg.apply(lambda x: x[1])
    level_name = {1: "专科", 2: "本科", 3: "硕士", 4: "博士"}
    df["min_degree_name"] = df["min_degree_level"].map(level_name)

    # 3.5 专业特征
    df["major_unlimited"] = df["major_requirement"].apply(is_major_unlimited)
    df["major_term_count"] = df["major_requirement"].apply(count_major_terms)

    # 3.6 核心工程特征：报录比 = 报名人数 / 招录人数
    valid = (df["employ_count"] > 0) & df["enrollment_count"].notna()
    df["report_ratio"] = np.where(
        valid, df["enrollment_count"] / df["employ_count"], np.nan)

    # 3.7 竞争激烈度分档（基于报录比）
    def heat_bucket(r):
        if pd.isna(r):
            return np.nan
        if r < 20:
            return "低竞争(<20)"
        if r < 50:
            return "中竞争(20-50)"
        if r < 100:
            return "高竞争(50-100)"
        return "极热(>=100)"
    df["heat_level"] = df["report_ratio"].apply(heat_bucket)

    # 3.8 是否 '若干'（招录人数未明确）
    df["employ_flexible"] = (df["employ_count_desc"].astype(str).str.strip()
                             == "若干").astype(int)

    return df


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    gongkao_dir = ensure_raw_extracted()
    raw = load_raw(gongkao_dir)
    df = clean(raw)

    out = os.path.join(CLEAN_DIR, "positions.parquet")
    df.to_parquet(out, index=False)
    print(f"[save] 已导出分析级数据集 -> {out}")

    # 数据质量小结
    print("\n===== 数据质量小结 =====")
    print(f"总记录数        : {len(df):,}")
    print(f"年份范围        : {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"国考/省考       : {df['exam_type'].value_counts().to_dict()}")
    print(f"覆盖省份数      : {df['province'].nunique()}")
    print(f"报录比可算比例  : {df['report_ratio'].notna().mean():.1%}")
    print(f"报录比中位数    : {df['report_ratio'].median():.1f}")


if __name__ == "__main__":
    main()
