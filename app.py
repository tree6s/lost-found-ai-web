from __future__ import annotations

import os
import socket
import uuid
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import qrcode
import streamlit as st
from PIL import Image

# AI image classification module from this project
try:
    from classify_item import predict_topk
except Exception as e:  # Keep the page alive even if TensorFlow/model imports fail online
    predict_topk = None
    CLASSIFY_IMPORT_ERROR = str(e)
else:
    CLASSIFY_IMPORT_ERROR = ""

# =========================
# Basic configuration
# =========================
st.set_page_config(page_title="校园失物招领系统（AI线上版）", layout="centered")

DATA_FILE = "lost_items.csv"
IMAGES_DIR = "images"
TEMP_DIR = "temp_uploads"

MODEL_PATH = "models/keras_model.h5"
LABELS_PATH = "models/labels.txt"
METADATA_PATH = "models/metadata.json"

# Keep this list consistent with the labels used when training the Teachable Machine model.
# If your labels.txt exists, the app will try to load model labels and use them as category options.
DEFAULT_CATEGORIES = ["水杯", "文具", "电子产品", "衣物", "证件/卡片", "其他"]

COLUMNS = [
    "name",
    "category",
    "location",
    "found_time",
    "image_path",
    "ai_predicted_category",
    "ai_confidence",
]

# =========================
# File and directory helpers
# =========================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_runtime_dirs() -> None:
    ensure_dir(IMAGES_DIR)
    ensure_dir(TEMP_DIR)


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV safely. Empty/missing/broken files return an empty table."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=COLUMNS)
    except Exception:
        # For classroom use, avoid crashing the whole app because of one corrupted CSV.
        df = pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def save_uploaded_image(uploaded_file: Any, target_dir: str = IMAGES_DIR) -> str:
    """Save an uploaded image and return its relative path."""
    if uploaded_file is None:
        return ""

    ensure_dir(target_dir)
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        ext = ".jpg"

    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = Path(target_dir) / filename
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(save_path)


# =========================
# Data helpers
# =========================
def load_items() -> pd.DataFrame:
    return safe_read_csv(DATA_FILE)


def save_df(df: pd.DataFrame) -> None:
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMNS]
    df.to_csv(DATA_FILE, index=False, encoding="utf-8")


def save_item(item: dict) -> None:
    df = load_items()
    df = pd.concat([df, pd.DataFrame([item])], ignore_index=True)
    save_df(df)


def clear_all_items() -> None:
    save_df(pd.DataFrame(columns=COLUMNS))
    img_dir = Path(IMAGES_DIR)
    if img_dir.exists():
        for fp in img_dir.iterdir():
            if fp.is_file():
                try:
                    fp.unlink()
                except Exception:
                    pass


# =========================
# AI helpers
# =========================
def model_ready() -> bool:
    return (
        predict_topk is not None
        and Path(MODEL_PATH).exists()
        and (Path(LABELS_PATH).exists() or Path(METADATA_PATH).exists())
    )


@st.cache_data(show_spinner=False)
def load_category_options() -> list[str]:
    """Use labels.txt as category options when available; otherwise use defaults."""
    labels_path = Path(LABELS_PATH)
    if labels_path.exists():
        labels: list[str] = []
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                parts = text.split(maxsplit=1)
                labels.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else text)
        if labels:
            return labels
    return DEFAULT_CATEGORIES


def run_ai_prediction(temp_image_path: str) -> tuple[dict | None, str | None]:
    if predict_topk is None:
        return None, f"AI 分类模块导入失败：{CLASSIFY_IMPORT_ERROR}"
    try:
        result = predict_topk(
            image_path=temp_image_path,
            model_path=MODEL_PATH,
            labels_path=LABELS_PATH if Path(LABELS_PATH).exists() else None,
            metadata_path=METADATA_PATH if Path(METADATA_PATH).exists() else None,
            topk=3,
        )
        return result, None
    except Exception as e:
        return None, str(e)


# =========================
# QR helpers
# =========================
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def build_possible_urls(port: int = 8501) -> dict[str, str]:
    local_ip = get_local_ip()
    urls = {
        "本机地址（只适合当前电脑）": f"http://localhost:{port}",
        "局域网地址（同一 Wi-Fi 下可用）": f"http://{local_ip}:{port}",
    }

    # Optional: set this in Streamlit Cloud secrets or environment variables.
    public_url = ""
    try:
        public_url = str(st.secrets.get("APP_PUBLIC_URL", "")).strip()
    except Exception:
        public_url = ""
    public_url = public_url or os.environ.get("APP_PUBLIC_URL", "").strip()
    if public_url:
        urls = {"正式线上地址（公网可访问）": public_url, **urls}
    return urls


def make_qr_image(url: str) -> Image.Image:
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    if hasattr(img, "get_image"):
        img = img.get_image()
    return img.convert("RGB")


# =========================
# Page 1: QR entrance
# =========================
def qr_page() -> None:
    st.header("📱 二维码入口")
    st.write("把网页链接变成二维码，方便别人用手机扫码进入失物招领系统。")

    urls = build_possible_urls(port=8501)

    st.subheader("1. 选择要生成二维码的链接")
    st.caption("本地/局域网链接只适合课堂调试；真正线上教学需要使用部署后的公网链接。")

    default_url = next(iter(urls.values()))
    for label, url in urls.items():
        st.write(f"**{label}**：`{url}`")

    share_url = st.text_input(
        "二维码链接",
        value=default_url,
        help="部署到 Streamlit Cloud 后，把正式的 https 链接粘贴到这里再生成二维码。",
    )

    if share_url:
        qr_img = make_qr_image(share_url)
        st.subheader("2. 二维码")
        st.image(qr_img, caption="用手机扫码打开网页", width=260)

        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ 下载二维码图片",
            data=buf.getvalue(),
            file_name="lost_found_qr.png",
            mime="image/png",
        )

    st.subheader("3. 手机端测试流程")
    st.markdown(
        """
1. 用手机扫码打开网页；  
2. 进入“登记失物”；  
3. 上传图片，观察 AI 推荐类别；  
4. 填写名称和地点后提交；  
5. 进入“查询失物”，确认记录是否出现。  
        """
    )


# =========================
# Page 2: Add item with AI suggestion
# =========================
def add_item_page() -> None:
    categories = load_category_options()

    st.header("🧾 登记失物（AI 推荐类别）")
    st.caption("上传图片后，AI 会推荐类别；最终类别仍然可以由用户手动修改。")

    uploaded_img = st.file_uploader(
        "上传物品图片（建议上传，AI 才能推荐类别）",
        type=["png", "jpg", "jpeg", "webp"],
    )

    predicted_category: str | None = None
    predicted_score: float | None = None
    top_results: list[dict] = []

    if uploaded_img is not None:
        st.image(uploaded_img, caption="待识别图片", width=260)

        if model_ready():
            temp_path = save_uploaded_image(uploaded_img, target_dir=TEMP_DIR)
            with st.spinner("AI 正在识别图片..."):
                result, error = run_ai_prediction(temp_path)

            try:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink()
            except Exception:
                pass

            if error:
                st.warning(f"AI 预测失败：{error}")
            elif result is not None:
                predicted_category = str(result["best_label"])
                predicted_score = float(result["best_score"])
                top_results = list(result.get("top_results", []))

                st.success(f"AI 推荐类别：{predicted_category}（置信度 {predicted_score:.1%}）")
                with st.expander("查看 Top-3 预测结果"):
                    for item in top_results:
                        st.write(f"{item['rank']}. {item['label']} —— {item['score']:.1%}")
        else:
            if CLASSIFY_IMPORT_ERROR:
                st.warning(f"AI 分类模块暂不可用：{CLASSIFY_IMPORT_ERROR}")
            else:
                st.info("未检测到模型文件，当前以普通手动登记模式运行。")

    default_index = 0
    if predicted_category in categories:
        default_index = categories.index(predicted_category)

    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("物品名称", placeholder="例如：黑色水杯")
        category = st.selectbox("类别", categories, index=default_index)
        location = st.text_input("拾到地点", placeholder="例如：教学楼三楼走廊")
        found_time = st.date_input("拾到时间", value=date.today())
        submitted = st.form_submit_button("✅ 提交登记")

    if submitted:
        if not name.strip() or not location.strip():
            st.warning("请至少填写物品名称和拾到地点。")
            return

        image_path = save_uploaded_image(uploaded_img, target_dir=IMAGES_DIR)
        item = {
            "name": name.strip(),
            "category": category,
            "location": location.strip(),
            "found_time": str(found_time),
            "image_path": image_path,
            "ai_predicted_category": predicted_category or "",
            "ai_confidence": float(predicted_score) if predicted_score is not None else "",
        }
        save_item(item)
        st.success("已登记！")
        st.caption("刚登记的记录：")
        st.write(item)
        if image_path and Path(image_path).exists():
            st.image(image_path, caption="已上传图片", width=240)


# =========================
# Page 3: Search items
# =========================
def search_page() -> None:
    categories = load_category_options()

    st.header("🔎 查询失物")
    df = load_items()

    if df.empty:
        st.info("目前还没有登记任何失物。")
        return

    st.subheader("筛选条件")
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.selectbox("按类别筛选", ["全部"] + categories)
    with col2:
        keyword = st.text_input("关键词搜索（名称/地点）", placeholder="例如：水杯 / 三楼")

    filtered = df.copy()
    if category_filter != "全部":
        filtered = filtered[filtered["category"] == category_filter]
    if keyword:
        kw = keyword.strip()
        filtered = filtered[
            filtered["name"].astype(str).str.contains(kw, case=False, na=False)
            | filtered["location"].astype(str).str.contains(kw, case=False, na=False)
        ]

    st.subheader("数据管理")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        if st.button("🗑️ 清空全部数据"):
            clear_all_items()
            st.success("已清空所有记录。")
            st.rerun()

    with col_b:
        filtered_with_idx = filtered.reset_index()
        if not filtered_with_idx.empty:
            options = [
                f"[{row['index']}] {row['name']}｜{row['category']}｜{row['location']}｜{row['found_time']}"
                for _, row in filtered_with_idx.iterrows()
            ]
            selected = st.selectbox("选择要删除的一条记录（基于当前筛选结果）", options)
            if st.button("❌ 删除所选记录"):
                raw_idx = int(selected.split("]")[0].strip("["))
                df_all = load_items()
                try:
                    img_path = str(df_all.loc[raw_idx, "image_path"])
                    if img_path and Path(img_path).exists():
                        Path(img_path).unlink()
                except Exception:
                    pass
                df_all = df_all.drop(index=raw_idx).reset_index(drop=True)
                save_df(df_all)
                st.success("已删除该条记录。")
                st.rerun()
        else:
            st.caption("当前筛选结果为空，暂无可删除的记录。")

    st.subheader("结果列表")
    if filtered.empty:
        st.warning("没有找到匹配的记录。")
        return

    for i, row in filtered.reset_index(drop=True).iterrows():
        img_col, text_col = st.columns([1, 3])
        with img_col:
            img_path = str(row.get("image_path", "") or "")
            if img_path and Path(img_path).exists():
                st.image(img_path, width=120)
            else:
                st.caption("无图片")
        with text_col:
            st.write(
                f"{i + 1}. **{row['name']}** ｜{row['category']} ｜{row['location']} ｜{row['found_time']}"
            )
            ai_cat = str(row.get("ai_predicted_category", "") or "").strip()
            ai_conf = row.get("ai_confidence", "")
            if ai_cat:
                try:
                    ai_conf_text = f"{float(ai_conf):.1%}"
                except Exception:
                    ai_conf_text = str(ai_conf)
                st.caption(f"AI 推荐：{ai_cat}（{ai_conf_text}）")

    with st.expander("查看表格形式"):
        st.dataframe(filtered, width="stretch")


# =========================
# Main entrance
# =========================
def main() -> None:
    ensure_runtime_dirs()

    st.title("📦 校园失物招领系统（AI线上版）")
    st.caption("功能：手机扫码入口 + 图片上传 + AI 推荐类别 + 登记查询。")

    page = st.sidebar.radio("导航", ["二维码入口", "登记失物", "查询失物"])

    if page == "二维码入口":
        qr_page()
    elif page == "登记失物":
        add_item_page()
    else:
        search_page()


if __name__ == "__main__":
    main()
