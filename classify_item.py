from __future__ import annotations

"""
升级版图片分类脚本：
1. 可在命令行中对单张图片做预测
2. 可被 Streamlit 网页直接 import 调用
3. 优先读取 labels.txt；若不存在，可退回读取 metadata.json

推荐目录结构：
models/
  keras_model.h5
  labels.txt           # 可选
  metadata.json        # 可选（Teachable Machine 导出时常见）
images/
  test.jpg

安装依赖：
pip install tensorflow pillow numpy

示例运行：
python classify_item.py --image images/test.jpg
python classify_item.py --image images/test.jpg --model models/keras_model.h5 --labels models/labels.txt
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


DEFAULT_MODEL_PATH = Path("models/keras_model.h5")
DEFAULT_LABELS_PATH = Path("models/labels.txt")
DEFAULT_METADATA_PATH = Path("models/metadata.json")


@lru_cache(maxsize=2)
def load_model_cached(model_path: str) -> tf.keras.Model:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型文件: {path}")

    class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D,
        },
    )


def load_labels_from_txt(labels_path: str | Path) -> List[str]:
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到标签文件: {path}")

    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(text)

    if not labels:
        raise ValueError("labels.txt 为空，无法读取类别名。")
    return labels


def load_labels_from_metadata(metadata_path: str | Path) -> List[str]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 metadata.json: {path}")

    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    for key in ["labels", "wordLabels", "classes"]:
        value = meta.get(key)
        if isinstance(value, list) and value:
            return [str(x).strip() for x in value]

    raise ValueError("metadata.json 中未找到 labels / wordLabels / classes。")



def load_labels(
    labels_path: str | Path | None = DEFAULT_LABELS_PATH,
    metadata_path: str | Path | None = DEFAULT_METADATA_PATH,
) -> List[str]:
    if labels_path is not None and Path(labels_path).exists():
        return load_labels_from_txt(labels_path)
    if metadata_path is not None and Path(metadata_path).exists():
        return load_labels_from_metadata(metadata_path)
    raise FileNotFoundError(
        "既找不到 labels.txt，也找不到 metadata.json。请至少提供其中一个。"
    )



def get_input_size(model: tf.keras.Model) -> Tuple[int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 4:
        raise ValueError(f"无法识别模型输入形状: {input_shape}")
    height = int(input_shape[1])
    width = int(input_shape[2])
    return width, height



def preprocess_image(image_path: str | Path, target_size: Tuple[int, int]) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到图片文件: {path}")

    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)

    # Teachable Machine 常见做法：缩放到 [-1, 1]
    image_array = (image_array / 127.5) - 1.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array



def _to_probabilities(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 1:
        raise ValueError(f"模型输出形状异常: {preds.shape}")

    total = float(np.sum(preds))
    if np.any(preds < 0) or abs(total - 1.0) > 1e-3:
        return tf.nn.softmax(preds).numpy()
    return preds



def predict_topk(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    labels_path: str | Path | None = DEFAULT_LABELS_PATH,
    metadata_path: str | Path | None = DEFAULT_METADATA_PATH,
    topk: int = 3,
) -> Dict[str, Any]:
    model = load_model_cached(str(model_path))
    labels = load_labels(labels_path=labels_path, metadata_path=metadata_path)
    target_size = get_input_size(model)
    x = preprocess_image(image_path, target_size)
    raw_preds = model.predict(x, verbose=0)[0]
    probs = _to_probabilities(raw_preds)

    if len(probs) != len(labels):
        raise ValueError(
            f"模型输出类别数为 {len(probs)}，但标签数量为 {len(labels)}，数量不一致。"
        )

    topk = max(1, min(int(topk), len(labels)))
    top_indices = np.argsort(probs)[::-1][:topk]
    top_results = [
        {
            "rank": rank,
            "index": int(idx),
            "label": labels[int(idx)],
            "score": float(probs[int(idx)]),
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]

    best = top_results[0]
    return {
        "best_label": best["label"],
        "best_score": best["score"],
        "labels": labels,
        "probabilities": probs,
        "top_results": top_results,
        "input_size": target_size,
    }



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Teachable Machine 导出的 TensorFlow/Keras 模型对单张图片分类"
    )
    parser.add_argument("--image", required=True, help="待预测图片路径，例如 images/test.jpg")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="模型文件路径")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="标签文件路径")
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA_PATH),
        help="metadata.json 路径；当 labels.txt 不存在时可自动读取",
    )
    parser.add_argument("--topk", type=int, default=3, help="显示前 k 个预测结果")
    return parser



def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    result = predict_topk(
        image_path=args.image,
        model_path=args.model,
        labels_path=args.labels if args.labels else None,
        metadata_path=args.metadata if args.metadata else None,
        topk=args.topk,
    )

    print("=" * 60)
    print(f"图片: {args.image}")
    print(f"预测结果: {result['best_label']}")
    print(f"置信度: {result['best_score']:.4f}")
    print("-" * 60)
    print(f"Top-{len(result['top_results'])} 结果：")
    for item in result["top_results"]:
        print(f"{item['rank']}. {item['label']:<20} {item['score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()