import os
import json

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon


# ============ 1. 加载模型 & 字典 ============

def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    # PP-OCR 字典约定：index 0 留给 CTC blank，最后一位常是空格
    return ["blank"] + lines + [" "]


class OrtSession:
    def __init__(self, model_path, use_gpu=False):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def run(self, img: np.ndarray):
        return self.sess.run(None, {self.input_name: img})


# ============ 2. 检测：预处理 ============

def det_resize(img, limit_side_len=960):
    h, w = img.shape[:2]
    ratio = 1.0
    if max(h, w) > limit_side_len:
        ratio = limit_side_len / max(h, w)
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    resized = cv2.resize(img, (resize_w, resize_h))
    ratio_h, ratio_w = resize_h / h, resize_w / w
    return resized, ratio_h, ratio_w


def det_normalize(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0).astype(np.float32)


def det_preprocess(img_bgr, limit_side_len=960):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized, ratio_h, ratio_w = det_resize(img_rgb, limit_side_len)
    inp = det_normalize(resized)
    return inp, ratio_h, ratio_w


# ============ 3. 检测：后处理（DB postprocess） ============

class DBPostProcess:
    def __init__(self, thresh=0.3, box_thresh=0.6, max_candidates=1000,
                 unclip_ratio=1.5, min_size=3):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def __call__(self, pred_map, ratio_h, ratio_w, src_h, src_w):
        prob_map = pred_map[0, 0]
        bitmap = (prob_map > self.thresh).astype(np.uint8)

        contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scores = []
        for contour in contours[: self.max_candidates]:
            if cv2.contourArea(contour) < self.min_size:
                continue
            box, min_side = self._get_min_box(contour)
            if min_side < self.min_size:
                continue
            score = self._box_score(prob_map, contour)
            if score < self.box_thresh:
                continue
            box = self._unclip(box)
            if box is None:
                continue
            box, min_side = self._get_min_box(box.reshape(-1, 1, 2))
            if min_side < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(box[:, 0] / ratio_w, 0, src_w)
            box[:, 1] = np.clip(box[:, 1] / ratio_h, 0, src_h)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return boxes, scores

    def _get_min_box(self, contour):
        rect = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(rect), key=lambda p: p[0])
        idx1, idx2 = (0, 1) if points[0][1] < points[1][1] else (1, 0)
        idx3, idx4 = (2, 3) if points[2][1] < points[3][1] else (3, 2)
        box = np.array([points[idx1], points[idx3], points[idx4], points[idx2]])
        return box, min(rect[1])

    def _box_score(self, prob_map, contour):
        h, w = prob_map.shape
        xmin, xmax = np.clip([contour[:, :, 0].min(), contour[:, :, 0].max()], 0, w - 1)
        ymin, ymax = np.clip([contour[:, :, 1].min(), contour[:, :, 1].max()], 0, h - 1)
        mask = np.zeros((int(ymax - ymin) + 1, int(xmax - xmin) + 1), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= int(xmin)
        shifted[:, :, 1] -= int(ymin)
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        return prob_map[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1][mask == 1].mean()

    def _unclip(self, box):
        poly = Polygon(box)
        if poly.area == 0 or poly.length == 0:
            return None
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        return np.array(expanded[0])


# ============ 4. 识别：预处理（按 box 裁剪 + resize 到统一高度） ============

def get_rotate_crop(img, box):
    box = box.astype(np.float32)
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
        crop = np.rot90(crop, k=3)
    return crop


def rec_preprocess(crops, img_h=48):
    max_ratio = max(c.shape[1] / c.shape[0] for c in crops) if crops else 1.0
    resize_w = int(img_h * max_ratio)
    batch = []
    for crop in crops:
        h, w = crop.shape[:2]
        ratio = w / h
        new_w = min(int(img_h * ratio), resize_w)
        resized = cv2.resize(crop, (max(new_w, 1), img_h))
        padded = np.zeros((img_h, resize_w, 3), dtype=np.uint8)
        padded[:, :resized.shape[1], :] = resized
        img = padded.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        batch.append(img)
    return np.stack(batch).astype(np.float32)


# ============ 5. 识别：CTC 解码 ============

def ctc_decode(preds, dict_list):
    texts = []
    for pred in preds:
        idx = pred.argmax(axis=1)
        conf = pred.max(axis=1)
        chars, scores = [], []
        last = -1
        for i, c in enumerate(idx):
            if c != 0 and c != last:
                chars.append(dict_list[c])
                scores.append(conf[i])
            last = c
        text = "".join(chars)
        score = float(np.mean(scores)) if scores else 0.0
        texts.append((text, score))
    return texts


# ============ 6. 按阅读顺序排序 + 拼整段文本 ============

def _box_center_y(box):
    return np.array(box)[:, 1].mean()

def _box_top_y(box):
    return np.array(box)[:, 1].min()

def _box_left_x(box):
    return np.array(box)[:, 0].min()

def _box_height(box):
    pts = np.array(box)
    return pts[:, 1].max() - pts[:, 1].min()


def sort_boxes_reading_order(results, y_thresh_ratio=0.6):
    """先按行分组（y 接近算同一行），行内再按 x 从左到右排"""
    if not results:
        return results

    items = sorted(results, key=lambda r: _box_top_y(r["box"]))

    lines = []
    for item in items:
        placed = False
        cy = _box_center_y(item["box"])
        for line in lines:
            line_cy = np.mean([_box_center_y(r["box"]) for r in line])
            line_h = np.mean([_box_height(r["box"]) for r in line])
            if abs(cy - line_cy) < line_h * y_thresh_ratio:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    lines.sort(key=lambda line: np.mean([_box_center_y(r["box"]) for r in line]))
    for line in lines:
        line.sort(key=lambda r: _box_left_x(r["box"]))

    return [item for line in lines for item in line]


def build_full_text(ordered_results, line_sep="\n"):
    return line_sep.join(r["text"] for r in ordered_results if r["text"].strip())


# ============ 7. 保存检测可视化图 ============

def save_det_vis(img_bgr, boxes, texts=None, scores=None, save_path="vis_det.jpg",
                  color=(0, 255, 0), thickness=2):
    vis = img_bgr.copy()
    for i, box in enumerate(boxes):
        pts = np.array(box).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
        label = None
        if texts is not None:
            label = texts[i]
        elif scores is not None:
            label = f"{scores[i]:.2f}"
        if label is not None:
            x, y = np.array(box)[0]
            cv2.putText(vis, str(label), (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(save_path, vis)
    return vis


# ============ 8. 保存结果 ============

def save_results_json(results, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_results_txt(results, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['text']}\t{r['score']:.4f}\n")


def save_full_text(full_text, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_text)


# ============ 9. 完整 pipeline ============

def ocr(img_path, det_model_path, rec_model_path, dict_path,
        use_gpu=False, save_dir="output"):
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    dict_list = load_dict(dict_path)
    det_sess = OrtSession(det_model_path, use_gpu)
    rec_sess = OrtSession(rec_model_path, use_gpu)
    db_post = DBPostProcess()

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"cannot read image: {img_path}")
    src_h, src_w = img_bgr.shape[:2]

    # --- det ---
    det_inp, ratio_h, ratio_w = det_preprocess(img_bgr)
    det_out = det_sess.run(det_inp)[0]
    prob_map = det_out if det_out.shape[1] == 1 else det_out[:, 0:1]
    boxes, det_scores = db_post(prob_map, ratio_h, ratio_w, src_h, src_w)

    det_only_vis_path = os.path.join(save_dir, f"{img_name}_det_only.jpg")
    save_det_vis(img_bgr, boxes, scores=det_scores, save_path=det_only_vis_path)
    print(f"[saved] det-only vis -> {det_only_vis_path}")

    # --- rec ---
    crops = [get_rotate_crop(img_bgr, box.astype(np.float32)) for box in boxes]
    results = []
    if crops:
        rec_inp = rec_preprocess(crops)
        rec_out = rec_sess.run(rec_inp)[0]
        decoded = ctc_decode(rec_out, dict_list)
        for box, (text, score) in zip(boxes, decoded):
            results.append({"box": box.tolist(), "text": text, "score": score})

    # --- 按阅读顺序排序 + 拼整段文本（这是你要的最终结果） ---
    ordered_results = sort_boxes_reading_order(results)
    full_text = build_full_text(ordered_results)

    # --- 可视化（带文字标注，非 ascii 用分数兜底显示，避免中文乱码）---
    final_vis_path = os.path.join(save_dir, f"{img_name}_result.jpg")
    labels = [r["text"] if r["text"].isascii() else f"{r['score']:.2f}" for r in ordered_results]
    save_det_vis(img_bgr, [r["box"] for r in ordered_results], texts=labels, save_path=final_vis_path)
    print(f"[saved] final vis -> {final_vis_path}")

    # --- 保存所有结果 ---
    json_path = os.path.join(save_dir, f"{img_name}_result.json")
    txt_path = os.path.join(save_dir, f"{img_name}_result.txt")
    full_text_path = os.path.join(save_dir, f"{img_name}_full_text.txt")

    save_results_json(ordered_results, json_path)
    save_results_txt(ordered_results, txt_path)
    save_full_text(full_text, full_text_path)

    print(f"[saved] per-box json -> {json_path}")
    print(f"[saved] per-box txt  -> {txt_path}")
    print(f"[saved] full text    -> {full_text_path}")

    return {
        "per_box_results": ordered_results,
        "full_text": full_text,
    }


if __name__ == "__main__":
    ckpt_p = "checkpoints/PaddleOCRv6"
    output = ocr(
        # img_path="data/images/book_rot180.jpg",
        # img_path="data/images/test_ocr_page2.png",
        img_path="data/images/image.png",
        det_model_path=os.path.join(ckpt_p, "pp-ocrv6_small_det.onnx"),
        rec_model_path=os.path.join(ckpt_p, "pp-ocrv6_small_rec.onnx"),
        dict_path=os.path.join(ckpt_p, "ppocrv6_dict.txt"),
        use_gpu=False,
        # use_gpu=True,
        save_dir="output",
    )

    print("\n===== FINAL TEXT =====")
    print(output["full_text"])