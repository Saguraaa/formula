"""Paired forward-only localization of the official recognizer's answer errors."""

from collections import Counter
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from weight_editing import direct_ocr_edit as direct
from weight_editing.direct_ocr_edit import ev, prior, official

OUT = ev.ROOT / "runs/ocr-mechanism-diagnosis-v1"
DONOR = prior.O3 / "best_accuracy.pdparams"
PROTOCOL = dict(
    primary="normalized whole transcription; compare to same-runtime O0, also compare native endpoints to official saved TextRecognition rows",
    large="net recovery of at least 30% of all original errors on full validation; descriptive operational cutoff",
    splits="symptom discovery on 290 training pages; replicate interventions on 96 held-out training pages and all 2738 validation crops; no test data",
    donor="existing O3 selected on validation, trained on all training pages including the 96 held out from diagnosis; a localization control, not a new generalization result",
    factors=[
        "PPLCNetV4 backbone",
        "LightSVTR CTC encoder",
        "CTC classifier including bias",
    ],
    ablations="all eight O0/O3 module combinations; O3 BatchNorm means and variances only in O0; O3 backbone stem/blocks2/3/4/5/6 individually in otherwise O0; O3 backbone weights alone exclude the encoder and classifier",
    output_probes="use first recognized equals time step, baseline frame path fixed, for frames strictly to its right subtract 1,2,4 from blank logit or add 1,2,4 to all digit logits. Include blank -2 before equals as a positional control.",
    policy="all listed interventions reported; no updated production checkpoint; no label changes",
)


def load_rows():
    disc = ev.load_json(direct.OUT / "discovery.json")
    rows = prior.read_jsonl(direct.OUT / "baseline-train.jsonl")
    held = set(disc["held_pages"])
    assert len(rows) == 13206 and len({r["crop"] for r in rows}) == len(rows)
    return rows, held


def rhs_only(gt, text):
    a, eq, b = gt.partition("=")
    x, sep, y = text.partition("=")
    return bool(eq and sep and a == x and b != y)


def symptoms(rows):
    c = Counter()
    for r in rows:
        gt, text = r["expected"], r["ocr_normalized"]
        c["total"] += 1
        if gt == text:
            continue
        c["errors"] += 1
        if rhs_only(gt, text):
            c["only_rhs"] += 1
        elif (
            gt.count("=") == 1
            and text.count("=") == 1
            and gt.split("=")[1] == text.split("=")[1]
        ):
            c["only_lhs"] += 1
        elif gt.count("=") == 1 and text.count("=") == 1:
            c["both_sides"] += 1
        else:
            c["equals_or_other_structure"] += 1
    return dict(c)


def setup():
    OUT.mkdir(exist_ok=True)
    rows, held = load_rows()
    val = prior.read_jsonl(direct.OUT / "baseline-val.jsonl")
    assert len(val) == 2738 and not {r["page"] for r in val} & {r["page"] for r in rows}
    sources = [
        official.INITIAL_OCR,
        DONOR,
        prior.O3 / "export.yaml",
        direct.OUT / "baseline-train.jsonl",
        direct.OUT / "baseline-val.jsonl",
        direct.OUT / "discovery.json",
        ev.DATA / "split_manifest.json",
    ]
    ev.save_json(
        OUT / "protocol.json",
        dict(settings=PROTOCOL, sha256={str(p): ev.sha(p) for p in sources}),
    )
    ev.save_json(
        OUT / "symptoms.json",
        {
            k: symptoms(rs)
            for k, rs in [
                ("discovery", [r for r in rows if r["page"] not in held]),
                ("train-held", [r for r in rows if r["page"] in held]),
                ("val", val),
            ]
        },
    )


def make_models(rt):
    import gc

    def build(checkpoint, stats=None):
        rt.BEST = checkpoint
        net, _, _, _, _, decoder, _rank = rt.build()
        changed = []
        if stats is not None:
            for k, v in net.state_dict().items():
                if k.endswith(("._mean", "._variance")) and not np.array_equal(
                    v.numpy(), stats[k]
                ):
                    v.set_value(stats[k])
                    changed.append(k)
            if not changed:
                raise RuntimeError("No normalization buffers transplanted")
        fused = rt.frozen_export_graph(net)
        return fused, decoder.character, changed

    base, chars, _ = build(official.INITIAL_OCR)
    donor, chars3, _ = build(DONOR)
    assert chars == chars3
    stats = rt.paddle.load(str(DONOR), return_numpy=True)
    bn, _, changed = build(official.INITIAL_OCR, stats)
    del stats
    gc.collect()
    ev.save_json(OUT / "bn-transplant.json", changed)
    return base, donor, bn, chars


def scan(split):
    from weight_editing import train_weight_edit as rt

    pd = rt.paddle
    pd.set_device("gpu:0")
    base, donor, bn, chars = make_models(rt)
    rows, held = load_rows()
    ref_o3 = (
        prior.read_jsonl(prior.OUT / "train-baseline.jsonl")
        if split == "train-held"
        else prior.read_jsonl(prior.controlled.RESULTS / "O3/val.jsonl")
    )
    if split == "val":
        rows = prior.read_jsonl(direct.OUT / "baseline-val.jsonl")
    elif split != "train-held":
        raise ValueError(split)
    o3 = {r["crop"]: r["ocr_text"] for r in ref_o3}
    ds = [i for i, c in enumerate(chars) if ev.normalize(c) in list("0123456789")]
    eqids = [i for i, c in enumerate(chars) if ev.normalize(c) == "="]
    dindex = pd.to_tensor(ds, dtype="int64")
    ds = np.asarray(ds)
    stages = ("conv1", "blocks2", "blocks3", "blocks4", "blocks5", "blocks6")

    def lattice(logits):
        return pd.argmax(logits, axis=-1).numpy()

    def score(h, net):
        head = net.head.ctc_head.fc
        return rt.F.linear(h, head.weight, head.bias)

    def native(bb, x):
        out = []
        for name in stages:
            x = getattr(bb, name)(x)
            out.append(x)
        return out

    def pool(x):
        return rt.F.avg_pool2d(x, [3, 2])

    def read_logits(logits):
        indices = lattice(logits)
        return [direct.decode(row, chars) for row in indices]

    import json

    with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as stream, pd.no_grad():
        count = 0
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            active = [
                i for i, r in enumerate(batch) if split == "val" or r["page"] in held
            ]
            if not active:
                continue
            # Keep the input padding and width of the original official batch.
            arr, _ = rt.input_batch([r["crop"] for r in batch])
            x = pd.to_tensor(arr)
            nets = [base, donor]
            features = [native(net.backbone, x) for net in nets]
            texts = {}
            for b in (0, 1):
                for e in (0, 1):
                    h = nets[e].head.ctc_encoder(pool(features[b][-1]))
                    for c in (0, 1):
                        name = f"B{b*3}E{e*3}C{c*3}"
                        z = score(h, nets[c])
                        texts[name] = read_logits(z)
                        if name != "B0E0C0":
                            continue
                        ids = lattice(z)
                        blank = z[:, :, 0].numpy()
                        nb = z[:, :, 1:]
                        nbid = pd.argmax(nb, -1, keepdim=True)
                        nbscore = pd.take_along_axis(nb, nbid, -1).squeeze(-1).numpy()
                        nbid = nbid.squeeze(-1).numpy() + 1
                        digits = pd.index_select(z, dindex, axis=2)
                        di = pd.argmax(digits, -1, keepdim=True)
                        dscore = pd.take_along_axis(digits, di, -1).squeeze(-1).numpy()
                        did = ds[di.squeeze(-1).numpy()]
                        top = np.maximum(blank, nbscore)
                        eq = np.isin(ids, eqids)
                        first = np.where(eq.any(1), eq.argmax(1), ids.shape[1])
                        right = np.arange(ids.shape[1])[None, :] > first[:, None]
                        left = (
                            np.arange(ids.shape[1])[None, :] < first[:, None]
                        ) & eq.any(1)[:, None]
                        for mag in (1, 2, 4):
                            for prefix, mask, switched in [
                                (
                                    "blank",
                                    right,
                                    np.where(blank - mag >= nbscore, 0, nbid),
                                ),
                                (
                                    "digit",
                                    right,
                                    np.where(dscore + mag > top, did, ids),
                                ),
                            ]:
                                alt = np.where(mask, switched, ids)
                                texts[f"{prefix}-rhs-{mag}"] = [
                                    direct.decode(v, chars) for v in alt
                                ]
                        alt = np.where(
                            left, np.where(blank - 2 >= nbscore, 0, nbid), ids
                        )
                        texts["blank-lhs-2"] = [direct.decode(v, chars) for v in alt]
            for j, name in enumerate(stages):
                mid = getattr(donor.backbone, name)(x if not j else features[0][j - 1])
                for rest in stages[j + 1 :]:
                    mid = getattr(base.backbone, rest)(mid)
                texts["only-" + name] = read_logits(
                    score(base.head.ctc_encoder(pool(mid)), base)
                )
            texts["only-bn-stats"] = read_logits(
                score(bn.head.ctc_encoder(bn.backbone(x)), bn)
            )
            for i in active:
                r = batch[i]
                line = dict(
                    crop=r["crop"],
                    page=r["page"],
                    expected=r["expected"],
                    truth_grade=r["truth_grade"],
                    official=r["ocr_text"],
                    o3=o3[r["crop"]],
                    probes={k: v[i] for k, v in texts.items()},
                )
                stream.write(json.dumps(line, ensure_ascii=False) + "\n")
            count += len(active)
            if count // 500 != (count - len(active)) // 500:
                print(split, count, flush=True)
    assert count == (3258 if split == "train-held" else 2738)
    print(split, count, "complete", flush=True)


def detail(split):
    """Resolve blocks5 into spatial/channel mixers; include O0-only gain controls."""
    import json
    from weight_editing import train_weight_edit as rt

    pd = rt.paddle
    pd.set_device("gpu:0")

    def fused(path):
        rt.BEST = path
        net, _, _, _, _, decoder, _ = rt.build()
        return rt.frozen_export_graph(net), decoder.character

    base, chars = fused(official.INITIAL_OCR)
    donor, _ = fused(DONOR)
    rows, held = load_rows()
    if split == "val":
        rows = prior.read_jsonl(direct.OUT / "baseline-val.jsonl")
    elif split != "train-held":
        raise ValueError(split)
    b0 = base.backbone
    layers = b0.blocks5
    donors = donor.backbone.blocks5
    assert len(layers) == len(donors) == 3 and len(b0.blocks6) == 0
    head = base.head.ctc_head.fc

    def texts(v):
        feat = base.head.ctc_encoder(rt.F.avg_pool2d(v, [3, 2]))
        ids = pd.argmax(rt.F.linear(feat, head.weight, head.bias), -1).numpy()
        return [direct.decode(row, chars) for row in ids]

    with (OUT / f"{split}-detail.jsonl").open(
        "w", encoding="utf-8"
    ) as stream, pd.no_grad():
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            active = [
                i for i, r in enumerate(batch) if split == "val" or r["page"] in held
            ]
            if not active:
                continue
            arr, _ = rt.input_batch([r["crop"] for r in batch])
            v = b0.blocks4(b0.blocks3(b0.blocks2(b0.conv1(pd.to_tensor(arr)))))
            states = [v]
            for layer in layers:
                states.append(layer(states[-1]))
            outputs = {}
            for j, (layer, other) in enumerate(zip(layers, donors)):

                def complete(name, y):
                    for k in range(j + 1, 3):
                        y = layers[k](y)
                    outputs[f"b5.{j}.{name}"] = texts(y)

                def block(t, c, gain=1):
                    y = t.token_mixer(states[j])
                    delta = c.channel_mixer(y)
                    return y + delta * gain if layer.has_residual else delta * gain

                complete("all", other(states[j]))
                complete("spatial", block(other, layer))
                complete("channel", block(layer, other))
                # Effective inference convolutions include their folded BatchNorm.
                token = layer.token_mixer(states[j])
                for part in ("expand", "compress"):
                    expander = (
                        other if part == "expand" else layer
                    ).channel_mixer.expand
                    compressor = (
                        other if part == "compress" else layer
                    ).channel_mixer.compress
                    y = compressor(layer.channel_mixer.act(expander(token)))
                    complete(part, y + token if layer.has_residual else y)
                if layer.has_residual:
                    for gain in (0.5, 1.5):
                        complete(
                            f"official-residual-x{gain}", block(layer, layer, gain)
                        )
            for i in active:
                stream.write(
                    json.dumps(
                        dict(
                            crop=batch[i]["crop"],
                            probes={k: v[i] for k, v in outputs.items()},
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def spatial(split):
    """Swap only feature positions inferred from the original prediction."""
    import json
    from weight_editing import train_weight_edit as rt

    pd = rt.paddle
    pd.set_device("gpu:0")

    def fused(path):
        rt.BEST = path
        net, _, _, _, _, decoder, _ = rt.build()
        return rt.frozen_export_graph(net), decoder.character

    base, chars = fused(official.INITIAL_OCR)
    donor, _ = fused(DONOR)
    rows, held = load_rows()
    if split == "val":
        rows = prior.read_jsonl(direct.OUT / "baseline-val.jsonl")
    elif split != "train-held":
        raise ValueError(split)
    bb = base.backbone
    head = base.head.ctc_head.fc
    eqids = [i for i, c in enumerate(chars) if ev.normalize(c) == "="]

    def ids(feats):
        h = base.head.ctc_encoder(feats)
        return pd.argmax(rt.F.linear(h, head.weight, head.bias), -1).numpy()

    with (OUT / f"{split}-spatial.jsonl").open(
        "w", encoding="utf-8"
    ) as stream, pd.no_grad():
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            active = [
                i for i, r in enumerate(batch) if split == "val" or r["page"] in held
            ]
            if not active:
                continue
            arr, _ = rt.input_batch([r["crop"] for r in batch])
            shared = bb.blocks4(bb.blocks3(bb.blocks2(bb.conv1(pd.to_tensor(arr)))))
            x = rt.F.avg_pool2d(bb.blocks5(shared), [3, 2])
            y = rt.F.avg_pool2d(donor.backbone.blocks5(shared), [3, 2])
            seq = ids(x)
            eq = np.isin(seq, eqids)
            first = np.where(eq.any(1), eq.argmax(1), seq.shape[1])
            index = np.arange(seq.shape[1])[None, :]
            right = pd.to_tensor((index > first[:, None])[:, None, None, :])
            left = pd.to_tensor(
                ((index < first[:, None]) & eq.any(1)[:, None])[:, None, None, :]
            )
            variants = {
                "feature-rhs-O3": pd.where(right, y, x),
                "feature-lhs-O3": pd.where(left, y, x),
                "feature-rhs-x0.75": pd.where(right, x * 0.75, x),
                "feature-rhs-x1.25": pd.where(right, x * 1.25, x),
            }
            preds = {
                n: [direct.decode(v, chars) for v in ids(feats)]
                for n, feats in variants.items()
            }
            for i in active:
                stream.write(
                    json.dumps(
                        dict(
                            crop=batch[i]["crop"],
                            probes={k: v[i] for k, v in preds.items()},
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def summarize(split):
    from functools import lru_cache
    from rapidfuzz.distance import Levenshtein
    from arithmetic import judge

    @lru_cache(maxsize=None)
    def grade(s):
        return judge(s)["is_correct"]

    rows = prior.read_jsonl(OUT / f"{split}.jsonl")
    for part in ("detail", "spatial"):
        details = prior.read_jsonl(OUT / f"{split}-{part}.jsonl")
        for r, d in zip(rows, details, strict=True):
            assert r["crop"] == d["crop"]
            r["probes"].update(d["probes"])
    source, held = load_rows()
    expected_rows = (
        [r for r in source if r["page"] in held]
        if split == "train-held"
        else prior.read_jsonl(direct.OUT / "baseline-val.jsonl")
    )
    assert [r["crop"] for r in rows] == [r["crop"] for r in expected_rows]
    assert len(set(r["crop"] for r in rows)) == len(rows)
    parity = {}
    for field, key in (("official", "B0E0C0"), ("o3", "B3E3C3")):
        diffs = [r["crop"] for r in rows if r[field] != r["probes"][key]]
        normalized = [
            r["crop"]
            for r in rows
            if ev.normalize(r[field]) != ev.normalize(r["probes"][key])
        ]
        parity[field] = dict(
            raw_count=len(diffs), normalized_count=len(normalized), crops=normalized
        )
    result = {}
    for model in rows[0]["probes"]:
        count = Counter()
        repair_pages = set()
        regress_pages = set()
        for r in rows:
            gt, truth = r["expected"], r["truth_grade"]
            orig = ev.normalize(r["probes"]["B0E0C0"])
            new = ev.normalize(r["probes"][model])
            ok_before, ok_after = orig == gt, new == gt
            fixed = not ok_before and ok_after
            broken = ok_before and not ok_after
            if fixed:
                repair_pages.add(r["page"])
            if broken:
                regress_pages.add(r["page"])
            old_grade, new_grade = grade(orig), grade(new)
            count.update(
                total=1,
                base_errors=int(not ok_before),
                exact=int(ok_after),
                repair=int(fixed),
                regression=int(broken),
                repaired_original_rhs=int(fixed and rhs_only(gt, orig)),
                net=int(fixed) - int(broken),
                characters=len(gt),
                edits=Levenshtein.distance(gt, new),
                gradeable=int(truth is not None),
                grade_correct=int(truth is not None and truth is new_grade),
                false_accept=int(truth is False and new_grade is True),
                new_false_accept=int(
                    truth is False and old_grade is not True and new_grade is True
                ),
                removed_false_accept=int(
                    truth is False and old_grade is True and new_grade is not True
                ),
                undetermined=int(truth is not None and new_grade is None),
            )
        result[model] = dict(
            **count,
            repair_pages=len(repair_pages),
            regress_pages=len(regress_pages),
            net_error_recovery=count["net"] / count["base_errors"],
        )
    return dict(metrics=result, parity=parity)


def report():
    results = {s: summarize(s) for s in ("train-held", "val")}
    ev.save_json(OUT / "comparison.json", results)
    symptoms_by_split = ev.load_json(OUT / "symptoms.json")
    val = results["val"]["metrics"]
    headline = val["only-blocks5"]
    rhs = val["feature-rhs-O3"]
    lhs = val["feature-lhs-O3"]
    channel = val["b5.0.channel"]
    lines = [
        "# 官方 PP-OCRv6 的算式答案识别故障定位",
        "",
        f"验证集的 164 个原始全文错误中有 150 个只错在等号右侧。仅交换 O3 的 blocks5，验证集修复 {headline['repair']} 题，退步 {headline['regression']} 题，覆盖 {headline['repair_pages']} 个出现修复的页面，原错误净减少 {headline['net_error_recovery']:.1%}。只换 blocks5.0 的通道混合器即可净修复 {channel['net']} 题。这里交换的 O3 模块含已经微调过的参数。",
        "",
        f"保持官方编码器和分类器，仅在预测等号之后的位置替换 blocks5 池化后的特征，可净修复 {rhs['net']} 题；仅替换等号之前的位置净变化 {lhs['net']:+d} 题。位置对照以原有 CTC 等号时间步近似空间位置，两侧覆盖长度不同。涉及大范围错误的原因已定位到视觉特征表示，仍无法据此将其归因于某个唯一的神经元或笔画。",
        "",
        "例如 132_0027.jpg 的学生答案为 310。官方输出 800-500=3/0，只换 blocks5.0 通道混合器后输出 800-500=310，保留了学生实际写错的答案。",
        "",
        "统计单位为同一道题。原始官方 OCR 使用项目统一规范化。等号两边以第一个等号为界；只有右边错误表示题干和等号都读对，答案文本不同。",
        "",
        "| 范围 | 题数 | 全文错 | 仅右侧错 | 比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for s, d in symptoms_by_split.items():
        lines.append(
            f"| {s} | {d['total']} | {d['errors']} | {d['only_rhs']} | {d['only_rhs']/d['errors']:.2%} |"
        )
    lines += [
        "",
        "B 表示 PPLCNetV4 骨干，E 表示 LightSVTR CTC 编码器，C 表示字符分类层。0 使用官方权重，3 使用历史 O3。only- 组从 O3 取一个阶段或归一化统计，其余使用官方模型。数值干预在官方模型原来读出的第一个等号处定位；无等号的行保持原输出。不使用正确答案决定模型输出。所有统计均相对于同一运行环境的全官方模型。",
        "",
    ]
    for s, info in results.items():
        lines.extend(
            [
                f"## {s}",
                "",
                f"共 {info['metrics']['B0E0C0']['total']} 张。与已存官方正式推理和 O3 正式推理的规范化输出差异分别为 {info['parity']['official']['normalized_count']} 和 {info['parity']['o3']['normalized_count']} 张。",
                "",
                "| 干预 | 全文正确 | 修复 | 其中原仅右侧错 | 退步 | 原错误净减少率 | 判题正确 | 错题判对 | 新增错题判对 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, c in info["metrics"].items():
            lines.append(
                f"| {name} | {c['exact']} | {c['repair']} | {c['repaired_original_rhs']} | {c['regression']} | {c['net_error_recovery']:.1%} | {c['grade_correct']} | {c['false_accept']} | {c['new_false_accept']} |"
            )
        lines.append("")
    lines += [
        "第二阶段根据训练留出页的结果选定 blocks5，逐个交换三个子块的空间混合器、通道混合器、扩展或压缩投影，各组均用官方模型其余部件。空间混合器在子块 1 也含 SE。扩展与压缩投影使用融合 BN 的推理卷积。official-residual 组只调整官方模型分支的激活幅度。详见 detail-protocol.json。",
        "",
        "O3 使用了全部训练页，并在历史验证集上选过轮次。其模块交换用于定位机制，不能作为无梯度独立训练结果或盲测成绩。train-held 仅表示这些页对本次症状发现留出。此诊断使用训练和验证划分，保留项目原标签。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup()
    elif sys.argv[1] == "report":
        report()
    elif sys.argv[1] == "detail":
        detail(sys.argv[2])
    elif sys.argv[1] == "spatial":
        spatial(sys.argv[2])
    else:
        scan(sys.argv[1])
