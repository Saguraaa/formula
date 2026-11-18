"""Forward-only direct editing of the official PP-OCRv6 character projection."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

import evaluate_controlled_ocr as ev
from weight_editing import official_weight_edit as official
from weight_editing import weight_edit as prior

OUT = ev.ROOT / "runs/direct-ocr-edit-v1"
CONFIG = dict(
    seed=42,
    strengths=[0.25, 0.5, 1.0],
    min_error_pages=4,
    target_alphabet="0123456789.+-*/=()",
    validation_rule="exact increases, edits and wrong_accepted do not increase",
    split="seed 42 shuffle sorted fixed train pages; last quarter reserved for intervention checks",
    target_rule="most discovery pages with isolated normalized substitution; then examples, lexicographic pair; both normalized symbols in alphabet and distinct; correct controls for both",
    alignment="greedy CTC nonblank runs, peak logit over winning run; require per-token normalization concatenates to normalized output",
    direction="unit(mean error activation - mean correctly read rival activation), one mean per page then equal page weights",
    strength_unit="median positive target-to-rival deficit divided by twice median positive error projection, using the ratio per error",
    checks="positive net exact repairs in held-out TRAIN pages and better than norm-matched random direction at the same strength; strongest net gain then fewer regressions then smallest strength; validate only locked strength",
    view="inspect discovery training error crops before computing direction; record ambiguous exclusions",
    params="CTC output weight, two actual dictionary columns; all other tensors frozen; no gradients, no optimizer",
)
BASE = prior.controlled.RESULTS / "O0"


def form(split):
    annotations, splits = prior.annotations_and_split()
    manifest = ev.load_json(prior.controlled.OUT / "manifest.json")
    prior.verify_labels(split, annotations, splits, manifest)
    if split == "train":
        if (
            official.old.model_hashes(official.OFFICIAL_OCR)
            != ev.load_json(BASE / "summary.json")["model_sha256"]
        ):
            raise ValueError("Official export hash mismatch")
        sources = [
            official.INITIAL_OCR,
            prior.O3 / "export.yaml",
            prior.SOURCE / "ppocr/utils/dict/ppocrv6_dict.txt",
            ev.DATA / "split_manifest.json",
            ev.ANNOTATIONS,
            prior.controlled.OUT / "train.txt",
            prior.controlled.OUT / "val.txt",
        ]
        if (
            ev.sha(official.INITIAL_OCR)
            != ev.load_json(prior.controlled.OUT / "initial-weight-verification.json")[
                "initial_sha256"
            ]
        ):
            raise ValueError("Official training checkpoint mismatch")
        ev.save_json(
            OUT / "protocol.json",
            dict(
                config=CONFIG,
                sources={str(p): ev.sha(p) for p in sources},
                official_export=prior.model_hashes(official.OFFICIAL_OCR),
            ),
        )
    rec = prior.recognizer(official.OFFICIAL_OCR)
    rows = prior.evaluate_split(rec, split, annotations, splits)
    hashes = (
        dict(ev.records(prior.controlled.OUT / "O1/train-image-sha256.tsv"))
        if split == "train"
        else {}
    )
    for r in rows:
        r["sha256"] = ev.sha(ev.OCR / r["crop"])
        if split == "train" and r["sha256"] != hashes[r["crop"]]:
            raise ValueError(r["crop"])
    ev.save_jsonl(OUT / f"baseline-{split}.jsonl", rows)
    if split == "val":
        ref = prior.read_jsonl(BASE / "val.jsonl")
        if [(r["crop"], r["ocr_text"]) for r in rows] != [
            (r["crop"], r["ocr_text"]) for r in ref
        ]:
            raise ValueError("O0 validation decode did not reproduce")
    print(split, ev.summarize(rows), flush=True)


def tokens(raw):
    """Align positions after normalization, rejecting length-changing glyphs."""
    mapped = [(i, ev.normalize(c)) for i, c in enumerate(raw) if ev.normalize(c)]
    if any(len(c) != 1 for _, c in mapped) or "".join(
        c for _, c in mapped
    ) != ev.normalize(raw):
        return None
    return [i for i, _ in mapped]


def discover():
    from PIL import Image, ImageDraw

    rows = prior.read_jsonl(OUT / "baseline-train.jsonl")
    pages = sorted({r["page"] for r in rows})
    random.Random(42).shuffle(pages)
    held = set(pages[-(len(pages) // 4) :])
    examples = defaultdict(list)
    correct = defaultdict(Counter)
    support = defaultdict(set)
    for r in rows:
        if r["page"] in held:
            continue
        raw, expected, actual = r["ocr_text"], r["expected"], r["ocr_normalized"]
        positions = tokens(raw)
        if positions is None or len(expected) != len(actual):
            continue
        diffs = [i for i in range(len(expected)) if expected[i] != actual[i]]
        if not diffs:
            for i, pos in enumerate(positions):
                correct[expected[i]][raw[pos]] += 1
                support[raw[pos]].add(r["page"])
        elif len(diffs) == 1:
            i = diffs[0]
            if (
                expected[i] in CONFIG["target_alphabet"]
                and actual[i] in CONFIG["target_alphabet"]
            ):
                examples[expected[i], raw[positions[i]]].append(
                    dict(
                        crop=r["crop"],
                        page=r["page"],
                        pos=i,
                        raw_pos=positions[i],
                        expected=expected,
                        ocr_text=raw,
                        sha256=r["sha256"],
                    )
                )
    ranking = []
    for (symbol, rival), items in examples.items():
        if not correct[symbol] or len(support[rival]) < 4:
            continue
        target = sorted(correct[symbol], key=lambda c: (-correct[symbol][c], c))[0]
        if len(support[target]) >= 4:
            ranking.append(
                dict(
                    target=target,
                    rival=rival,
                    pages=len({r["page"] for r in items}),
                    count=len(items),
                )
            )
    ranking.sort(key=lambda r: (-r["pages"], -r["count"], r["target"], r["rival"]))
    if not ranking or ranking[0]["pages"] < CONFIG["min_error_pages"]:
        raise RuntimeError(f"No repeated eligible substitution: {ranking[:4]}")
    best = ranking[0]
    items = examples[ev.normalize(best["target"]), best["rival"]]
    ev.save_json(
        OUT / "discovery.json",
        dict(held_pages=sorted(held), ranked=ranking, selected=best, examples=items),
    )
    for offset in range(0, len(items), 40):
        group = items[offset : offset + 40]
        sheet = Image.new("RGB", (4 * 265, ((len(group) + 3) // 4) * 88), "white")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(group):
            with Image.open(ev.OCR / r["crop"]) as img:
                img = img.convert("RGB")
                img.thumbnail((257, 60))
                x, y = (j % 4) * 265, (j // 4) * 88
                sheet.paste(img, (x, y))
                draw.text(
                    (x, y + 61),
                    f'{offset+j}: {Path(r["crop"]).stem} {r["expected"]}',
                    fill="black",
                )
        sheet.save(OUT / f"review-{offset:03d}.png")
    print("Discovery", best, "from", len(pages) - len(held), "pages", flush=True)


def instrument():
    """Expose existing pre-softmax logits and input features in a private copy of the official PIR graph."""
    import copy
    import shutil

    dst = OUT / "probe-inference"
    dst.mkdir(exist_ok=True)
    src = official.OFFICIAL_OCR
    for name in ("inference.pdiparams", "inference.yml"):
        shutil.copyfile(src / name, dst / name)
    graph = ev.load_json(src / "inference.json")

    def visit(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "ops" and isinstance(v, list):
                    yield v
                else:
                    yield from visit(v)
        if isinstance(obj, list):
            for v in obj:
                yield from visit(v)

    (ops,) = list(visit(graph))
    mat, add, soft, fetch = ops[-4:]
    if [op["#"].split(".")[-1] for op in (mat, add, soft, fetch)] != [
        "matmul",
        "add",
        "softmax",
        "fetch",
    ]:
        raise ValueError("Unexpected CTC graph")
    feature = ops[-5]["O"][0]
    if (
        mat["I"][0]["%"] != feature["%"]
        or mat["O"][0]["%"] != add["I"][0]["%"]
        or add["O"][0]["%"] != soft["I"][0]["%"]
    ):
        raise ValueError("Unexpected classifier wiring")
    fetch["I"] = [{"%": add["O"][0]["%"]}]
    second = copy.deepcopy(fetch)
    second["I"] = [{"%": feature["%"]}]
    second["O"][0] = {"%": fetch["O"][0]["%"] + 1, "TT": feature["TT"]}
    for a in second["A"]:
        if a["N"] == "name":
            a["AT"]["D"] = "fetch_name_1"
        if a["N"] == "col":
            a["AT"]["D"] = 1
    ops.append(second)
    ev.save_json(dst / "inference.json", graph)
    return dst


def extract():
    disc = ev.load_json(OUT / "discovery.json")
    rows = prior.read_jsonl(OUT / "baseline-train.jsonl")
    dst = instrument()
    rec = prior.recognizer(dst)
    pred = rec.paddlex_predictor
    chars = (
        ["<blank>"]
        + yaml.safe_load((dst / "inference.yml").read_text(encoding="utf-8"))[
            "PostProcess"
        ]["character_dict"]
        + [" "]
    )
    a, b = [chars.index(disc["selected"][k]) for k in ("target", "rival")]
    state_dir = OUT / "train-forward"
    state_dir.mkdir(exist_ok=True)
    disagreements = []
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        path = state_dir / f"{start:05d}.npz"
        x = pred.pre_tfs["Read"](imgs=[ev.read_bgr(ev.OCR / r["crop"]) for r in batch])
        x = pred.pre_tfs["ToBatch"](imgs=pred.pre_tfs["ReisizeNorm"](imgs=x))[0]
        outputs = pred.runner(x=[x])
        logits, h = sorted(outputs, key=lambda v: -v.shape[-1])
        if (
            logits.shape[-1] != len(chars)
            or h.shape[-1] != 192
            or h.shape[:2] != logits.shape[:2]
        ):
            raise ValueError("Invalid probe shapes")
        ab = logits[:, :, [a, b]].copy()
        logits[:, :, a] = -np.inf
        logits[:, :, b] = -np.inf
        other = logits.argmax(2).astype("int32")
        other_score = np.take_along_axis(logits, other[:, :, None], 2)[:, :, 0]
        ids = winner(ab, other, other_score, a, b)
        for i, r in enumerate(batch):
            dec = decode(ids[i], chars)
            if dec != r["ocr_text"]:
                disagreements.append(
                    dict(crop=r["crop"], official=r["ocr_text"], probe=dec)
                )
        np.savez(path, h=h, ab=ab, other=other, score=other_score)
        if start % 2048 == 0:
            print("Extract", start, "of", len(rows), flush=True)
    ev.save_json(
        OUT / "probe-audit.json",
        dict(count=len(rows), mismatches=disagreements, ids=[a, b]),
    )
    ev.save_json(OUT / "dictionary.json", chars)
    print("Forward mismatches", len(disagreements), flush=True)


def winner(ab, other, score, a, b):
    # Match argmax's lower-index tie break among all original vocabulary entries.
    ids = other.copy()
    top = score.copy()
    for k, c in enumerate((a, b)):
        v = ab[..., k]
        mask = (v > top) | ((v == top) & (c < ids))
        ids[mask] = c
        top[mask] = v[mask]
    return ids


def decode(ids, chars):
    keep = (ids != 0) & np.r_[True, ids[1:] != ids[:-1]]
    return "".join(chars[i] for i in ids[keep])


def batches(rows):
    for start in range(0, len(rows), 16):
        with np.load(OUT / "train-forward" / f"{start:05d}.npz") as cache:
            yield rows[start : start + 16], cache["h"], cache["ab"], cache[
                "other"
            ], cache["score"]


def design():
    disc = ev.load_json(OUT / "discovery.json")
    review = ev.load_json(OUT / "review.json")
    errors = {
        r["crop"]: r for r in disc["examples"] if r["crop"] not in review["exclude"]
    }
    held = set(disc["held_pages"])
    chars = ev.load_json(OUT / "dictionary.json")
    audit = ev.load_json(OUT / "probe-audit.json")
    a, b = audit["ids"]
    rows = prior.read_jsonl(OUT / "baseline-train.jsonl")
    vectors = defaultdict(lambda: defaultdict(list))
    error_h, deficits, positions = [], [], []
    for batch, h, ab, other, score in batches(rows):
        ids = winner(ab, other, score, a, b)
        for i, r in enumerate(batch):
            if r["page"] in held or decode(ids[i], chars) != r["ocr_text"]:
                continue
            runs = np.flatnonzero(np.r_[True, ids[i, 1:] != ids[i, :-1]])
            spans = [
                (s, e, int(ids[i, s]))
                for s, e in zip(runs, [*runs[1:], len(ids[i])])
                if ids[i, s] != 0
            ]
            for j, (s, e, char) in enumerate(spans):
                if char not in (a, b):
                    continue
                role = None
                if r["exact"]:
                    role = "correct_target" if char == a else "correct_rival"
                elif r["crop"] in errors and j == errors[r["crop"]]["raw_pos"]:
                    if char != b:
                        raise ValueError("Alignment changed")
                    role = "error"
                if role is None:
                    continue
                k = int(char == b)
                t = int(s + ab[i, s:e, k].argmax())
                feat = h[i, t].copy()
                vectors[role][r["page"]].append(feat)
                if role == "error":
                    error_h.append(feat)
                    deficits.append(float(ab[i, t, 1] - ab[i, t, 0]))
                positions.append(
                    dict(
                        crop=r["crop"],
                        page=r["page"],
                        role=role,
                        raw_pos=j,
                        glyph=chars[char],
                        run=[int(s), int(e)],
                        peak=t,
                    )
                )
    if len(vectors["error"]) < CONFIG["min_error_pages"]:
        raise ValueError("Insufficient aligned reviewed error pages")

    def center(role):
        return np.mean([np.mean(xs, axis=0) for xs in vectors[role].values()], axis=0)

    direction = center("error") - center("correct_rival")
    direction /= np.linalg.norm(direction)
    rng = np.random.default_rng(42)
    random_d = rng.normal(size=direction.shape).astype("float32")
    random_d /= np.linalg.norm(random_d)
    proj = np.array(error_h) @ direction
    ratios = np.asarray(deficits)[proj > 0] / (2 * proj[proj > 0])
    unit = float(np.median(ratios[ratios > 0]))
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("No positive intervention evidence")
    np.savez(
        OUT / "directions.npz", direct=direction, random=random_d, unit=unit, ids=[a, b]
    )
    ev.save_jsonl(OUT / "aligned-discovery.jsonl", positions)
    projections = {}
    for role, by_page in vectors.items():
        vals = np.concatenate(list(by_page.values())) @ direction
        projections[role] = dict(
            count=len(vals),
            pages=len(by_page),
            mean=float(vals.mean()),
            positive=int((vals > 0).sum()),
            median=float(np.median(vals)),
        )
    ev.save_json(
        OUT / "direction.json",
        dict(
            projections=projections,
            unit=unit,
            error_margin_median=float(np.median(deficits)),
            direction_norm=float(np.linalg.norm(direction)),
        ),
    )
    scores = {
        (k, s): Counter() for k in ("direct", "random") for s in CONFIG["strengths"]
    }
    effects = []
    for batch, h, ab, other, score in batches(rows):
        use = np.array([r["page"] in held for r in batch])
        if not use.any():
            continue
        h, ab, other, score = h[use], ab[use], other[use], score[use]
        sub = [r for i, r in enumerate(batch) if use[i]]
        initial = [
            ev.normalize(decode(ids, chars)) for ids in winner(ab, other, score, a, b)
        ]
        for k, d in (("direct", direction), ("random", random_d)):
            ds = h @ d
            for strength in CONFIG["strengths"]:
                candidate = ab.copy()
                dx = ds * (unit * strength)
                candidate[:, :, 0] += dx
                candidate[:, :, 1] -= dx
                ids = winner(candidate, other, score, a, b)
                for i, r in enumerate(sub):
                    new = ev.normalize(decode(ids[i], chars))
                    was = initial[i] == r["expected"]
                    now = new == r["expected"]
                    ctr = scores[k, strength]
                    ctr.update(
                        total=1,
                        repair=int(now and not was),
                        regression=int(was and not now),
                        changed=int(initial[i] != new),
                        probe_vs_official=int(initial[i] != r["ocr_normalized"]),
                    )
                    if new != initial[i]:
                        effects.append(
                            dict(
                                crop=r["crop"],
                                method=k,
                                strength=strength,
                                expected=r["expected"],
                                before=initial[i],
                                after=new,
                            )
                        )
    tables = []
    for (k, s), c in scores.items():
        tables.append(
            dict(method=k, strength=s, net=c["repair"] - c["regression"], **c)
        )
    directed = [r for r in tables if r["method"] == "direct"]
    for r in directed:
        control = next(
            q
            for q in tables
            if q["method"] == "random" and q["strength"] == r["strength"]
        )
        r["passes"] = r["net"] > max(0, control["net"])
    chosen = sorted(
        directed,
        key=lambda r: (-int(r["passes"]), -r["net"], r["regression"], r["strength"]),
    )[0]
    ev.save_jsonl(OUT / "training-holdout-changes.jsonl", effects)
    ev.save_json(
        OUT / "train-lock.json",
        dict(
            tables=tables,
            chosen=chosen,
            alpha=chosen["strength"] * unit,
            directions_sha256=ev.sha(OUT / "directions.npz"),
            review_sha256=ev.sha(OUT / "review.json"),
        ),
    )
    print("Held-out training pages", chosen, flush=True)


def pack():
    import shutil
    import paddle

    lock = ev.load_json(OUT / "train-lock.json")
    state = paddle.load(str(official.INITIAL_OCR), return_numpy=True)
    key = "head.ctc_head.fc.weight"
    w = state[key]
    old = w.tobytes()
    binary = (official.OFFICIAL_OCR / "inference.pdiparams").read_bytes()
    offset = binary.find(old)
    if offset < 0 or binary.find(old, offset + 1) >= 0:
        raise ValueError(
            "Export does not contain exactly one bit-identical original matrix"
        )
    with np.load(OUT / "directions.npz") as archive:
        a, b = map(int, archive["ids"])
        probe = np.load(OUT / "train-forward/00000.npz")
        features = probe["h"].astype("float64")
        first_logits = features @ w[:, [a, b]] + state["head.ctc_head.fc.bias"][[a, b]]
        probe_error = np.abs(first_logits - probe["ab"])
        if probe_error.max() > 0.10:
            raise ValueError(
                f"Classifier features/logits disagree: {probe_error.max()}"
            )
        for mode in ("direct", "random"):
            folder = OUT / mode
            infer = folder / "inference"
            infer.mkdir(parents=True, exist_ok=True)
            new = w.copy()
            update = archive[mode] * lock["alpha"]
            new[:, a] += update
            new[:, b] -= update
            shift = features @ update
            direct_change = features @ (new[:, [a, b]] - w[:, [a, b]])
            gap = np.max(np.abs(direct_change - np.stack((shift, -shift), axis=-1)))
            if gap > 0.0001:
                raise ValueError("Merged columns differ from the planned intervention")
            if not np.array_equal(
                np.flatnonzero((new != w).any(axis=0)), sorted((a, b))
            ):
                raise ValueError("Unexpected columns")
            data = binary[:offset] + new.tobytes() + binary[offset + len(old) :]
            for name in ("inference.json", "inference.yml"):
                shutil.copyfile(official.OFFICIAL_OCR / name, infer / name)
            (infer / "inference.pdiparams").write_bytes(data)
            merged = dict(state)
            merged[key] = new
            paddle.save(merged, str(folder / "model.pdparams"))
            check = paddle.load(str(folder / "model.pdparams"), return_numpy=True)
            if set(check) != set(merged) or any(
                not np.array_equal(check[k], merged[k]) for k in merged
            ):
                raise ValueError("Checkpoint round-trip failed")
            ev.save_json(
                folder / "patch.json",
                dict(
                    parameter=key,
                    shape=list(w.shape),
                    column_ids=[a, b],
                    changed=int(np.count_nonzero(new != w)),
                    relative_frobenius=float(
                        np.linalg.norm(new - w) / np.linalg.norm(w)
                    ),
                    alpha=lock["alpha"],
                    offset=offset,
                    checkpoint_sha256=ev.sha(folder / "model.pdparams"),
                    original_probe_logit_max_error=float(probe_error.max()),
                    merged_delta_probe_max_error=float(gap),
                    export_sha256=prior.model_hashes(infer),
                    original_matrix_sha256=hashlib.sha256(old).hexdigest(),
                ),
            )


def paired(before, after):
    changes = Counter(
        dict(
            repaired=0,
            regressed=0,
            both_wrong=0,
            both_right=0,
            grade_repaired=0,
            grade_regressed=0,
            added_false_accept=0,
            removed_false_accept=0,
            output_changed=0,
        )
    )
    details = []
    for x, y in zip(before, after, strict=True):
        if (x["page"], x["box_index"], x["expected_raw"]) != (
            y["page"],
            y["box_index"],
            y["expected_raw"],
        ):
            raise ValueError("Row identity changed")
        if not x["expected"]:
            continue
        category = (
            ("both_right" if x["exact"] else "repaired")
            if y["exact"]
            else ("regressed" if x["exact"] else "both_wrong")
        )
        changes[category] += 1
        changed = x["ocr_normalized"] != y["ocr_normalized"]
        truth = x["truth_grade"]
        old, new = x["ocr_grade"], y["ocr_grade"]
        added = truth is False and new is True and old is not True
        removed = truth is False and old is True and new is not True
        changes.update(
            output_changed=int(changed),
            added_false_accept=int(added),
            removed_false_accept=int(removed),
            grade_repaired=int(truth is not None and new is truth and old is not truth),
            grade_regressed=int(
                truth is not None and old is truth and new is not truth
            ),
        )
        details.append(
            dict(
                page=x["page"],
                box_index=x["box_index"],
                expected=x["expected"],
                category=category,
                before=x["ocr_text"],
                after=y["ocr_text"],
                truth_grade=truth,
                before_grade=old,
                after_grade=new,
                added_false_accept=added,
                removed_false_accept=removed,
                **(
                    {
                        "candidates_before": x.get("candidates"),
                        "candidates_after": y.get("candidates"),
                        "match_failure": y.get("match_failure"),
                    }
                    if y.get("matched") is False
                    else {}
                ),
            )
        )
    return dict(changes), details


def targeted(before, after):
    selected = ev.load_json(OUT / "discovery.json")["selected"]
    wanted, rival = (ev.normalize(selected[k]) for k in ("target", "rival"))
    counts = Counter(
        dict(
            isolated_target_errors=0,
            repaired_target_errors=0,
            originally_exact_with_rival=0,
            damaged_exact_with_rival=0,
            originally_exact_with_target=0,
            damaged_exact_with_target=0,
        )
    )
    for x, y in zip(before, after, strict=True):
        gt = x["expected"]
        text = x["ocr_normalized"] or ""
        if not gt:
            continue
        if x["exact"]:
            for glyph, name in ((wanted, "target"), (rival, "rival")):
                if glyph in gt:
                    counts["originally_exact_with_" + name] += 1
                    counts["damaged_exact_with_" + name] += int(not y["exact"])
        elif len(gt) == len(text) and [(u, v) for u, v in zip(gt, text) if u != v] == [
            (wanted, rival)
        ]:
            counts["isolated_target_errors"] += 1
            counts["repaired_target_errors"] += int(y["exact"])
    return dict(counts)


def verify_packed(mode, rec):
    rows = prior.read_jsonl(OUT / "baseline-train.jsonl")
    disc = ev.load_json(OUT / "discovery.json")
    probes = {r["crop"] for r in disc["examples"]}
    # Include affected held-out pages as well as the first full batch of controls.
    changes = prior.read_jsonl(OUT / "training-holdout-changes.jsonl")
    lock = ev.load_json(OUT / "train-lock.json")
    probes.update(
        r["crop"]
        for r in changes
        if r["method"] == mode and r["strength"] == lock["chosen"]["strength"]
    )
    starts = {0} | {(i // 16) * 16 for i, r in enumerate(rows) if r["crop"] in probes}
    chars = ev.load_json(OUT / "dictionary.json")
    data = np.load(OUT / "directions.npz")
    a, b = map(int, data["ids"])
    checked = 0
    mismatches = []
    for start in sorted(starts):
        batch = rows[start : start + 16]
        with np.load(OUT / "train-forward" / f"{start:05d}.npz") as z:
            ab = z["ab"]
            dx = (z["h"] @ data[mode]) * lock["alpha"]
            ab[:, :, 0] += dx
            ab[:, :, 1] -= dx
            ids = winner(ab, z["other"], z["score"], a, b)
        preds = ev.recognize(rec, [ev.read_bgr(ev.OCR / r["crop"]) for r in batch])
        for r, seq, p in zip(batch, ids, preds, strict=True):
            predicted = decode(seq, chars)
            if p["ocr_text"] != predicted:
                mismatches.append(
                    dict(crop=r["crop"], analytic=predicted, export=p["ocr_text"])
                )
            checked += 1
    ev.save_json(
        OUT / mode / "export-parity.json", dict(checked=checked, mismatches=mismatches)
    )
    if mismatches:
        raise ValueError(f"{mode} exported decode parity: {mismatches[:2]}")


def evaluate(mode, split):
    folder = OUT / mode
    patch = ev.load_json(folder / "patch.json")
    if prior.model_hashes(folder / "inference") != patch["export_sha256"]:
        raise ValueError("Model changed")
    if split == "test":
        locked = ev.load_json(OUT / "selection-lock.json")
        if locked["models"][mode] != patch["export_sha256"]:
            raise ValueError("Candidate differs from validation lock")
    elif split != "val":
        raise ValueError(split)
    rec = prior.recognizer(folder / "inference")
    ann, splits = prior.annotations_and_split()
    manifest = ev.load_json(prior.controlled.OUT / "manifest.json")
    if split == "val":
        verify_packed(mode, rec)
    prior.verify_labels(split, ann, splits, manifest)
    crops = prior.evaluate_split(rec, split, ann, splits)
    sections = {split: crops}
    baseline = {
        split: prior.read_jsonl(
            OUT / "baseline-val.jsonl" if split == "val" else BASE / "test.jsonl"
        )
    }
    if split == "test":
        cache = prior.controlled.RESULTS / "shared-detections.json"
        cached = ev.load_json(cache)
        base_summary = ev.load_json(BASE / "summary.json")
        if (
            ev.sha(cache) != base_summary["detection_cache_sha256"]
            or cached["detector_sha256"] != ev.sha(ev.DETECTOR)
            or [p["image"] for p in cached["pages"]] != splits["test"]
            or len(splits["test"]) != 88
            or any(cached["settings"][k] != v for k, v in ev.DETECTION_CONFIG.items())
        ):
            raise ValueError("Unexpected Y2 detections")
        pages = list(ev.evaluate_pages(rec, cached, ann))
        ev.save_jsonl(folder / "pages-predictions.jsonl", pages)
        sections["pages"] = [r for p in pages for r in p["targets"]]
        baseline["pages"] = prior.read_jsonl(BASE / "pages.jsonl")
        for scope in ("test", "pages"):
            for label, subset in (
                ("fraction", lambda r: r["page"] == "10.jpg" and r["expected"]),
                ("ordinary", lambda r: Path(r["page"]).stem not in ev.FRACTION_PAGES),
            ):
                group = f"{scope}-{label}"
                sections[group] = [r for r in sections[scope] if subset(r)]
                baseline[group] = [r for r in baseline[scope] if subset(r)]
                if label == "fraction" and len(sections[group]) != 23:
                    raise ValueError("Missing fraction")
                if label == "ordinary":
                    errors, items = ev.character_errors(sections[group])
                    ev.save_json(folder / f"{group}-characters.json", errors)
                    ev.save_jsonl(folder / f"{group}-characters.jsonl", items)
    results = {}
    for scope, rows in sections.items():
        delta, details = paired(baseline[scope], rows)
        results[scope] = dict(
            metrics=ev.summarize(rows),
            baseline=ev.summarize(baseline[scope]),
            changes=delta,
            target_behavior=targeted(baseline[scope], rows),
        )
        ev.save_jsonl(folder / f"{scope}-pairs.jsonl", details)
        if scope in ("test", "val", "pages"):
            ev.save_jsonl(folder / f"{scope}.jsonl", rows)
    ev.save_json(folder / f"{split}-metrics.json", results)
    print(mode, split, results[split], flush=True)


def lock():
    val = ev.load_json(OUT / "direct/val-metrics.json")["val"]
    ev.load_json(OUT / "random/val-metrics.json")
    before, now = val["baseline"]["counts"], val["metrics"]["counts"]
    passed = (
        now["exact"] > before["exact"]
        and now["edits"] <= before["edits"]
        and now["wrong_accepted"] <= before["wrong_accepted"]
        and ev.load_json(OUT / "train-lock.json")["chosen"]["passes"]
    )
    ev.save_json(
        OUT / "selection-lock.json",
        dict(
            accepted=passed,
            recommended="direct" if passed else "O0",
            models={
                m: prior.model_hashes(OUT / m / "inference")
                for m in ("direct", "random")
            },
            test_policy="Report both fixed models including nonaccepted diagnostic; no selection on test.",
            validation_changes=val["changes"],
            detector_sha256=ev.sha(ev.DETECTOR),
        ),
    )


def report():
    import swanlab

    discovery = ev.load_json(OUT / "discovery.json")
    chosen = discovery["selected"]
    audit = ev.load_json(OUT / "probe-audit.json")
    train = ev.load_json(OUT / "train-lock.json")
    choice = ev.load_json(OUT / "selection-lock.json")
    direction = ev.load_json(OUT / "direction.json")
    modes = ("direct", "random")
    results = {
        m: ev.load_json(OUT / m / "val-metrics.json")
        | ev.load_json(OUT / m / "test-metrics.json")
        for m in modes
    }
    ev.save_json(OUT / "comparison.json", results)
    tracking_path = OUT / "swanlab-status.json"
    tracking = ev.load_json(tracking_path) if tracking_path.exists() else []
    log_dir = OUT / "swanlab"
    log_dir.mkdir(exist_ok=True)
    for mode in modes:
        if any(item["method"] == mode and Path(item["directory"]).is_dir() for item in tracking):
            continue
        with swanlab.init(
            project="ModelAdjust",
            name=f"direct-ocr-edit-v1-{mode}",
            group="direct-ocr-edit-v1",
            mode="offline",
            log_dir=str(log_dir),
            config=CONFIG | {"method": mode},
            settings=swanlab.Settings(interactive=False),
        ) as run:
            for step, table in enumerate(
                t for t in train["tables"] if t["method"] == mode
            ):
                run.log(
                    {
                        f"training_probe/{k}": v
                        for k, v in table.items()
                        if k != "method"
                    },
                    step=step,
                )
            run.log(
                {
                    f"{scope}/{key}": value
                    for scope, data in results[mode].items()
                    for key, value in (
                        data["metrics"]["counts"]
                        | data["metrics"]["metrics"]
                        | data["changes"]
                    ).items()
                    if value is not None
                },
                step=3,
            )
            tracking.append(
                dict(
                    method=mode,
                    run_id=run.id,
                    directory=str(run.dir),
                    synchronized=False,
                )
            )
    ev.save_json(OUT / "swanlab-status.json", tracking)
    lines = [
        "# 官方 PaddleOCR 无梯度直接权重编辑",
        "",
        f"固定训练划分 406 页，其中 386 页有带文字标签的训练裁图。按整页留出 {len(discovery['held_pages'])} 页检验干预强度，另外 {386-len(discovery['held_pages'])} 页用于发现与构造方向。原始 13,206 张裁图全部完成官方前向识别。独立验证集用于接受判断。测试为已有测试集上的后续探索。",
        "",
        f"发现页中，真值字符 {chosen['target']!r} 被识别成 {chosen['rival']!r} 的孤立替换跨 {chosen['pages']} 页、{chosen['count']} 张。核图记录见 review.json。CTC 头输入为 192 维，使用错误位置与真竞争字符位置的逐页激活均值差，归一化后加到目标列，同时从竞争列减去。只改一处原始权重矩阵，最多改动 384 个数值，无反向传播与优化器。随机组使用相同幅度的随机单位方向。",
        "",
        f"前向探针共覆盖 {audit['count']} 张训练图，原始解码与官方的差异数为 {len(audit['mismatches'])}。幅度单位为 {direction['unit']:.7g}，锁定的倍数为 {train['chosen']['strength']}，实际系数为 {train['alpha']:.7g}。定向和随机组均直接保存了可独立加载的训练检查点和标准 TextRecognition 模型。",
        "",
        "| 训练留出页方向 | 幅度倍数 | 全文修复 | 全文退步 | 净修复 |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in train["tables"]:
        lines.append(
            f"| {p['method']} | {p['strength']} | {p['repair']} | {p['regression']} | {p['net']} |"
        )
    lines += [
        "",
        "| 范围 | 模型 | 全文正确 | CER | 判题正确 | 错题判对 | 无法判定 | 全文修复 / 退步 | 新增 / 消除错题判对 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in results["direct"]:
        for mode in ("O0", *modes):
            row = results["direct" if mode == "O0" else mode][scope]
            data = row["baseline" if mode == "O0" else "metrics"]
            c = data["counts"]
            ch = Counter() if mode == "O0" else row["changes"]
            lines.append(
                f"| {scope} | {mode} | {c['exact']}/{c['labeled']} | {data['metrics']['cer']:.4%} | "
                f"{c['grade_correct']}/{c['gradeable']} | {c['wrong_accepted']} | {c['undetermined']} | "
                f"{ch['repaired']} / {ch['regressed']} | {ch['added_false_accept']} / {ch['removed_false_accept']} |"
            )
    lines += [
        "",
        "| 定向补丁 | 原有孤立目标混淆 | 修复为全文正确 | 原本全文正确且含真竞争字符 | 其中退步 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope in ("val", "test", "pages"):
        t = results["direct"][scope]["target_behavior"]
        lines.append(
            f"| {scope} | {t['isolated_target_errors']} | {t['repaired_target_errors']} | {t['originally_exact_with_rival']} | {t['damaged_exact_with_rival']} |"
        )
    reviewed_crop = "crops/val/456_0012.jpg"
    issue = next(
        r
        for r in prior.read_jsonl(OUT / "direct/val-pairs.jsonl")
        if r["page"] == "456.jpg" and r["box_index"] == 12
    )
    ev.save_json(
        OUT / "label-review.json",
        [
            {
                "crop": reviewed_crop,
                "sha256": ev.sha(ev.OCR / reviewed_crop),
                **issue,
                "visual_observation": "印刷题干可见 1000×0.01，现有标签的第二个因数为 0.1。手写答案看似 10。请核对原图。",
                "action": "列为疑点，保留原标签、接受标准和模型选择。",
            }
        ],
    )
    lines += [
        "",
        f"验证选择推荐 **{choice['recommended']}**。接受条件为全文读对数增加、CER 不升、错题判对数不增加，同时满足训练留出页的方向检验。",
        "",
        "验证集新增的一道错题判对涉及疑似标注问题。456.jpg 第 12 号框的标签是 `1000*0.1=10`，原图印刷题干可见 `1000×0.01`。官方输出 `1000×0.01=0`，编辑后为 `1000×0.01=10`。正式结果仍按现有标签统计，详见 label-review.json。",
        "",
        "整页沿用 Y2-geometry-2 的缓存检测框、置信度 0.50 和 IoU 0.50。每个模型目录的 *-pairs.jsonl 记录逐题实际输出，pages-predictions.jsonl 保留未匹配候选。ordinary 的 characters 文件统计普通算式的数字、小数点与符号。10.jpg 单列 23 道分式题。",
        "",
        "历史 200 步 CTC 全量微调在同一测试裁图上全文读对 2,600/2,767、判对 2,606/2,757。它使用 3,200 张训练裁图与梯度更新，本实验扫描 13,206 张训练图并只改两列，属于历史效果参照。训练与验证选择均保持学生实际转录。SwanLab 已记录本地离线结果。",
        "",
        "## 复现顺序",
        "",
        "在 ModelAdjust 下使用 labelstudioml 的 Python，环境变量与 weight_editing/run_official_weight_edit.ps1 相同。weight_editing/direct_ocr_edit.py 的子命令依序为 `baseline train`、`baseline val`、`discover`、`extract`、`design`、`pack`、`evaluate direct val`、`evaluate random val`、`lock`、`evaluate direct test`、`evaluate random test`、`report`。design 使用本目录已记录的训练核图决定 review.json。probe-inference 为读取激活的实验图，最终部署使用 direct/inference 或 random/inference。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "baseline":
        form(sys.argv[2])
    else:
        commands = {
            f.__name__: f
            for f in (discover, extract, design, pack, evaluate, lock, report)
        }
        commands[sys.argv[1]](*sys.argv[2:])


if __name__ == "__main__":
    main()
