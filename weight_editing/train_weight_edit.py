"""Optimize one CTC classifier matrix against the frozen O3 inference branch."""
import argparse
from collections import Counter
import copy
import hashlib
import math
import os
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from weight_editing.weight_edit import (
    BEST, CONFIG, GROUPS, O3, OCR, OUT, ROOT, SEED, SOURCE, STEPS,
    load_json, read_bgr, read_jsonl, save_json, sha,
)

# Match the Windows training launcher: do not load PyTorch into the Paddle process.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.modules["torch"] = None
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / "vendor/ocr_deps"))

import cv2
import paddle
from paddle.nn import functional as F
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.export_model import convert_bn

def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    paddle.seed(SEED)


def build():
    cfg = yaml.safe_load((O3 / "export.yaml").read_text(encoding="utf-8"))
    architecture = copy.deepcopy(cfg["Architecture"])
    cfg["Global"]["character_dict_path"] = str(SOURCE / cfg["Global"]["character_dict_path"])
    decoder = build_post_process(cfg["PostProcess"], cfg["Global"])
    exported = yaml.safe_load((O3 / "inference/inference.yml").read_text(encoding="utf-8"))
    # Both decoders append a space at runtime; the YAML holds only the raw dictionary.
    exported_chars = exported["PostProcess"]["character_dict"] + [" "]
    if decoder.character[1:] != exported_chars:
        raise ValueError("Training and exported inference dictionaries differ")
    n = len(decoder.character)
    architecture["Head"]["out_channels_list"] = dict(CTCLabelDecode=n, SARLabelDecode=n+2,
                                                      NRTRLabelDecode=n+3)
    net = build_model(architecture)
    state = paddle.load(str(BEST), return_numpy=True)
    actual = net.state_dict()
    if set(state) != set(actual) or any(tuple(actual[key].shape) != state[key].shape for key in state):
        raise ValueError(f"Training checkpoint mismatch: {set(state) ^ set(actual)}")
    net.set_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.stop_gradient = True

    neck = net.head.ctc_encoder
    head = net.head.ctc_head
    classifier = head.fc if head.mid_channels is None else head.fc2
    if not isinstance(classifier, paddle.nn.Linear) or head.use_guide or head.mid_channels is not None:
        raise ValueError("The inference branch differs from the audited CTC path")
    names = [name for name, p in net.named_parameters() if p is classifier.weight]
    if len(names) != 1 or len(classifier.weight.shape) != 2 or not net.use_backbone or (
        net.use_transform or net.use_neck or net.head.use_pool or net.return_all_feats
    ):
        raise ValueError("Unexpected inference path or ambiguous classifier weight")
    name = names[0]
    width, classes = classifier.weight.shape
    if classes != n:
        raise ValueError("Classifier width differs from the decoder")
    rank = min(CONFIG["rank"], width, classes)
    return net, neck, classifier, name, state, decoder, rank


def audit_state(net, reference, permitted=()):
    changed = [name for name, value in net.state_dict().items()
               if not np.array_equal(value.numpy(), reference[name])]
    if set(changed) != set(permitted):
        raise ValueError(f"Unexpected changed parameters or buffers: {changed}")
    if any(not p.stop_gradient or p.grad is not None for p in net.parameters()):
        raise ValueError("An original parameter was trainable or has a gradient")
    if net.training or any(layer.training for layer in net.sublayers()):
        raise ValueError("Original model has a layer in training mode")
    return changed


def resize_image(image, fixed=False):
    height, width = image.shape[:2]
    ratio = width / float(height)
    canvas_width = 320 if fixed else min(3200, int(48 * max(320 / 48, ratio)))
    active_width = min(canvas_width, math.ceil(48 * ratio))
    canvas = np.zeros((3, 48, canvas_width), dtype=np.float32)
    pixels = cv2.resize(image, (active_width, 48)).astype(np.float32).transpose((2, 0, 1)) / 255
    pixels -= 0.5
    pixels /= 0.5
    canvas[:, :, :active_width] = pixels
    return canvas, active_width


def input_batch(crops, fixed=False):
    arrays, widths = zip(*(resize_image(read_bgr(OCR / crop), fixed) for crop in crops), strict=True)
    max_width = max(a.shape[2] for a in arrays)
    tensor = np.zeros((len(crops), 3, 48, max_width), dtype=np.float32)
    for index, array in enumerate(arrays):
        tensor[index, :, :, :array.shape[-1]] = array
    return tensor, np.array(widths)


def time_steps(width):
    # Stem width strides 2, 2 (SAME); stages keep width, final pool is width 2/stride 2.
    return ((width + 3) // 4) // 2


def frozen_export_graph(net):
    fused = copy.deepcopy(net)
    convert_bn(fused)
    fused.eval()
    with paddle.no_grad():
        for layer in fused.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped", False):
                layer.rep()
    fused.eval()
    for p in fused.parameters():
        p.stop_gradient = True
    return fused


def probability_difference(a, b):
    error = np.abs(a-b)
    return dict(max_abs=float(error.max()), mean_abs=float(error.mean()),
                p99_9_abs=float(np.quantile(error, 0.999)),
                argmax_disagreements=int((a.argmax(-1) != b.argmax(-1)).sum()))


def check_parity(net, neck, classifier, decoder, adapters):
    batch = np.load(OUT / "parity-probe.npz")
    crops = load_json(OUT / "parity-probe.json")["crops"]
    x, widths = input_batch(crops)
    np.testing.assert_array_equal(x, batch["input"])
    np.testing.assert_array_equal(widths, batch["widths"])
    t = paddle.to_tensor(x)
    with paddle.no_grad():
        features = neck(net.backbone(t))
        baseline = net(t)
        frozen_repeat = net(t)
        same = F.softmax(F.linear(features, classifier.weight + adapters.delta(), classifier.bias), axis=-1)
        standalone_widths = sorted(set([7, 8, 9, 11, 12, 13, 79, 80, 81, 319, 320, 321, x.shape[-1]]
                                       + [int(width) for width in widths]))
        measured = {width: net(paddle.zeros([1, 3, 48, width])).shape[1] for width in standalone_widths}
        fixed, _ = input_batch(crops, fixed=True)
        fixed_probs = net(paddle.to_tensor(fixed)).numpy()
    if measured != {w: time_steps(w) for w in measured} or baseline.shape[1] != time_steps(x.shape[-1]):
        raise ValueError(f"Invalid measured CTC width-to-length function: {measured}")
    base = baseline.numpy()
    exported = batch["probabilities"]
    np.testing.assert_array_equal(frozen_repeat.numpy(), base)
    np.testing.assert_array_equal(same.numpy(), base)
    difference = probability_difference(base, exported)
    print(f"Dynamic/export probability difference: {difference}", flush=True)
    baseline_text = [result[0] for result in decoder(base)]
    exported_text = [r["ocr_text"] for r in load_json(OUT / "parity-probe.json")["predictions"]]
    if baseline_text != exported_text or difference["max_abs"] > 0.075 or difference["mean_abs"] > 1e-5:
        raise ValueError(f"Baseline decode parity failed: {baseline_text} vs {exported_text}")
    fixed_text = [result[0] for result in decoder(fixed_probs)]
    return dict(crops=len(crops), exported_width=x.shape[-1], o3_training_width=fixed.shape[-1],
                export_vs_dynamic_max_abs=float(np.max(np.abs(base-exported))),
                export_vs_dynamic=difference,
                zero_delta_max_abs=float(np.max(np.abs(base-same.numpy()))),
                repeated_eval_max_abs=float(np.max(np.abs(base-frozen_repeat.numpy()))),
                fixed_320_eval_decodes_different=sum(a != b for a, b in zip(fixed_text, baseline_text, strict=True)),
                batch_time_steps=base.shape[1], active_time_steps=time_steps(widths).tolist(),
                standalone_width_time_steps=measured, pixels_exact=True, decode_exact=True,
                width_rule="floor(ceil(effective_width / 4) / 2); clamped to batched time dimension; "
                           "KL masks steps at or beyond this valid length",
                ctc_input_lengths="Full batched output length, matching original PaddleOCR CTCLoss and decoder",
                preprocess="BGR, height 48, aspect-preserving width with floor 320 and cap 3200 for the canvas; "
                           "right zero-pad to batch maximum; fixed-width O3 train/Eval uses width 320",
                mode="Frozen eval: CTC only, dropout disabled and BN statistics fixed. O3 train mode has adaptive width 40; "
                     "eval/export uses ordinary pooling with dynamic time axis.")


class Factors(paddle.nn.Layer):
    def __init__(self, shape, rank):
        super().__init__()
        self.u = self.create_parameter([shape[0], rank], dtype="float32",
                                       default_initializer=paddle.nn.initializer.Normal(std=0.01))
        self.v = self.create_parameter([shape[1], rank], dtype="float32",
                                       default_initializer=paddle.nn.initializer.Constant(0))

    def delta(self):
        return paddle.matmul(self.u, self.v, transpose_y=True)


def training_data(decoder):
    char_ids = {c: i for i, c in enumerate(decoder.character)}
    labels = {}
    roles = {}
    for key in ("edit", "retain"):
        rows = read_jsonl(OUT / f"{key}.jsonl")
        for row in rows:
            name = row["crop"]
            if name in labels or not name.startswith("crops/train/") or sha(OCR / name) != row["sha256"]:
                raise ValueError(f"Edited or overlapping sample: {name}")
            target = row["expected"]
            if not target or len(target) > 25 or any(c not in char_ids or char_ids[c] == 0 for c in target):
                raise ValueError(f"Incompatible {key} label: {name} {target!r}")
            labels[name] = [char_ids[c] for c in target]
            roles[name] = key
    schedule = read_jsonl(OUT / "sampling-sequence.jsonl")
    if len(schedule) != 200:
        raise ValueError("Incomplete sampling schedule")
    for step, row in enumerate(schedule, 1):
        if row["step"] != step or any(len(row[key]) != 8 or any(roles.get(name) != key for name in row[key])
                                      for key in ("edit", "retain")):
            raise ValueError(f"Invalid schedule row {step}")
    return labels, roles, schedule


def make_batch(row, labels, net, neck):
    crops = row["edit"] + row["retain"]
    inputs, widths = input_batch(crops)
    with paddle.no_grad():
        features = neck(net.backbone(paddle.to_tensor(inputs)))
    t = features.shape[1]
    valid = np.clip(time_steps(widths), 1, t)
    if t != time_steps(inputs.shape[-1]):
        raise ValueError("Unexpected CTC output width")
    targets = [labels[name] for name in crops]
    can_align = [len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) <= t for ids in targets]
    if not all(can_align):
        raise ValueError(f"CTC cannot align labels at step {row['step']}: "
                         f"{[name for name, good in zip(crops, can_align, strict=True) if not good]}")
    y = np.zeros((16, max(map(len, targets))), dtype=np.int32)
    for i, ids in enumerate(targets):
        y[i, :len(ids)] = ids
    cpu = paddle.CPUPlace()
    # The bridge uses CPU for actual WarpCTC and returns the gradient to the input place.
    return (features, paddle.to_tensor(y, place=cpu), paddle.to_tensor([t]*16, dtype="int64", place=cpu),
            paddle.to_tensor([len(ids) for ids in targets], dtype="int64", place=cpu),
            paddle.to_tensor((np.arange(t)[None, :] < valid[8:, None]).astype(np.float32)))


def losses(batch, layer, factors):
    features, labels, lengths, label_lengths, mask = batch
    delta = factors.delta()
    logits = F.linear(features, layer.weight + delta, layer.bias)
    ctc = ctc_loss_vector(logits.transpose([1, 0, 2]), labels, lengths, label_lengths)
    with paddle.no_grad():
        teacher = F.log_softmax(layer(features[8:]), axis=-1)
    student = F.log_softmax(logits[8:], axis=-1)
    if not teacher.stop_gradient:
        raise ValueError("Teacher unexpectedly has gradients")
    # Float32 cancellation can make KL for near-identical rows slightly negative.
    # p*(exp(x)-1-x), x=log q-log p, is equivalent after summing normalized classes;
    # use its Taylor expansion near zero to avoid cancellation.
    x = student - teacher
    small = paddle.abs(x) < 0.01
    per_class = paddle.where(small, x*x*(0.5 + x/6 + x*x/24), paddle.expm1(x) - x)
    token_kl = paddle.sum(paddle.exp(teacher) * per_class, axis=-1)
    kl = (token_kl * mask).sum() / mask.sum()
    relative_squared = delta.square().sum() / (layer.weight.square().sum() + CONFIG["epsilon"])
    return ctc[:8].mean(), ctc[8:].mean(), kl, relative_squared


def ctc_loss_vector(logits, labels, lengths, label_lengths):
    if os.name != "nt" or not logits.place.is_gpu_place():
        return F.ctc_loss(logits, labels, lengths, label_lengths, blank=0, reduction="none")
    # Reuse the project's CPU WarpCTC strategy on Windows. Its nested PyLayer
    # backwards crashes on repeated/component gradient queries in this build.
    # Each item in WarpCTC is independent; differentiating their sum gives all
    # per-item logit gradients in one pass. Attach that first-order derivative
    # explicitly on the GPU, so each of the two sub-batch means remains exact.
    cpu_logits = logits.detach().cpu()
    cpu_logits.stop_gradient = False
    values = F.ctc_loss(cpu_logits, labels, lengths, label_lengths, blank=0, reduction="none")
    values.sum().backward()
    derivative = cpu_logits.grad.detach().to(logits.place)
    zero_with_gradient = ((logits - logits.detach()) * derivative).sum(axis=[0, 2])
    return values.detach().to(logits.place) + zero_with_gradient


def check_windows_ctc():
    if os.name != "nt":
        return None
    cpu = paddle.CPUPlace()
    rng = np.random.default_rng(SEED)
    input_array = rng.normal(size=(7, 2, 6)).astype(np.float32)
    labels = paddle.to_tensor([[1, 2, 2], [4, 4, 0]], place=cpu, dtype="int32")
    lengths = paddle.to_tensor([7, 6], place=cpu, dtype="int64")
    target_lengths = paddle.to_tensor([3, 2], place=cpu, dtype="int64")
    gpu = paddle.to_tensor(input_array, stop_gradient=False)
    original = paddle.to_tensor(input_array, place=cpu, stop_gradient=False)
    direct = F.ctc_loss(original, labels, lengths, target_lengths, blank=0, reduction="none")
    bridged = ctc_loss_vector(gpu, labels, lengths, target_lengths)
    np.testing.assert_array_equal(bridged.numpy(), direct.numpy())
    difference = 0.0
    for coefficients in ([1.0, 0.0], [0.0, 0.25], [0.5, 2.0]):
        (ref,) = paddle.grad([(direct * paddle.to_tensor(coefficients, place=cpu)).sum()],
                             [original], retain_graph=True)
        (actual,) = paddle.grad([(bridged * paddle.to_tensor(coefficients)).sum()], [gpu], retain_graph=True)
        np.testing.assert_allclose(actual.numpy(), ref.numpy(), atol=1e-7, rtol=1e-6)
        difference = max(difference, float(np.max(np.abs(actual.numpy() - ref.numpy()))))
    return dict(forward_exact=True, derivative_max_abs=difference,
                engine="Paddle CPU WarpCTC with exact first-order gradient copied to GPU")


def norm(tensors):
    return math.sqrt(sum(float(t.square().sum().item()) for t in tensors))


def checkpoint(group_dir, step, net, fused, classifier, key, reference, factors, decoder):
    current = group_dir / f"step-{step:03}"
    current.mkdir(parents=True, exist_ok=False)
    probe = np.load(OUT / "parity-probe.npz")
    inputs = paddle.to_tensor(probe["input"])
    original = classifier.weight.clone()
    with paddle.no_grad():
        delta = factors.delta()
        merged = original + delta
        features = fused.head.ctc_encoder(fused.backbone(inputs))
        before = F.softmax(F.linear(features, merged, classifier.bias), axis=-1).numpy()
        additive = F.softmax(F.linear(features, original, classifier.bias) + features @ delta, axis=-1).numpy()
        if np.max(np.abs(before-additive)) > 0.002:
            raise ValueError("Separate and combined matrix products diverged")
        classifier.weight.set_value(merged)
        fused.head.ctc_head.fc.weight.set_value(merged)
        try:
            after = fused(inputs).numpy()
            np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
            if step and np.array_equal(merged.numpy(), original.numpy()):
                raise ValueError("Edited checkpoint has zero representable weight change")
            changed = audit_state(net, reference, (key,) if step else ())
            paddle.save(net.state_dict(), str(current / "merged.pdparams"))
        finally:
            classifier.weight.set_value(original)
            fused.head.ctc_head.fc.weight.set_value(original)
    u, v = factors.u.numpy(), factors.v.numpy()
    np.savez(current / "low-rank-factors.npz", U=u, V=v)
    np.save(current / "parity-probabilities.npy", after)
    save_json(current / "checkpoint.json", {
        "step": step, "relative_weight_delta": norm([delta]) / norm([original]),
        "merge_probability_max_abs": float(np.max(np.abs(after-before))),
        "separate_delta_probability_max_abs": float(np.max(np.abs(additive-after))),
        "merged_changed_state_keys": changed, "frozen_state_verified": len(reference),
        "probe_decodes": [result[0] for result in decoder(after)],
        "factor_hashes": dict(U=hashlib.sha256(u.tobytes()).hexdigest(),
                              V=hashlib.sha256(v.tobytes()).hexdigest()),
        "weights_sha256": sha(current / "merged.pdparams"),
    })


def run_group(group, n_steps, smoke):
    seed()
    paddle.set_device(CONFIG["device"])
    net, neck, classifier, name, reference, decoder, rank = build()
    seed()
    factors = Factors(classifier.weight.shape, rank)
    parameters = factors.parameters()
    if any(p.stop_gradient for p in parameters) or len(parameters) != 2:
        raise ValueError("Unexpected adapter parameters")
    opt = paddle.optimizer.Adam(learning_rate=CONFIG["learning_rate"], parameters=parameters)
    directory = OUT / ("smoke" if smoke else group)
    directory.mkdir(parents=True, exist_ok=False)
    with paddle.no_grad():
        initial = [p.numpy().copy() for p in parameters]
    architecture = dict(parameter=name, shape=list(classifier.weight.shape), rank=rank,
        upstream_encoder=type(neck.encoder).__name__,
        upstream_projection_parameters=[key for key, p in net.named_parameters()
            if key.startswith("head.ctc_encoder.") and len(p.shape) >= 2],
        factors=dict(U=list(parameters[0].shape), V=list(parameters[1].shape)),
        trainable_parameters=sum(math.prod(p.shape) for p in parameters),
        original_state_entries=len(reference), initial_state_changes=audit_state(net, reference),
        eval_dropout_layers=sum(isinstance(layer, paddle.nn.Dropout) for layer in net.sublayers()),
        eval_batchnorm_buffers=sum(key.endswith(("._mean", "._variance")) for key in reference))
    save_json(directory / "architecture.json", architecture)
    print(f"{group}: editing {architecture}", flush=True)
    fused = frozen_export_graph(net)
    fused_reference = {k: v.numpy() for k, v in fused.state_dict().items()}
    parity = check_parity(fused, fused.head.ctc_encoder, fused.head.ctc_head.fc, decoder, factors)
    with paddle.no_grad():
        unrep = net(paddle.to_tensor(np.load(OUT / "parity-probe.npz")["input"])).numpy()
    parity["unfused_checkpoint_vs_export"] = probability_difference(
        unrep, np.load(OUT / "parity-probe.npz")["probabilities"])
    parity["windows_ctc"] = check_windows_ctc()
    save_json(directory / "parity.json", parity)
    checkpoint(directory, 0, net, fused, classifier, name, reference, factors, decoder)
    labels, roles, schedule = training_data(decoder)
    use = Counter()
    lam, mu = GROUPS[group]
    loss_names = ("ctc_edit", "ctc_retain", "kl_retain", "relative_delta_squared")
    import json
    with (directory / "training.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        for row in schedule[:n_steps]:
            step = row["step"]
            batch = make_batch(row, labels, fused, fused.head.ctc_encoder)
            if smoke:
                print(f"Smoke {step}: prepared frozen features", flush=True)
            parts = losses(batch, classifier, factors)
            if smoke:
                print(f"Smoke {step}: computed losses", flush=True)
            total = parts[0] + parts[1] + lam * parts[2] + mu * parts[3]
            values = dict(zip(loss_names, (float(loss.item()) for loss in parts), strict=True))
            values.update(step=step, total=float(total.item()), lambda_kl=lam, mu=mu,
                          masked_retain_steps=int(batch[-1].sum().item()),
                          padded_retain_steps=int(math.prod(batch[-1].shape) - batch[-1].sum().item()))
            if not all(math.isfinite(x) for x in values.values()):
                raise ValueError(f"Non-finite objective {values}")
            if smoke or step in (1, *STEPS):
                for key, part in zip(loss_names, parts, strict=True):
                    if smoke:
                        print(f"Smoke {step}: measuring {key} gradient", flush=True)
                    component = paddle.grad([part], parameters, retain_graph=True)
                    values[f"grad_{key}"] = norm(component)
            total.backward()
            if any(p.grad is None for p in parameters):
                raise ValueError("Missing low-rank gradient")
            values["grad_u"], values["grad_v"] = [norm([p.grad]) for p in parameters]
            if not all(math.isfinite(x) for x in values.values()) or values["grad_v"] <= 0:
                raise ValueError(f"Invalid factor gradient {values}")
            if step == 1 and values["grad_u"] != 0:
                raise ValueError("Zero V must give zero U gradient at initialization")
            opt.step()
            opt.clear_grad()
            use.update(row["edit"] + row["retain"])
            log.write(json.dumps(values, allow_nan=False) + "\n")
            log.flush()
            if smoke or step % 10 == 0:
                print(group, step, round(values["total"], 6), round(values["kl_retain"], 7), flush=True)
            if step in STEPS or step == n_steps:
                audit_state(net, reference)
                audit_state(fused, fused_reference)
                checkpoint(directory, step, net, fused, classifier, name, reference, factors, decoder)
    if any(np.array_equal(p.numpy(), start) for p, start in zip(parameters, initial, strict=True)):
        raise ValueError("A low-rank factor did not change")
    if sum(use.values()) != n_steps * 16:
        raise ValueError("Actual usage does not match optimizer updates")
    save_json(directory / "usage.json", dict(steps=n_steps, counts={key: use[key] for key in sorted(roles)},
                                            draw_count=sum(use.values())))
    save_json(directory / "completed.json", dict(group=group, steps=n_steps, seed=SEED,
        audit_original_state=audit_state(net, reference), optimizer_updates=n_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_group(args.group, 3 if args.smoke else CONFIG["maximum_update_steps"], args.smoke)

