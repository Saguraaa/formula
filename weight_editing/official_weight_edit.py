"""Fixed-budget detector and OCR comparisons from official training weights."""
import argparse
import copy
import math
import os
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from weight_editing import weight_edit as old
from evaluate_controlled_ocr import (
    ANNOTATIONS, DATA, OCR, ROOT, candidate_boxes, score, detect_crops, evaluate_crops, iou, match, normalize,
    read_bgr, recognize, records, save_json, save_jsonl, sha, summarize, load_json, box_to_yolo,
)

OUT = ROOT / "runs/official-weight-edit-v1"
SOURCE = old.SOURCE
INITIAL_OCR = ROOT / "weights/PP-OCRv6_medium_rec_pretrained.pdparams"
INITIAL_YOLO = ROOT / "weights/yolo11n.pt"
OFFICIAL_OCR = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", str(Path.home() / ".paddlex"))) / "official_models/PP-OCRv6_medium_rec"
SEED = 42
RANK = 8
UPDATES = 200
OCR_BATCH = 16
YOLO_BATCH = 8
OCR_LR = 1e-4
YOLO_LR = 1e-3
MODES = ("full", "edit")
COMBINATIONS = {"initial": ("initial","initial"), "full": ("full","full"),
                "edit": ("edit","edit"), "edit-yolo": ("edit","full"), "edit-ocr": ("full","edit")}


def cycles(names, count):
    rng = random.Random(SEED)
    sequence = []
    while len(sequence) < count:
        items = list(names)
        rng.shuffle(items)
        sequence.extend(items)
    return sequence[:count]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_json(old.controlled.OUT / "manifest.json")
    annotations, splits = old.annotations_and_split()
    old.verify_labels("train", annotations, splits, audit)
    old.verify_labels("val", annotations, splits, audit)
    if sha(ANNOTATIONS) != audit["source_annotations_sha256"] or sha(DATA/"split_manifest.json") != audit["split_sha256"]:
        raise ValueError("Data differs from audited experiment")
    baseline = load_json(old.controlled.RESULTS/"O0/summary.json")
    if old.model_hashes(OFFICIAL_OCR) != baseline["model_sha256"]:
        raise ValueError("Official OCR inference assets changed")
    if sha(INITIAL_OCR) != load_json(old.controlled.OUT/"initial-weight-verification.json")["initial_sha256"]:
        raise ValueError("Official OCR training checkpoint changed")
    labels = old.controlled.OUT/"train.txt"
    ocr_sequence = cycles(sorted(name for name, _ in records(labels)), UPDATES*OCR_BATCH)
    yolo_sequence = cycles(sorted(splits["train"]), UPDATES*YOLO_BATCH)
    inventory = []
    for page in sorted(splits["train"]):
        image = DATA/"yolo/images/train"/page
        label = DATA/"yolo/labels/train"/f"{Path(page).stem}.txt"
        inventory.append((page, sha(image), sha(label)))
    save_json(OUT/"train-schedule.json", dict(ocr=ocr_sequence, yolo=yolo_sequence))
    save_json(OUT/"provenance.json", dict(
        seed=SEED, rank=RANK, update_count=UPDATES, ocr_batch=OCR_BATCH, yolo_batch=YOLO_BATCH,
        ocr_lr=OCR_LR, yolo_lr=YOLO_LR, train_pages=len(inventory), train_crops=len(records(labels)),
        ocr_draws=len(ocr_sequence), yolo_draws=len(yolo_sequence), ocr_unique=len(set(ocr_sequence)),
        yolo_unique=len(set(yolo_sequence)), original_ocr_image_hash_manifest=sha(old.controlled.OUT/"O1/train-image-sha256.tsv"),
        input_sha256={str(path):sha(path) for path in (INITIAL_OCR, INITIAL_YOLO, 
            old.O3/"export.yaml", ANNOTATIONS, DATA/"split_manifest.json", labels, OUT/"train-schedule.json")},
        official_ocr_inference_sha256=old.model_hashes(OFFICIAL_OCR), train_yolo_hashes=inventory))
    print(f"Prepared {len(inventory)} training pages and {len(set(ocr_sequence))} scheduled OCR crops.", flush=True)


def train_ocr(mode, steps, directory):
    from weight_editing import train_weight_edit as runtime

    pd, F = runtime.paddle, runtime.F
    pd.set_device("gpu:0")
    runtime.seed()
    # Isolated process reuses the audited model builder with the official checkpoint.
    runtime.BEST = INITIAL_OCR
    net, neck, layer, key, initial, decoder, rank = runtime.build()
    runtime.seed()
    factors = runtime.Factors(layer.weight.shape, rank) if mode == "edit" else None
    if factors is None:
        for module in (net.backbone, neck, layer):
            for p in module.parameters():
                p.stop_gradient = False
        parameters = [p for p in net.parameters() if not p.stop_gradient]
    else:
        parameters = factors.parameters()
    optimizer = pd.optimizer.Adam(learning_rate=OCR_LR, parameters=parameters)
    unused_names = []
    dictionary = {c:i for i,c in enumerate(decoder.character)}
    labels = dict(records(old.controlled.OUT/"train.txt"))
    sequence = load_json(OUT/"train-schedule.json")["ocr"][:steps*OCR_BATCH]
    if len(sequence) != steps*OCR_BATCH:
        raise ValueError("Missing scheduled OCR samples")
    image_hashes = dict(records(old.controlled.OUT/"O1/train-image-sha256.tsv"))
    for name in set(sequence):
        if sha(OCR/name) != image_hashes[name]:
            raise ValueError(f"Changed training crop: {name}")
    if factors is not None:
        with pd.no_grad():
            x, _ = runtime.input_batch(sequence[:OCR_BATCH])
            x = pd.to_tensor(x)
            h = neck(net.backbone(x))
            np.testing.assert_array_equal(F.softmax(F.linear(h, layer.weight + factors.delta(), layer.bias), -1).numpy(), net(x).numpy())
    cpu = pd.CPUPlace()
    losses = []
    with (directory/"loss.tsv").open("w",encoding="utf-8") as log:
        for start in range(0,len(sequence),OCR_BATCH):
            step = start//OCR_BATCH+1
            names = sequence[start:start+OCR_BATCH]
            x, _ = runtime.input_batch(names)
            features = neck(net.backbone(pd.to_tensor(x)))
            t = features.shape[1]
            targets = [[dictionary[c] for c in labels[name]] for name in names]
            if any(not target or 0 in target or len(target)+sum(a==b for a,b in zip(target,target[1:])) > t for target in targets):
                raise ValueError(f"Illegal CTC label: {names}")
            y = np.zeros((len(targets),max(map(len,targets))),dtype=np.int32)
            for i,target in enumerate(targets):
                y[i,:len(target)] = target
            weight = layer.weight if factors is None else layer.weight + factors.delta()
            logits = F.linear(features,weight,layer.bias)
            loss = runtime.ctc_loss_vector(logits.transpose([1,0,2]), pd.to_tensor(y,place=cpu),
                    pd.to_tensor([t]*len(names),dtype="int64",place=cpu),
                    pd.to_tensor([len(v) for v in targets],dtype="int64",place=cpu)).mean()
            value = float(loss.item())
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite OCR loss at {step}")
            loss.backward()
            if step == 1 and any(p.grad is None for p in parameters):
                if factors is not None:
                    raise ValueError("A low-rank OCR factor has no gradient")
                # Some deploy/fusion tensors are allocated in the original model
                # but are inactive in the unfused training checkpoint's graph.
                unused_names = [name for name, p in net.named_parameters() if not p.stop_gradient and p.grad is None]
                for p in parameters:
                    if p.grad is None:
                        p.stop_gradient = True
                parameters = [p for p in parameters if not p.stop_gradient]
                if layer.weight.grad is None or not any(p.grad is not None for p in net.backbone.parameters()):
                    raise ValueError("OCR full gradient did not reach backbone and classifier")
                optimizer = pd.optimizer.Adam(learning_rate=OCR_LR, parameters=parameters)
                print("Inactive OCR tensors:", unused_names, flush=True)
            optimizer.step()
            optimizer.clear_grad()
            losses.append(value)
            log.write(f"{step}\t{value}\n")
            log.flush()
            if step<=2 or step%20==0:
                print("ocr",mode,step,value,flush=True)
    with pd.no_grad():
        probe_names = sequence[:4]
        probe_x, _ = runtime.input_batch(probe_names)
        probe_x = pd.to_tensor(probe_x)
        if factors is not None:
            runtime.audit_state(net,initial)
            delta = factors.delta()
            relative = runtime.norm([delta])/runtime.norm([layer.weight])
            probe_features = neck(net.backbone(probe_x))
            unmerged = F.softmax(F.linear(probe_features, layer.weight + delta, layer.bias), axis=-1).numpy()
            np.savez(directory/"factors.npz",U=factors.u.numpy(),V=factors.v.numpy())
            layer.weight.set_value(layer.weight + delta)
            changed = runtime.audit_state(net,initial,(key,))
            np.testing.assert_allclose(net(probe_x).numpy(), unmerged, atol=1e-6, rtol=1e-5)
        else:
            relative = None
            changed = [k for k,v in net.state_dict().items() if not np.array_equal(v.numpy(),initial[k])]
        probe_text = [text for text, _ in decoder(net(probe_x).numpy())]
    save_json(directory/"probe.json",dict(crops=probe_names, texts=probe_text))
    pd.save(net.state_dict(),str(directory/"model.pdparams"))
    save_json(directory/"training.json", dict(mode=mode, steps=steps, trainable=sum(math.prod(p.shape) for p in parameters),
        changed=changed, inactive_tensors=unused_names, relative_weight_delta=relative,
        first_loss=losses[0], last_loss=losses[-1],
        checkpoint_sha256=sha(directory/"model.pdparams")))


def train_yolo(mode, steps, directory):
    import cv2
    import torch
    from torch import nn
    from torch.nn.utils import parametrize
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    class Update(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.u = nn.Parameter(torch.empty(weight.shape[0], RANK, device=weight.device).normal_(std=.01))
            self.v = nn.Parameter(torch.zeros(weight[0].numel(), RANK, device=weight.device))

        def forward(self, weight):
            return weight + (self.u @ self.v.T).reshape_as(weight)

    torch.manual_seed(SEED)
    model = YOLO(str(INITIAL_YOLO)).model.float()
    head = model.model[-1]
    if head.nc != 80 or head.end2end or head.reg_max != 16:
        raise ValueError("Unexpected YOLO11 detection head")
    projections = []
    candidates = []
    for box, cls, stride in zip(head.cv2,head.cv3,head.stride,strict=True):
        previous = [layer for layer in cls.modules() if isinstance(layer,nn.Conv2d)][-2]
        matches = (box[-1],previous)
        if any(layer.kernel_size!=(1,1) or layer.groups!=1 for layer in matches):
            raise ValueError("Unexpected detection projection layout")
        candidates.extend(matches)
        old_output = cls[-1]
        new = nn.Conv2d(old_output.in_channels,1,1)
        nn.init.constant_(new.bias,math.log(5/(640/float(stride))**2))
        cls[-1] = new
        projections.append(new)
    head.nc = 1
    head.no = 1 + 4*head.reg_max
    model.yaml["nc"] = 1
    model.nc = 1
    model.names = {0:"formula"}
    model.args = get_cfg()
    model = model.cuda()
    names = {id(module):name for name,module in model.named_modules()}
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # Both groups start from identical new class projections and the official transferable weights.
    init_digest = __import__("hashlib").sha256()
    for k, v in initial.items():
        init_digest.update(k.encode())
        init_digest.update(v.numpy().tobytes())
    if mode=="full":
        torch.save({"model":copy.deepcopy(model).cpu().eval()},directory/"initial.pt")
    standard_trainable = {k for k in initial if not k.endswith("dfl.conv.weight")}
    for name,p in model.named_parameters():
        p.requires_grad_(mode=="full" and name in standard_trainable)
    adapters = {}
    if mode=="edit":
        for layer in projections:
            layer.requires_grad_(True)
        for layer in candidates:
            block = Update(layer.weight)
            parametrize.register_parametrization(layer,"weight",block)
            if not torch.equal(layer.weight,layer.parametrizations.weight.original):
                raise ValueError("YOLO zero update differs from start")
            adapters[names[id(layer)]]=block
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=YOLO_LR)
    model.train()
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            module.eval()
    sequence=load_json(OUT/"train-schedule.json")["yolo"][:steps*YOLO_BATCH]
    source=load_json(OUT/"provenance.json")
    expected={page:(img_hash,label_hash) for page,img_hash,label_hash in source["train_yolo_hashes"]}
    data={}
    size=640
    for name in set(sequence):
        path=DATA/"yolo/images/train"/name
        labels=DATA/"yolo/labels/train"/f"{Path(name).stem}.txt"
        if (sha(path),sha(labels))!=expected[name]:
            raise ValueError(f"Changed detector training sample: {name}")
        raw=read_bgr(path)
        h,w=raw.shape[:2]
        scale=min(size/w,size/h)
        nw,nh=round(w*scale),round(h*scale)
        left,top=round((size-nw)/2-.1),round((size-nh)/2-.1)
        canvas=np.full((size,size,3),114,np.uint8)
        canvas[top:top+nh,left:left+nw]=cv2.resize(raw,(nw,nh))
        targets=[]
        for row in labels.read_text(encoding="utf-8").splitlines():
            cls,x,y,bw,bh=map(float,row.split())
            if cls!=0 or any(not 0<=n<=1 for n in (x,y,bw,bh)) or min(bw,bh)<=0:
                raise ValueError(f"Invalid detector label: {name}: {row}")
            targets.append([(x*w*scale+left)/size,(y*h*scale+top)/size,bw*w*scale/size,bh*h*scale/size])
        data[name]=(canvas[:,:,::-1].transpose(2,0,1).copy(),targets)
    del source
    with (directory/"loss.tsv").open("w",encoding="utf-8") as log:
        for start in range(0,len(sequence),YOLO_BATCH):
            step=start//YOLO_BATCH+1
            batch=[data[name] for name in sequence[start:start+YOLO_BATCH]]
            indices=[i for i,(_,boxes) in enumerate(batch) for _ in boxes]
            boxes=[box for _, targets in batch for box in targets]
            batch_inputs=dict(img=torch.as_tensor(np.stack([img for img,_ in batch]),device="cuda").float()/255,
                batch_idx=torch.tensor(indices,device="cuda"), cls=torch.zeros((len(boxes),1),device="cuda"),
                bboxes=torch.tensor(boxes,dtype=torch.float32,device="cuda").reshape(-1,4))
            optimizer.zero_grad(set_to_none=True)
            parts, components=model.loss(batch_inputs)
            loss=parts.sum()/len(batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite detector loss at {step}")
            loss.backward()
            if step==1 and any(p.grad is None for p in parameters):
                raise ValueError("A YOLO trainable parameter has no gradient")
            optimizer.step()
            detached = components.values() if isinstance(components, dict) else components.flatten()
            values = [float(loss.detach()), *(float(value.detach()) for value in detached)]
            log.write("\t".join(map(str,[step,*values]))+"\n")
            log.flush()
            if step<=2 or step%20==0:
                print("yolo",mode,step,values,flush=True)
    model.eval()
    relative={}
    if mode=="edit":
        with torch.no_grad():
            probe=batch_inputs["img"][:1]
            before=model(probe)[0]
            factors={}
            for layer in candidates:
                name=names[id(layer)]
                block=adapters[name]
                original=layer.parametrizations.weight.original
                if not torch.equal(original.cpu(),initial[name+".weight"]):
                    raise ValueError(f"Frozen YOLO original changed: {name}")
                update=(block.u @ block.v.T).reshape_as(original)
                relative[name]=float(update.norm()/original.norm())
                factors[name]={"U":block.u.cpu(),"V":block.v.cpu()}
                parametrize.remove_parametrizations(layer,"weight",leave_parametrized=True)
            torch.testing.assert_close(before,model(probe)[0])
            torch.save(factors,directory/"factors.pt")
    changed=[name for name,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[name])]
    if mode=="edit":
        allowed={names[id(module)]+suffix for module in projections for suffix in (".weight",".bias")}
        allowed |= {names[id(module)]+".weight" for module in candidates}
        edited_weights = set(changed) & {name+".weight" for name in adapters}
        if not set(changed)<=allowed or not edited_weights:
            raise ValueError(f"Unexpected or entirely unchanged detector edit: {changed}")
    model.criterion=None
    torch.save({"model":model.cpu()},directory/"model.pt")
    save_json(directory/"training.json",dict(mode=mode,steps=steps,trainable=sum(p.numel() for p in parameters),
                 changed=changed,start_tensor_sha256=init_digest.hexdigest(),relative_weight_delta=relative,
                 checkpoint_sha256=sha(directory/"model.pt")))


def export(mode):
    from train_ocr import launch
    directory=OUT/f"ocr-{mode}"
    cfg=yaml.safe_load((old.O3/"export.yaml").read_text(encoding="utf-8"))
    cfg["Global"].update(pretrained_model=(directory/"model.pdparams").as_posix(),
        save_model_dir=directory.as_posix(),save_inference_dir=(directory/"inference").as_posix())
    config=directory/"export.yaml"
    config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    launch(SOURCE,"export_model.py",config,directory/"export.log")



def detector_path(mode):
    if mode == "initial":
        return OUT/"yolo-full/initial.pt"
    return OUT/f"yolo-{mode}/model.pt"


def ocr_path(mode):
    return OFFICIAL_OCR if mode == "initial" else OUT/f"ocr-{mode}/inference"


def run_detection(split):
    from ultralytics import YOLO

    _, splits=old.annotations_and_split()
    for mode in ("initial",*MODES):
        path=detector_path(mode)
        directory=OUT/split/f"detector-{mode}"
        directory.mkdir(parents=True,exist_ok=True)
        model=YOLO(str(path))
        pages=[]
        for page in splits[split]:
            source=DATA/"yolo/images"/split/page
            image=read_bgr(source)
            pages.append(dict(page=page, sha256=sha(source), detections=detect_crops(model,image,0,.5,640)))
        save_json(directory/"boxes.json",dict(weights=sha(path),pages=pages))
        print(split,mode,len(pages),"detected pages",flush=True)


def page_rows(rec, split, detector_mode, annotations, splits, detection_cache=None):
    cached=load_json(detection_cache or OUT/split/f"detector-{detector_mode}"/"boxes.json")
    if cached["weights"]!=sha(detector_path(detector_mode)) or [r["page"] for r in cached["pages"]]!=splits[split]:
        raise ValueError("Wrong detection cache")
    output=[]
    false_positives=0
    for page in cached["pages"]:
        name=page["page"]
        path=DATA/"yolo/images"/split/name
        if sha(path)!=page["sha256"]:
            raise ValueError(f"Changed evaluation page: {name}")
        image=read_bgr(path)
        height,width=image.shape[:2]
        boxes=page["detections"]
        crops=[image[y1:y2,x1:x2].copy() for x1,y1,x2,y2 in (box["bbox"] for box in boxes)]
        predictions=[dict(box,**pred) for box,pred in zip(boxes,recognize(rec,crops),strict=True)]
        targets=[]
        for box in annotations[name]:
            x,y,w,h=box_to_yolo(box,width,height)
            raw=box.get("text")
            targets.append(dict(bbox=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height],
                                text=raw if isinstance(raw,str) else ""))
        pairs=match(predictions,targets,.50)
        inv={pred:target for target,pred in pairs.items()}
        false_positives+=len(predictions)-len(pairs)
        for index,target in enumerate(targets):
            pred=predictions[pairs[index]] if index in pairs else None
            row=dict(page=name,box_index=index,matched=pred is not None,
                     target_bbox=target["bbox"], bbox=pred["bbox"] if pred else None,
                     iou=iou(target["bbox"],pred["bbox"]) if pred else None, **score(target["text"],pred))
            if pred is None:
                row["candidates"]=candidate_boxes(target["bbox"],predictions,inv)
                row["match_failure"]="no_iou_match"
            output.append(row)
    return output,false_positives


def grade_summary(rows):
    result=summarize(rows)
    counts=result["counts"]
    wrong_correct=sum(r["truth_grade"] is False and r["ocr_grade"] is False for r in rows if r["expected"])
    right_correct=sum(r["truth_grade"] is True and r["ocr_grade"] is True for r in rows if r["expected"])
    right_targets=counts["gradeable"]-counts["wrong_targets"]
    wrong_targets=counts["wrong_targets"]
    counts.update(wrong_correct=wrong_correct,right_correct=right_correct,
                  right_rejected=sum(r["truth_grade"] is True and r["ocr_grade"] is False for r in rows if r["expected"]))
    result["metrics"].update(
        wrong_recall=wrong_correct/wrong_targets if wrong_targets else None,
        balanced_grading_accuracy=(right_correct/right_targets+wrong_correct/wrong_targets)/2
        if right_targets and wrong_targets else None)
    return result


def evaluate(split):
    annotations,splits=old.annotations_and_split()
    old.verify_labels(split,annotations,splits,load_json(old.controlled.OUT/"manifest.json"))
    for mode in ("initial",*MODES):
        rec=old.recognizer(ocr_path(mode))
        if mode != "initial":
            probe = load_json(OUT/f"ocr-{mode}"/"probe.json")
            actual = recognize(rec, [read_bgr(OCR/name) for name in probe["crops"]])
            actual_text = [r["ocr_text"] for r in actual]
            if actual_text != probe["texts"]:
                raise ValueError(f"OCR export disagrees with dynamic model on probe: {mode}")
        rows=old.evaluate_split(rec,split,annotations,splits)
        dest=OUT/split/f"ocr-{mode}"
        dest.mkdir(parents=True,exist_ok=True)
        save_jsonl(dest/"rows.jsonl",rows)
        save_json(dest/"metrics.json",grade_summary(rows))
        for combo,(detector,ocr) in COMBINATIONS.items():
            if ocr!=mode:
                continue
            rows,fp=page_rows(rec,split,detector,annotations,splits)
            summary=grade_summary(rows)
            if summary["counts"]["labeled"]!=old.COUNTS[split]:
                raise ValueError("Whole-page and crop counts differ")
            summary["counts"]["false_positive_boxes"]=fp
            dest=OUT/split/combo
            dest.mkdir(parents=True,exist_ok=True)
            save_jsonl(dest/"rows.jsonl",rows)
            save_json(dest/"metrics.json",summary)
            print(split,combo,summary["counts"],flush=True)
        del rec


def lock():
    target=OUT/"test-lock.json"
    if target.exists():
        raise FileExistsError(target)
    detector_runs=[load_json(OUT/f"yolo-{mode}"/"training.json") for mode in MODES]
    if detector_runs[0]["start_tensor_sha256"]!=detector_runs[1]["start_tensor_sha256"]:
        raise ValueError("Detector modes have different starts")
    files=[OUT/"train-schedule.json"]
    for mode in ("initial",*MODES):
        files.append(detector_path(mode))
        files.extend(ocr_path(mode)/item for item in old.MODEL_FILES)
    for stage in ("yolo","ocr"):
        for mode in MODES:
            meta=load_json(OUT/f"{stage}-{mode}"/"training.json")
            if meta["steps"]!=UPDATES:
                raise ValueError(f"Incomplete {stage}, {mode}")
    for combo in COMBINATIONS:
        files.append(OUT/"val"/combo/"metrics.json")
    save_json(target,dict(combinations=COMBINATIONS,updates=UPDATES,conf=.5,iou=.5,imgsz=640,
                          method="Prespecified final step; all combinations tested without choosing on test results.",
                          sha256={str(path):sha(path) for path in files}))


def check_lock():
    data=load_json(OUT/"test-lock.json")
    for path,digest in data["sha256"].items():
        if sha(Path(path))!=digest:
            raise ValueError(f"Test lock input changed: {path}")


def track(metrics):
    import socket
    import swanlab

    root = OUT/"swanlab"
    root.mkdir(exist_ok=True)
    try:
        socket.getaddrinfo("api.swanlab.cn", 443)
        reason = "offline tracking for the local fixed-budget experiment"
    except OSError as exc:
        reason = f"SwanLab DNS unavailable: {exc}"
    status = []
    for name in ("yolo-full", "yolo-edit", "ocr-full", "ocr-edit", "evaluation"):
        index = root/f"{name}.json"
        if index.exists():
            status.append(load_json(index))
            continue
        with swanlab.init(project="ModelAdjust", name=f"official-weight-edit-v1-{name}",
                          group="official-weight-edit-v1", mode="offline", log_dir=str(root),
                          settings=swanlab.Settings(interactive=False)) as run:
            if name == "evaluation":
                run.log({f"{split}/{group}/{key}": value for split, by_group in metrics.items()
                         for group, item in by_group.items() if group != "paired"
                         for key, value in (item["metrics"] | item["counts"]).items() if value is not None})
            else:
                for line in (OUT/name/"loss.tsv").read_text(encoding="utf-8").splitlines():
                    step, loss, *extra = line.split("\t")
                    fields = dict(zip(("box", "cls", "dfl"), map(float, extra)))
                    run.log({"train/total":float(loss), **{f"train/{k}":v for k,v in fields.items()}}, step=int(step))
            item = dict(name=name, path=str(run.dir), run_id=run.id, synchronized=False, reason=reason)
        save_json(index,item)
        status.append(item)
    save_json(OUT/"swanlab-status.json",status)


def report():
    check_lock()
    metrics={}
    for split in ("val","test"):
        metrics[split]={key:load_json(OUT/split/key/"metrics.json") for key in (*COMBINATIONS,*(f"ocr-{m}" for m in ("initial",*MODES)))}
        groups={key:old.read_jsonl(OUT/split/key/"rows.jsonl") for key in COMBINATIONS if key!="initial"}
        comparisons={}
        for name,candidate in groups.items():
            if name=="full":
                continue
            evidence=[]
            for ref,edit in zip(groups["full"],candidate,strict=True):
                if any(ref[k]!=edit[k] for k in ("page","box_index","expected","truth_grade","target_bbox")):
                    raise ValueError("Mismatched ground truth for paired grade analysis")
                truth=ref["truth_grade"]
                changed=ref["ocr_grade"] is not edit["ocr_grade"]
                if truth is not None and changed:
                    evidence.append(dict(before=ref,after=edit,
                        recovered=ref["ocr_grade"] is not truth and edit["ocr_grade"] is truth,
                        regressed=ref["ocr_grade"] is truth and edit["ocr_grade"] is not truth,
                        new_wrong_accept=truth is False and ref["ocr_grade"] is not True and edit["ocr_grade"] is True))
            comparisons[name]={flag:sum(row[flag] for row in evidence) for flag in ("recovered","regressed","new_wrong_accept")}
            save_jsonl(OUT/split/f"{name}-versus-full.jsonl",evidence)
        metrics[split]["paired"]=comparisons
    save_json(OUT/"comparison.json",metrics)
    lines=["# 官方权重低秩编辑与全量微调对照","",
           "同一固定整页划分，种子 42，每个适配模型 200 次更新。YOLO 使用同样的 1,600 次抽样，"
           "OCR 使用同样的 3,200 张互异原始训练裁图。每个组件的两组使用相同损失、Adam 和学习率。"
           "所有模型使用末步预先锁定结果，未在测试集调阈值。以下是已有测试集上的单种子探索。","",
           "YOLO 从官方 COCO 权重复制全部兼容张量，替换三路最后一层类别投影，使用相同随机初始化接到单类 formula。"
           "两组均优化新增投影。full 更新其余可训练检测权重，edit 仅在三路原框投影和三路原分类特征投影上学习秩 8 增量。"
           "两组固定 BatchNorm 运行统计。","",
           "OCR 从官方 PP-OCRv6_medium_rec 训练检查点开始。full 更新 CTC 推理路径，edit 只编辑 CTC 分类矩阵。"
           "两组以相同真实 OCR 标签训练 CTC，不使用答题正确性改写文本。整页复用同一算术判题器。","",
           "模型对照名 initial 指同一单类 YOLO 初始结构与官方 OCR，full 与 edit 分别是两个组件都用该方法。"
           "edit-yolo 只替换检测组件，edit-ocr 只替换识别组件。","",
           "| 划分 | 组合 | 匹配框 | 全文读对 | 判题准确率 | 平衡判题准确率 | 错题判对 | 无法判定 |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split,scores in metrics.items():
        for method in (*COMBINATIONS,*(f"ocr-{m}" for m in ("initial",*MODES))):
            counts=scores[method]["counts"]
            m=scores[method]["metrics"]
            lines.append(f"| {split} | {method} | {counts.get('matched','—')} | {counts['exact']}/{counts['labeled']} | "
                         f"{m['grading_accuracy']:.2%} | {m['balanced_grading_accuracy']:.2%} | "
                         f"{counts['wrong_accepted']} | {counts['undetermined']} |")
    lines.extend(["","## 逐题判题变化","",
                  "| 划分 | 相对于 full | 修复 | 退步 | 新增错题判对 |","|---|---|---:|---:|---:|"])
    for split,scores in metrics.items():
        for method,counts in scores["paired"].items():
            lines.append(f"| {split} | {method} | {counts['recovered']} | {counts['regressed']} | {counts['new_wrong_accept']} |")
    lines.extend(["", "## 可训练参数量", "", "| 模型 | 全量适配 | 低秩编辑 |", "|---|---:|---:|"])
    for component in ("yolo", "ocr"):
        full, edit = (load_json(OUT/f"{component}-{mode}"/"training.json")["trainable"] for mode in MODES)
        lines.append(f"| {component} | {full:,} | {edit:,} |")
    lines.extend(["", "SwanLab 曲线和最终指标均以离线方式保存。同步状态见 swanlab-status.json。"])
    (OUT/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    track(metrics)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("prepare","train-yolo","train-ocr","export","detect","eval","lock","report"))
    cli.add_argument("--mode",choices=MODES)
    cli.add_argument("--split",choices=("val","test"))
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    if args.stage=="prepare":
        prepare()
    elif args.stage.startswith("train-"):
        if not args.mode:
            cli.error("--mode required")
        component=args.stage.removeprefix("train-")
        folder=(OUT/"smoke" if args.smoke else OUT)/f"{component}-{args.mode}"
        folder.mkdir(parents=True,exist_ok=False)
        (train_yolo if component=="yolo" else train_ocr)(args.mode,3 if args.smoke else UPDATES,folder)
    elif args.stage=="export":
        if not args.mode:
            cli.error("--mode required")
        export(args.mode)
    elif args.stage in ("detect","eval"):
        if not args.split:
            cli.error("--split required")
        if args.split=="test":
            check_lock()
        (run_detection if args.stage=="detect" else evaluate)(args.split)
    elif args.stage=="lock":
        lock()
    else:
        report()


if __name__=="__main__":
    main()

