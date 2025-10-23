import os, json
import numpy as np
from typing import Dict, Tuple, List
from model import MLP, CNN
from optim import SGD, Adam

def _ensure_X(X, model_type: str):
    X = np.asarray(X)
    if model_type == "mlp":
        # MLP는 2D (N, D) 필요 → 자동 flatten
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
    elif model_type == "cnn":
        # CNN은 (N, C, H, W); (N, H, W)이면 채널=1 추가
        if X.ndim == 3:
            X = X[:, None, :, :]
        if X.ndim != 4:
            raise ValueError("For CNN, X must be (N,C,H,W) or (N,H,W).")
    else:
        raise ValueError("model_type must be 'mlp' or 'cnn'")
    return X

def iterate_minibatches(X, y, batch_size=128, shuffle=True):
    N = len(X)
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, N, batch_size):
        sel = idx[i:i + batch_size]
        yield X[sel], y[sel]

def build_model(
    model_type: str,
    input_info,
    output_size: int,
    hidden_sizes: List[int] = [128],
    use_batchnorm: bool = False,       # 현재 MLP에서만 의미 있음
    use_dropout: bool = False,
    dropout_ratio: float = 0.5,
    seed: int = 42,
):
    if model_type == "mlp":
        if isinstance(input_info, (tuple, list)):
            D = int(np.prod(input_info))
        else:
            D = int(input_info)
        return MLP(
            input_size=D,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            use_batchnorm=use_batchnorm,
            use_dropout=use_dropout,
            dropout_ratio=dropout_ratio,
            seed=seed,
            weight_decay=0.0,
        )

    elif model_type == "cnn":
        if not (isinstance(input_info, (tuple, list)) and len(input_info) == 3):
            raise ValueError("For CNN, input_info must be (C,H,W)")
        C, H, W = map(int, input_info)
        hidden_fc = hidden_sizes[0] if len(hidden_sizes) > 0 else 128
        return CNN(
            input_shape=(C, H, W),
            output_size=output_size,
            hidden_fc=hidden_fc,
            seed=seed,
            use_dropout=use_dropout,
            dropout_ratio=dropout_ratio,
        )

    else:
        raise ValueError("model_type must be 'mlp' or 'cnn'")

def train_one_model(
    Xtr, ytr, Xva, yva,
    input_info,
    output_size: int,
    model_type: str = "cnn",
    hidden_sizes: List[int] = [128],
    lr: float = 0.001,
    batch_size: int = 128,
    epochs: int = 20,
    use_batchnorm: bool = False,
    use_dropout: bool = False,
    dropout_ratio: float = 0.5,
    optimizer_name: str = "adam",
    seed: int = 42,
):
    Xtr = _ensure_X(Xtr, model_type)
    Xva = _ensure_X(Xva, model_type)

    net = build_model(
        model_type=model_type,
        input_info=input_info,
        output_size=output_size,
        hidden_sizes=list(hidden_sizes),
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
        dropout_ratio=dropout_ratio,
        seed=seed,
    )

    optim = Adam(lr=lr) if optimizer_name.lower() == "adam" else SGD(lr=lr)

    best = {"acc": -1.0, "state": None, "epoch": -1}
    for ep in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(Xtr, ytr, batch_size=batch_size, shuffle=True):
            loss = net.loss(xb, yb, train_flg=True)
            net.backward()
            params, grads = net.params_and_grads()
            optim.update(params, grads)

        tr_acc = net.accuracy(Xtr, ytr)
        va_acc = net.accuracy(Xva, yva)
        print(f"[{model_type.upper()}][Epoch {ep:03d}] train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")

        if va_acc > best["acc"]:
            best["acc"] = va_acc
            best["state"] = net.snapshot_state()
            best["epoch"] = ep

    if best["state"] is not None:
        net.load_state(best["state"])
        print(f"=> Restored best epoch {best['epoch']} (val_acc={best['acc']:.4f})")

    return net, best

def evaluate_and_save(net, Xte, yte, idx2label: Dict[int, str], out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    model_type = "cnn" if np.asarray(Xte).ndim >= 3 else "mlp"
    Xte = _ensure_X(Xte, model_type)

    acc = net.accuracy(Xte, yte)
    print(f"[Validation] accuracy = {acc:.4f}  (N={len(yte)})")

    scores = net.predict(Xte, train_flg=False)
    pred = np.argmax(scores, axis=1)
    C = max(int(max(yte.max(), pred.max())) + 1, len(idx2label))

    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(yte, pred):
        cm[t, p] += 1

    np.save(os.path.join(out_dir, "cm.npy"), cm)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
    print("Saved:", os.path.join(out_dir, "cm.npy"), "and labels.json")
    return acc, cm

def grid_search_train_cnn(
    Xtr_all, ytr_all,
    input_info, output_size,
    hidden_sizes_list=[(128,), (256,)],
    lrs=(0.001, 0.0005),
    batch_sizes=(64, 128),
    epochs_list=(5,),
    use_dropout_list=(True, False),
    dropout_ratio_list=(0.5, 0.3),
    seed=42,
):
    Xtr_all = _ensure_X(Xtr_all, "cnn")
    ytr_all = np.asarray(ytr_all, dtype=np.int64)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(Xtr_all))
    rng.shuffle(idx)
    n_val = max(1, int(len(Xtr_all) * 0.2))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xva, yva = Xtr_all[va_idx], ytr_all[va_idx]
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]

    best = {"acc": -1.0, "cfg": None, "state": None}
    for hs in hidden_sizes_list:
        for lr in lrs:
            for bs in batch_sizes:
                for ep in epochs_list:
                    for udo in use_dropout_list:
                        for dr in dropout_ratio_list:
                            print(f"=== try hs={hs}, lr={lr}, bs={bs}, ep={ep}, do={udo}, dr={dr}")
                            net, info = train_one_model(
                                Xtr, ytr, Xva, yva,
                                input_info, output_size,
                                model_type="cnn",
                                hidden_sizes=list(hs),
                                lr=lr, batch_size=bs, epochs=ep,
                                use_dropout=udo, dropout_ratio=dr,
                                optimizer_name="adam", seed=seed,
                            )
                            if info["acc"] > best["acc"]:
                                best.update(
                                    acc=info["acc"],
                                    cfg={
                                        "hidden_sizes": list(hs),
                                        "lr": lr,
                                        "batch_size": bs,
                                        "epochs": ep,
                                        "use_dropout": udo,
                                        "dropout_ratio": dr,
                                    },
                                    state=net.snapshot_state(),
                                )
    if best["cfg"] is None:
        raise RuntimeError("grid_search_train_cnn: no model trained")

    cfg = best["cfg"]
    net = build_model(
        "cnn", input_info, output_size,
        hidden_sizes=cfg["hidden_sizes"],
        use_dropout=cfg["use_dropout"],
        dropout_ratio=cfg["dropout_ratio"],
        seed=seed,
    )
    net.load_state(best["state"])
    return net, cfg
