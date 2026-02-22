# coding: utf-8
import os, pickle, time, subprocess, sys
import numpy as np
import jittor as jt
from jittor import nn
from evit_jittor_min import EViTMin

jt.flags.use_cuda = 1
jt.set_global_seed(0)
np.random.seed(0)

# 当前 CIFAR-10 解压后的目录
DATA_DIR = os.path.expanduser("~/.cache/jittor/dataset/cifar_data/cifar-10-batches-py")

# 输出目录：日志、csv、曲线图
OUT_DIR = os.path.join(os.getcwd(), "outputs_jittor")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "train_log.txt")
CSV_PATH = os.path.join(OUT_DIR, "metrics.csv")
LOSS_PNG = os.path.join(OUT_DIR, "loss_curve.png")
ACC_PNG  = os.path.join(OUT_DIR, "acc_curve.png")


def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(d[b"labels"], dtype=np.int32)
    return x, y


def load_cifar10(data_dir):
    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
        xs.append(x); ys.append(y)
    x_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    x_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    return x_train, y_train, x_test, y_test


def eval_acc(model, x_np, y_np, batch=256):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(x_np), batch):
        xb = jt.array(x_np[i:i+batch])
        logits = model(xb)
        pred = logits.data.argmax(axis=1)   # numpy
        gt = y_np[i:i+batch]
        correct += int((pred == gt).sum())
        total += len(gt)
    return 100.0 * correct / total


def write_text(path, text, mode="a"):
    with open(path, mode, encoding="utf-8") as f:
        f.write(text)


def ensure_matplotlib():
    """
    如果没装 matplotlib，就尝试 pip 安装。
    安装失败也不抛异常，返回 False。
    """
    try:
        import matplotlib  # noqa
        return True
    except Exception:
        pass

    print("[Warn] matplotlib 未安装，尝试自动安装：python -m pip install matplotlib")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib  # noqa
        return True
    except Exception as e:
        print("[Warn] 自动安装 matplotlib 失败（但训练日志/CSV仍会生成）:", str(e))
        return False


def plot_curves(epochs, losses, accs):
    ok = ensure_matplotlib()
    if not ok:
        return

    import matplotlib.pyplot as plt

    # 美观
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    # 1) Loss
    plt.figure(figsize=(8.5, 5.0), dpi=150)
    plt.plot(epochs, losses, marker="o", linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(LOSS_PNG)
    plt.show(block=False)
    plt.pause(0.8)
    plt.close()

    # 2) Acc
    plt.figure(figsize=(8.5, 5.0), dpi=150)
    plt.plot(epochs, accs, marker="o", linewidth=2)
    plt.title("Test Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(ACC_PNG)
    plt.show(block=False)
    plt.pause(0.8)
    plt.close()

    print("Saved figures:")
    print(" -", LOSS_PNG)
    print(" -", ACC_PNG)

    # WSL/Ubuntu
    try:
        subprocess.Popen(["xdg-open", LOSS_PNG], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["xdg-open", ACC_PNG],  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    # 数据目录检查
    if not os.path.isdir(DATA_DIR):
        print("ERROR: 找不到 DATA_DIR:", DATA_DIR)
        print("请确认目录里有 data_batch_1~5 和 test_batch")
        return

    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    print("Train:", x_train.shape, y_train.shape, "Test:", x_test.shape, y_test.shape)

    # 模型（简化 EViT）
    model = EViTMin(
        img_size=32, patch=4, num_classes=10,
        embed_dim=192, depth=6, heads=3,
        keep_rate=0.5, prune_after=2
    )

    opt = nn.Adam(model.parameters(), lr=3e-4)

    batch_size = 128
    epochs = 10
    N = x_train.shape[0]
    idx = np.arange(N)

    # 清空日志/CSV
    write_text(LOG_PATH, "", mode="w")
    write_text(CSV_PATH, "epoch,loss,test_acc\n", mode="w")

    losses, accs = [], []

    print("Start training...")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        total_loss = 0.0
        steps = 0

        model.train()
        for i in range(0, N, batch_size):
            j = idx[i:i+batch_size]
            xb = jt.array(x_train[j])
            yb = jt.array(y_train[j]).int32()

            logits = model(xb)
            loss = nn.cross_entropy_loss(logits, yb)
            opt.step(loss)

            total_loss += float(loss.data)
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        acc = eval_acc(model, x_test, y_test, batch=256)

        losses.append(avg_loss)
        accs.append(acc)

        line = f"Epoch {ep:03d}, Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%\n"
        print(line.strip())
        write_text(LOG_PATH, line, mode="a")
        write_text(CSV_PATH, f"{ep},{avg_loss:.6f},{acc:.4f}\n", mode="a")

    print("Done. Time(s):", round(time.time() - t0, 2))
    print("Saved:")
    print(" - Log:", LOG_PATH)
    print(" - CSV:", CSV_PATH)

    # 训练完画图 + 保存 PNG
    epochs_list = list(range(1, epochs + 1))
    plot_curves(epochs_list, losses, accs)


if __name__ == "__main__":
    main()
