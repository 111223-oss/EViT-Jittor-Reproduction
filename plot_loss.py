import re
import matplotlib.pyplot as plt

log_file = "jittor_train.log"

losses = []
epochs = []

pattern = re.compile(r"Epoch\s+(\d+),\s*Loss:\s*([0-9.]+)")

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            ep = int(m.group(1))
            loss = float(m.group(2))
            epochs.append(ep)
            losses.append(loss)

if not losses:
    raise RuntimeError("没有从日志里解析到 Loss，请确认日志里有类似：Epoch 1, Loss: 2.3749")

plt.plot(epochs, losses, marker="o")
plt.title("Jittor Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("jittor_loss.png", dpi=200)
print("Saved -> jittor_loss.png")
