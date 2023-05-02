from matplotlib import pyplot as plt


def get_value_from_logs(line, s, digits=4):
    start = line.index(s)
    value = line[start + len(s) : start + len(s) + digits]
    return float(value)


with open("logs/train/fold_3_exp1/log.txt") as fp:
    logs = fp.read()

logs_list = logs.split("*** ")
metrics = {"train": {"losses": [], "ious": []}, "val": {"losses": [], "ious": []}}
for line in logs_list:
    if line.startswith("Validation"):
        miou = get_value_from_logs(line, "mIoU: ")
        metrics["val"]["ious"].append(miou)
        loss = get_value_from_logs(line, "Avg L: ", 7)
        metrics["val"]["losses"].append(loss)
    else:
        miou = get_value_from_logs(line, "mIoU: ")
        metrics["train"]["ious"].append(miou)
        loss = get_value_from_logs(line, "Avg L: ", 7)
        metrics["train"]["losses"].append(loss)

plt.plot(metrics["train"]["ious"])
plt.plot(metrics["val"]["ious"])
plt.legend(["train", "val"])
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.show()
plt.plot(metrics["train"]["losses"])
plt.plot(metrics["val"]["losses"])
plt.legend(["train", "val"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
