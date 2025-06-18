import re
import matplotlib.pyplot as plt

log_file_path = "log.txt"

labels = []
losses = []

pattern = r"Epoch \[(\d+)/\d+\], Step \[(\d+)/\d+\], Loss: ([\d.]+)"

with open(log_file_path, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            loss = float(match.group(3))

            labels.append(f"E{epoch}S{step}")
            losses.append(loss)

plt.figure(figsize=(16, 6))
plt.plot(range(len(losses)), losses, label='Loss', marker='o')

n = 100
plt.xticks(ticks=range(0, len(labels), n), labels=labels[::n], rotation=90)

plt.xlabel('Epoch/Step')
plt.ylabel('Loss')
plt.title('Loss vs Epoch/Step')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
