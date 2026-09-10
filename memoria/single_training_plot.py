"""Dibuja la curva de métricas de validación de una sola ejecución a partir de su log.

Extrae del log las líneas bbox_mAP_copypaste y representa frente a la época el mAP,
el mAP@50, el mAP@75 y su desglose por tamaño de objeto. Es una utilidad de
inspección rápida de una ejecución concreta y muestra la figura por pantalla.
"""

import re
import matplotlib.pyplot as plt

LOG_PATH = "co_dino_swimL_121250.log"   # change if needed
OUTPUT = "bbox_mAP_curve.png"

METRIC_NAMES = [
    "mAP",
    "mAP@50",
    "mAP@75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
]

def extract_bbox_metrics(log_file):
    metrics = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "bbox_mAP_copypaste" in line:
                nums = re.findall(r"\d+\.\d+", line)
                if nums:
                    metrics.append([float(n) for n in nums])

    return metrics


def main():
    all_metrics = extract_bbox_metrics(LOG_PATH)
    
    if not all_metrics:
        print("❌ No bbox metrics found in log.")
        return

    epochs = list(range(1, len(all_metrics) + 1))

    # transpose: epochs × metrics → metrics × epochs
    metrics_by_type = list(zip(*all_metrics))

    plt.figure(figsize=(11, 6))

    for i, values in enumerate(metrics_by_type):
        label = METRIC_NAMES[i] if i < len(METRIC_NAMES) else None
        if label:
            plt.plot(epochs, values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation bbox Metrics per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # plt.savefig(OUTPUT, dpi=200)
    plt.show()

    print(f"✅ Metric plot saved as: {OUTPUT}")


if __name__ == "__main__":
    main()
