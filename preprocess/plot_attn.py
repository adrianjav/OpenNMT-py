import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(word, pred, attn):
    # attn is a tensor of size [prediction size, word size]
    fig, axis = plt.subplots(figsize=(len(pred), len(word)))
    axis.imshow(attn, cmap=plt.cm.Reds, interpolation='nearest')

    axis.set_xticks(range(len(word)))
    axis.set_xticklabels(word)
    axis.xaxis.tick_top()

    axis.set_yticks(range(len(pred)))
    axis.set_yticklabels(pred)

    axis.set_aspect('auto')
    plt.show()


def read_example(iter, show=True):
    word = next(iter).split()
    pred, attn = [], []
    for line in iter:
        line = line.replace("*", "").split()
        pred += [line[0]]
        attn += [np.array(line[1:], dtype=np.float)]
        if pred[-1] == "</s>":
            break

    if show:
        attn = np.stack(attn)
        plot_heatmap(word, pred, attn)

if __name__ == "__main__":
    iter = open(sys.argv[1], 'r') if len(sys.argv) >= 2 else sys.stdin
    samples = [int(x) for x in sys.argv[2:]] if len(sys.argv) >= 3 else None
    try:
        i = 1
        while True:
            read_example(iter, samples is None or i in samples)
            i += 1
    except StopIteration:
        pass
