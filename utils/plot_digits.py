import matplotlib.pyplot as plt

def plot_images(images, labels, predictions=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        title = f"Label: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()