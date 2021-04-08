import matplotlib.pyplot as plt


def plot_lung(
    image,
    title: str = 'Pulmao',
) -> None:
    plt.imshow(image,cmap='gray')
    plt.title(title)
    plt.show()
    return None
