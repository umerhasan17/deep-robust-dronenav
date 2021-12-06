import matplotlib.pyplot as plt
import numpy as np


def plot_displacements(displacements_file):
    with open(displacements_file, 'rb') as file:
        disp = np.load(file)
        plt.plot(disp[:, 1], disp[:, 2])
        plt.plot(disp[:, 4], disp[:, 5])
        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    plot_displacements('data/nuevo_displacements.npy')
