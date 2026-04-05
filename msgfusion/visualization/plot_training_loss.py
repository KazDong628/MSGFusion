"""Loss history (.mat) to PNG plot; Agg backend for headless environments."""

import scipy.io as scio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_curve_from_matfile(mat_path: str, output_image_path: str) -> None:
    plt.cla()
    plt.clf()
    if not mat_path:
        return

    bundle = scio.loadmat(mat_path)
    loss = bundle["Loss"][0]
    horiz = range(0, len(loss))

    plt.plot(horiz, loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(output_image_path)


plot_training_loss_curve = save_loss_curve_from_matfile
showLossChart = save_loss_curve_from_matfile
