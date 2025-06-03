import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from neuralop.models.fno import FNO3d

# Classe per il dataset generato 3D (carica file .npz)
class CFD3DDataset(Dataset):
    """
    Dataset che restituisce per ogni indice:
      - 'x': tensor (3, H, H, H) corrispondente all'input [inlet_mask, inlet_vel, outlet_mask]
      - 'y': tensor (4, H, H, H) corrispondente all'output [u_x, u_y, u_z, p]
    """

    def __init__(self, samples_dir: str):
        self.files = sorted(
            os.path.join(samples_dir, fname)
            for fname in os.listdir(samples_dir)
            if fname.endswith(".npz")
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        inp = torch.from_numpy(data["inp"]).float()   # shape (3, H, H, H)
        out = torch.from_numpy(data["out"]).float()   # shape (4, H, H, H)
        return {"x": inp, "y": out}


# Funzioni di utilità per estrazione e visualizzazione
def magnitude(u: np.ndarray) -> np.ndarray:
    """
    Calcola la magnitudine del campo vettoriale u di shape (3, H, H, H).
    Restituisce array 3D (H, H, H).
    """
    return np.linalg.norm(u, axis=0)


def absolute_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calcola l'errore assoluto |a - b| fra due array a e b.
    """
    return np.abs(a - b)


def take_slice(vol: np.ndarray, axis: int, k: int) -> np.ndarray:
    """
    Estrae una slice dall'array vol in base all'asse specificato:
      - Se vol.ndim == 3: restituisce vol[k,:,:] (axis=0), vol[:,k,:] (axis=1) o vol[:,:,k] (axis=2)
      - Se vol.ndim == 4 e vol.shape[0] in (3,4): interpreta il primo indice come canale e restituisce
        vol[:,k,:,:] (axis=0), vol[:,:,k,:] (axis=1), vol[:,:,:,k] (axis=2)
      - Altrimenti usa vol.take(indices=k, axis=axis).
    """
    if vol.ndim == 3:
        if axis == 0:
            return vol[k]
        if axis == 1:
            return vol[:, k]
        return vol[:, :, k]

    if vol.ndim == 4 and vol.shape[0] in (3, 4):
        if axis == 0:
            return vol[:, k, :, :]
        if axis == 1:
            return vol[:, :, k, :]
        return vol[:, :, :, k]

    return vol.take(indices=k, axis=axis)


def _add_colorbar(ax, im) -> None:
    """
    Aggiunge una colorbar all'asse ax per l'immagine im, posizionandola a destra.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)


# Funzioni di plotting
def plot_magnitude_slices(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    axis: int,
    ks: list[int],
    cmap: str = "viridis",
    err_cmap: str = "magma_r",
) -> None:
    """
    Confronta la magnitudine del campo vero (u_true) e predetto (u_pred), mostrando
    la slice di magnitudine e l'errore assoluto su più profondità ks lungo l'asse axis.
    """
    # Calcola le magnitudini in tutto il volume
    uT = magnitude(u_true)
    uP = magnitude(u_pred)

    fig, axs = plt.subplots(
        len(ks), 3, figsize=(14, 4 * len(ks)), constrained_layout=True
    )

    for i, k in enumerate(ks):
        sT = take_slice(uT, axis, k)
        sP = take_slice(uP, axis, k)
        err = absolute_error(sP, sT)

        # Immagine ground truth
        im0 = axs[i, 0].imshow(sT, cmap=cmap)
        axs[i, 0].set_title(f"Ground Truth Magnitude\n(axis={axis}, k={k})")
        axs[i, 0].axis("off")

        # Immagine predizione
        im1 = axs[i, 1].imshow(sP, cmap=cmap)
        axs[i, 1].set_title(f"Predicted Magnitude\n(axis={axis}, k={k})")
        axs[i, 1].axis("off")

        # Immagine errore assoluto
        im2 = axs[i, 2].imshow(err, cmap=err_cmap)
        axs[i, 2].set_title(f"Absolute Error\n(axis={axis}, k={k})")
        axs[i, 2].axis("off")

        # Aggiungi colorbar solo alla prima riga
        if i == 0:
            _add_colorbar(axs[0, 0], im0)
            _add_colorbar(axs[0, 1], im1)
            _add_colorbar(axs[0, 2], im2)

    fig.suptitle(
        f"Velocity Magnitude Slices (GT vs Pred vs Abs Error)\nAxis = {axis}",
        fontsize=16,
        y=1.02,
    )
    plt.show()


def plot_quiver_slices(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    axis: int,
    ks: list[int],
    step: int = 2,
) -> None:
    """
    Mostra i campi vettoriali in slice multiple per confronto ground truth vs predizione.
    - axis: 0=yz-plane, 1=xz-plane, 2=xy-plane
    - ks: lista di profondità (indice k) da campionare
    - step: passo per il quiver
    """
    # Per ogni slice, riduci il campo 3D (3,H,H,H) a 2 componenti per quiver
    comp = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]

    for k in ks:
        # Estrai la slice per ground truth e predizione
        uvT = take_slice(u_true, axis, k)[list(comp)]
        uvP = take_slice(u_pred, axis, k)[list(comp)]

        # Coordinate reticolari per quiver
        Y, X = np.meshgrid(
            np.arange(uvT.shape[1]), np.arange(uvT.shape[2]), indexing="ij"
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # Quiver per ground truth
        ax[0].quiver(
            Y[::step, ::step],
            X[::step, ::step],
            uvT[0, ::step, ::step],
            uvT[1, ::step, ::step],
        )
        ax[0].set_title(f"GT Vector Field\n(axis={axis}, k={k})")
        ax[0].axis("off")
        ax[0].invert_yaxis()

        # Quiver per predizione
        ax[1].quiver(
            Y[::step, ::step],
            X[::step, ::step],
            uvP[0, ::step, ::step],
            uvP[1, ::step, ::step],
        )
        ax[1].set_title(f"Predicted Vector Field\n(axis={axis}, k={k})")
        ax[1].axis("off")
        ax[1].invert_yaxis()

        fig.suptitle(
            f"Velocity Vector Comparison\nAxis = {axis}, Slice k={k}",
            fontsize=16,
            y=1.02,
        )
        plt.show()


def plot_pressure_slices(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    axis: int,
    ks: list[int],
    cmap: str = "viridis",
    err_cmap: str = "magma_r",
) -> None:
    """
    Confronta il campo di pressione vero (p_true) e predetto (p_pred) su slice multiple:
      - non centra i campi (perché stiamo mostrando l'errore assoluto)
      - visualizza GT, Pred e errore assoluto su ciascuna slice in ks lungo axis
    """
    fig, axs = plt.subplots(
        len(ks), 3, figsize=(14, 4 * len(ks)), constrained_layout=True
    )

    for i, k in enumerate(ks):
        sT = take_slice(p_true, axis, k)
        sP = take_slice(p_pred, axis, k)
        err_abs = absolute_error(sP, sT)

        # Pressure GT
        im0 = axs[i, 0].imshow(sT, cmap=cmap)
        axs[i, 0].set_title(f"GT Pressure\n(axis={axis}, k={k})")
        axs[i, 0].axis("off")

        # Pressure Predizione
        im1 = axs[i, 1].imshow(sP, cmap=cmap)
        axs[i, 1].set_title(f"Predicted Pressure\n(axis={axis}, k={k})")
        axs[i, 1].axis("off")

        # Errore assoluto pressione
        im2 = axs[i, 2].imshow(err_abs, cmap=err_cmap)
        axs[i, 2].set_title(f"Absolute Error Pressure\n(axis={axis}, k={k})")
        axs[i, 2].axis("off")

        # Aggiungi colorbar solo alla prima riga
        if i == 0:
            _add_colorbar(axs[0, 0], im0)
            _add_colorbar(axs[0, 1], im1)
            _add_colorbar(axs[0, 2], im2)

    fig.suptitle(
        f"Pressure Field Slices (GT vs Pred vs Abs Error)\nAxis = {axis}",
        fontsize=16,
        y=1.02,
    )
    plt.show()


# Nota: la funzione plot_divergence è stata rimossa/disabilitata perché non richiesta.

# Funzione main
def main():
    parser = argparse.ArgumentParser(
        description="Visualizza risultati di FNO3d su campioni 3D"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples-16",
        help="Directory dei file .npz contenenti dati",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/fno3d.pt",
        help="Percorso al file dei pesi del modello",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=5,
        help="Indice del campione da visualizzare",
    )
    parser.add_argument(
        "--axes",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Assi da analizzare: 0=yz, 1=xz, 2=xy",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="Profondità delle slice per magnitudine e quiver",
    )
    parser.add_argument(
        "--pressure_ks",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="Profondità delle slice per pressione",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Passo di sottocampionamento per i plot quiver",
    )
    args = parser.parse_args()

    # Carica i dati dal file .npz selezionato
    ds = CFD3DDataset(args.samples_dir)
    data = np.load(ds.files[args.sample_idx])

    u_true = data["out"][:3]  # campi velocità (3, H, H, H)
    p_true = data["out"][3]   # campo pressione (H, H, H)
    inp = torch.from_numpy(data["inp"]).unsqueeze(0)  # aggiunge dimensione batch

    # Carica il modello FNO3d con i pesi salvati
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO3d(
        n_modes_height=16,
        n_modes_width=16,
        n_modes_depth=16,
        hidden_channels=64,
        in_channels=3,
        out_channels=4,
        positional_embedding="grid",
    ).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)

    # Esegui inferenza sul singolo campione scelto
    with torch.no_grad():
        out = model(inp.to(device)).squeeze(0).cpu().numpy()
    u_pred, p_pred = out[:3], out[3]

    # 1) Plot slices di magnitudine (GT vs Pred vs Errore)
    for axis in args.axes:
        plot_magnitude_slices(u_true, u_pred, axis, args.ks)

    # 2) Plot campi vettoriali (quiver) per GT vs Pred
    for axis in args.axes:
        plot_quiver_slices(u_true, u_pred, axis, args.ks, step=args.step)

    # 3) Plot slices di pressione (GT vs Pred vs Errore Assoluto)
    for axis in args.axes:
        plot_pressure_slices(p_true, p_pred, axis, args.pressure_ks)


if __name__ == "__main__":
    main()
