import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from neuralop.models.fno import FNO3d
from neuralop.losses import LpLoss, H1Loss
from neuralop.training import Trainer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

# Classe per il dataset generato dalla simulazione
class CFD3DDataset(Dataset):
    """
    Dataset che carica file .npz contenenti:
      - 'inp': array (3, H, H, H) -> [inlet_mask, inlet_vel, outlet_mask]
      - 'out': array (4, H, H, H) -> [u_x, u_y, u_z, p]
    Il dataset restituisce un dizionario con chiavi 'x' (input) e 'y' (output).
    """

    def __init__(self, samples_dir: str):
        # Lista ordinata di tutti i file .npz nella cartella indicata
        self.files = sorted(
            os.path.join(samples_dir, fname)
            for fname in os.listdir(samples_dir)
            if fname.endswith(".npz")
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        inp = torch.from_numpy(data["inp"]).float()   # (3, H, H, H)
        out = torch.from_numpy(data["out"]).float()   # (4, H, H, H)
        return {"x": inp, "y": out}


def main():
    # Configurazioni
    samples_dir = "samples-16"      # Directory con file .npz per training/validation
    samples_dir_20 = "samples-20"   # Dataset con risoluzione 20^3 per validazione
    samples_dir_32 = "samples-32"   # Dataset con risoluzione 32^3 per validazione

    batch_size = 8
    n_epochs   = 100
    train_frac = 0.8
    lr         = 1e-3

    # Creazione istanze di dataset
    dataset     = CFD3DDataset(samples_dir)
    dataset_20  = CFD3DDataset(samples_dir_20)
    dataset_32  = CFD3DDataset(samples_dir_32)

    # Fissiamo il seme per la riproducibilità della divisione train/test
    gen = torch.Generator().manual_seed(1234)

    # Calcola dimensioni training/test basate su train_frac
    n_train = int(train_frac * len(dataset))
    n_test  = len(dataset) - n_train

    # Suddividi il dataset originale in training e test
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)

    # Salva gli indici di test su file per poterli riutilizzare in seguito
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(test_ds.indices, "checkpoints/test_indices.pt")

    # Creazione dei DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    # DataLoader per i dataset di validazione a risoluzioni differenti
    test_loader_20 = DataLoader(
        dataset_20,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader_32 = DataLoader(
        dataset_32,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Impostazione del device (CUDA se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Definizione del modello FNO3d
    model = FNO3d(
        n_modes_height=16,
        n_modes_width=16,
        n_modes_depth=16,
        hidden_channels=64,
        in_channels=3,    # inlet_mask, inlet_vel, outlet_mask
        out_channels=4,   # u_x, u_y, u_z, p
        positional_embedding="grid",
    ).to(device)

    # DataProcessor per preprocess dei batch
    processor = DefaultDataProcessor().to(device)

    # Configurazione del Trainer
    trainer = Trainer(
        model=model,
        device=device,
        n_epochs=n_epochs,
        verbose=True,
        data_processor=processor,
    )

    # Ottimizzatore AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3
    )
    # Scheduler Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs
    )

    # Funzioni di loss: L2 e H1 in dimensione 3D
    loss_l2 = LpLoss(d=3, p=2)
    loss_h1 = H1Loss(d=3)

    # Avvio del training
    trainer.train(
        train_loader,
        {
            "val": test_loader,
            "val_20": test_loader_20,
            "val_32": test_loader_32,
        },
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=loss_h1,            # Loss principale per il training (H1)
        eval_losses={"l2": loss_l2, "h1": loss_h1},  # Metriche di validazione
    )

    # Salva i pesi del modello al termine del training
    torch.save(model.state_dict(), "checkpoints/fno3d.pt")
    print("Training completato. Pesi salvati in «checkpoints/fno3d.pt»")


if __name__ == "__main__":
    main()
