from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from dolfin import (
    UnitCubeMesh,
    VectorElement,
    FiniteElement,
    FunctionSpace,
    near,
    Expression,
    Constant,
    DirichletBC,
    TrialFunctions,
    TestFunctions,
    inner,
    grad,
    div,
    dx,
    assemble_system,
    KrylovSolver,
    Function,
    File,
)

#Costanti
NE = 15                                # Numero di suddivisioni per asse (mesh NE×NE×NE)
H = NE + 1                             # Numero di punti di campionamento lungo ogni asse
U_MAX = 1.0                            # Velocità massima del profilo parabolico di ingresso
TOL = 1e-6                             # Tolleranza per near(x, ...) nelle BC
RNG = np.random.default_rng()          # Generatore di numeri casuali
BETA_SHAPE = (0.5, 0.5)                # Parametri (alpha, beta) per la distribuzione Beta
SAMPLES_DIR = Path("samples-32")       # Cartella dove salvare i file .npz
PVD_DIR = Path("pvd")                  # Cartella dove salvare i primi 5 file .pvd per visualizzazione

# Creazione directory se non esistenti
SAMPLES_DIR.mkdir(exist_ok=True)
PVD_DIR.mkdir(exist_ok=True)

# Griglie regolari di coordinate (lunghezza H lungo ogni asse)
xs = np.linspace(0.0, 1.0, H)
ys = np.linspace(0.0, 1.0, H)
zs = np.linspace(0.0, 1.0, H)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")


def _beta() -> float:
    """
    Estrae un valore dalla distribuzione Beta(alpha, beta) definita in BETA_SHAPE.
    """
    return RNG.beta(*BETA_SHAPE)


def random_geom() -> tuple[float, float, float, float, float, float, float, float]:
    """
    Genera in modo casuale due rettangoli (inlet e outlet) sul cubo [0,1]^3.
    - Ciascun rettangolo ha dimensioni (dy, dz) tra 0.20 e 0.80.
    - Le posizioni y0, z0 sono scelte uniformemente in modo che il rettangolo stia interamente in [0,1]x[0,1].
    - Assicura che i centroidi di inlet e outlet non coincidano esattamente (scarto > 0.05).
    Ritorna:
        (in_y0, in_dy, in_z0, in_dz, out_y0, out_dy, out_z0, out_dz)
    """
    while True:
        # Dimensioni casuali tramite Beta → intervallo [0.20, 0.80]
        in_dy, in_dz = 0.20 + 0.60 * _beta(), 0.20 + 0.60 * _beta()
        out_dy, out_dz = 0.20 + 0.60 * _beta(), 0.20 + 0.60 * _beta()

        # Posizioni casuali y0, z0 garantendo il rettangolo in [0,1]
        in_y0 = RNG.uniform(0.0, 1.0 - in_dy)
        in_z0 = RNG.uniform(0.0, 1.0 - in_dz)
        out_y0 = RNG.uniform(0.0, 1.0 - out_dy)
        out_z0 = RNG.uniform(0.0, 1.0 - out_dz)

        # Centroidi dei due rettangoli
        in_cy, in_cz = in_y0 + in_dy / 2.0, in_z0 + in_dz / 2.0
        out_cy, out_cz = out_y0 + out_dy / 2.0, out_z0 + out_dz / 2.0

        # Assicura che i centroidi non siano troppo vicini (differenza > 0.05 in y o z)
        if abs(in_cy - out_cy) > 0.05 or abs(in_cz - out_cz) > 0.05:
            return (in_y0, in_dy, in_z0, in_dz, out_y0, out_dy, out_z0, out_dz)


def solve_stokes(
    geom: tuple[float, float, float, float, float, float, float, float],
    pvd_basename: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Risolve le equazioni di Stokes in un cubo con inlet/outlet rettangolari definiti da `geom`.
    Se `pvd_basename` è fornito, salva i primi 5 campioni di velocity e pressure in formato .pvd.

    Parametri:
        geom: tuple con (in_y0, in_dy, in_z0, in_dz, out_y0, out_dy, out_z0, out_dz)
        pvd_basename: Path (senza estensione) per salvare .pvd (solo se i < 5)

    Ritorna:
        u: array di forma (3, H, H, H) contenente i campioni di velocità [u_x, u_y, u_z]
        p: array di forma (H, H, H) contenente i campioni di pressione
    """
    # Estrai parametri geometrici
    in_y0, in_dy, in_z0, in_dz, out_y0, out_dy, out_z0, out_dz = geom

    # Crea mesh del cubo [0,1]^3
    mesh = UnitCubeMesh(NE, NE, NE)

    # Spazio V = [P2]^3 (velocità), Q = P1 (pressione), coppia Taylor–Hood stabile
    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V * Q)

    # Definizione delle condizioni al contorno (BC)
    # Funzioni lambda per individuare inlet e outlet sul bordo x=0 e x=1
    inlet = lambda x, on: (
        on and near(x[0], 0.0, TOL)
        and (in_y0 <= x[1] <= in_y0 + in_dy)
        and (in_z0 <= x[2] <= in_z0 + in_dz)
    )
    outlet = lambda x, on: (
        on and near(x[0], 1.0, TOL)
        and (out_y0 <= x[1] <= out_y0 + out_dy)
        and (out_z0 <= x[2] <= out_z0 + out_dz)
    )

    # Centro del rettangolo di ingresso (per profilo parabolico)
    yc, zc = in_y0 + in_dy / 2.0, in_z0 + in_dz / 2.0

    # Espressione simbolica per il profilo parabolico di ingresso:
    # u_x = U_MAX * (1 - ((2*(y - yc)/dy)^2)) * (1 - ((2*(z - zc)/dz)^2)), u_y = u_z = 0
    inflow = Expression(
        (
            "U*(1 - pow((2*(x[1] - yc)/dy), 2))"
            "* (1 - pow((2*(x[2] - zc)/dz), 2))",
            "0",
            "0"
        ),
        degree=2,
        U=U_MAX,
        yc=yc,
        dy=in_dy,
        zc=zc,
        dz=in_dz,
    )

    # BC 1: No-slip (u = 0) su tutte le facce eccetto inlet/outlet
    bc_noslip = DirichletBC(
        W.sub(0),
        Constant((0.0, 0.0, 0.0)),
        lambda x, on: on and not inlet(x, on) and not outlet(x, on),
    )
    # BC 2: Velocity inlet sul rettangolo inlet
    bc_inlet = DirichletBC(W.sub(0), inflow, inlet)
    # BC 3: Pressione zero (p = 0) sul rettangolo outlet
    bc_outlet = DirichletBC(W.sub(1), Constant(0.0), outlet)

    bcs = [bc_noslip, bc_inlet, bc_outlet]

    # -------------------------------------------------------------------
    # Definizione della forma variazionale per Stokes
    # -------------------------------------------------------------------
    (u, p), (v, q) = TrialFunctions(W), TestFunctions(W)

    # a(u, p; v, q) = ∫ grad(u):grad(v) dx + ∫ p * div(v) dx + ∫ q * div(u) dx
    a = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx

    # Termino di carico nullo (L(v, q) = 0)
    L = Constant(0.0) * q * dx

    # Assembla il sistema A * sol = b con BC
    A, b = assemble_system(a, L, bcs)

    # Costruisci matrice di precondizionamento P
    P, _ = assemble_system(
        inner(grad(u), grad(v)) * dx + p * q * dx,
        L,
        bcs
    )

    # Risoluzione del sistema lineare con solutore MINRES + ILU
    solver = KrylovSolver("minres", "ilu")
    solver.set_operators(A, P)

    sol = Function(W)
    solver.solve(sol.vector(), b)

    # Scompone la soluzione in velocità u_sol e pressione p_sol
    u_sol, p_sol = sol.split(deepcopy=True)

    # Salvataggio .pvd (se richiesto, per i primi 5 campioni)
    if pvd_basename is not None:
        vel_path = str(pvd_basename.with_name(f"{pvd_basename.stem}_velocity.pvd"))
        pres_path = str(pvd_basename.with_name(f"{pvd_basename.stem}_pressure.pvd"))
        File(vel_path) << u_sol
        File(pres_path) << p_sol

    # Campionamento su griglia regolare (H×H×H)
    u_arr = np.zeros((3, H, H, H), dtype=np.float32)  # canali: u_x, u_y, u_z
    p_arr = np.zeros((H, H, H), dtype=np.float32)

    # Loop su ciascuna coordinata di griglia (xs[ix], ys[iy], zs[iz])
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                ux, uy, uz = u_sol(x, y, z)  # eval delle funzioni vel/pres
                u_arr[:, ix, iy, iz] = (ux, uy, uz)
                p_arr[ix, iy, iz] = p_sol(x, y, z)

    return u_arr, p_arr


def build_input(geom: tuple[float, float, float, float, float, float, float, float]) -> np.ndarray:
    """
    Costruisce l'array di input 3D con tre canali:
      1. inlet_mask   (1 se il punto è all'inlet, 0 altrove)
      2. inlet_vel    (profilo parabolico di velocità solo su inlet, zero altrove)
      3. outlet_mask  (1 se il punto è all'uscita, 0 altrove)

    Ritorna:
        array di shape (3, H, H, H) con dtype float32.
    """
    in_y0, in_dy, in_z0, in_dz, out_y0, out_dy, out_z0, out_dz = geom

    # Maschera booleana per inlet (x ≃ 0 e y,z dentro il rettangolo inlet)
    inlet_mask = (
        np.isclose(X, 0.0, TOL)
        & (Y >= in_y0) & (Y <= in_y0 + in_dy)
        & (Z >= in_z0) & (Z <= in_z0 + in_dz)
    ).astype(np.float32)

    # Maschera booleana per outlet (x ≃ 1 e y,z dentro il rettangolo outlet)
    outlet_mask = (
        np.isclose(X, 1.0, TOL)
        & (Y >= out_y0) & (Y <= out_y0 + out_dy)
        & (Z >= out_z0) & (Z <= out_z0 + out_dz)
    ).astype(np.float32)

    # Costruzione del profilo di velocità parabolico solo sui punti dell'inlet
    inlet_vel = np.zeros_like(inlet_mask, dtype=np.float32)
    if inlet_mask.any():
        yc, zc = in_y0 + in_dy / 2.0, in_z0 + in_dz / 2.0
        # Applica la formula parabolica: U_MAX * (1 - ((y - yc)*2/in_dy)^2)*(1 - ((z - zc)*2/in_dz)^2)
        mask_indices = inlet_mask == 1.0
        inlet_vel[mask_indices] = (
            U_MAX
            * (1.0 - ((Y[mask_indices] - yc) * 2.0 / in_dy) ** 2)
            * (1.0 - ((Z[mask_indices] - zc) * 2.0 / in_dz) ** 2)
        )

    # Stack dei tre canali (shape: 3×H×H×H)
    return np.stack([inlet_mask, inlet_vel, outlet_mask], axis=0)


def main(nsamples: int) -> None:
    """
    Genera nsamples di flussi Stokes. Per ciascun campione:
      - Genera un inlet/outlet casuale
      - Risolve Stokes → ottiene u (3×H×H×H) e p (H×H×H)
      - Salva i primi 5 in formato .pvd (velocità e pressione)
      - Salva tutti i campioni in .npz (inp, out)
    """
    for i in range(nsamples):
        geom = random_geom()
        base = PVD_DIR / f"sample_{i:05d}"

        # Per i primi 5, genero anche i file .pvd per visualizzazione
        if i < 5:
            u, p = solve_stokes(geom, pvd_basename=base)
        else:
            u, p = solve_stokes(geom, pvd_basename=None)

        # Costruisco l'input (3×H×H×H) e output (4×H×H×H = [u_x,u_y,u_z,p])
        inp = build_input(geom).astype(np.float32)
        out = np.concatenate([u, p[np.newaxis, ...]], axis=0)

        # Salvataggio in formato compresso .npz
        np.savez_compressed(
            SAMPLES_DIR / f"sample_{i:05d}.npz",
            inp=inp,
            out=out,
        )

        # Aggiorna barra di avanzamento
        print(f"\r{i + 1:>4}/{nsamples} campioni generati", end="")

    print(f"\n✓ Dataset generato in «{SAMPLES_DIR}», primi 5 PVD in «{PVD_DIR}»")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera un dataset di flussi di Stokes in un cubo con inlet/outlet casuali"
    )
    parser.add_argument(
        "-n", "--nsamples",
        type=int,
        default=500,
        help="Numero di simulazioni da generare (default: 500)"
    )
    args = parser.parse_args()
    main(args.nsamples)
