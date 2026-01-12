# 3D Neural Operators for CFD Acceleration (Stokes Flow)

## 🎯 Obiettivo del Progetto
Il progetto mira a creare un modello basato su Deep Learning per accelerare la simulazione di fluidi in 3D. L'obiettivo è sostituire o affiancare i costosi solver numerici classici con un operatore neurale (FNO3d) in grado di predire istantaneamente campi di velocità e pressione data una geometria e condizioni al contorno variabili.

Il caso studio specifico riguarda il **flusso di Stokes** in un dominio cubico unitario con ingressi e uscite randomizzati.

---

## ⚙️ 1. Generazione Dati (Fisica e Simulazione)
Il modulo di generazione dati (`generate_samples.py`) utilizza il metodo degli elementi finiti tramite la libreria **FEniCS (legacy dolfin)** per creare il *Ground Truth*.

### Setup Fisico
* **Equazioni:** Equazioni di Stokes (flusso viscoso a bassi numeri di Reynolds).
* **Dominio:** Cubo unitario $[0,1]^3$.
* **Mesh:** Mesh tetraedrica strutturata (`UnitCubeMesh`), discretizzata con elementi finiti misti (Taylor-Hood) per garantire stabilità LBB:
    * Velocità: Elementi Lagrangiani di ordine 2 ($P2$).
    * Pressione: Elementi Lagrangiani di ordine 1 ($P1$).

### Condizioni al Contorno (BC) e Variabilità
Per addestrare la rete a generalizzare, ogni campione presenta una geometria diversa:
* **Inlet/Outlet:** Due aree rettangolari generate casualmente sulle facce opposte del cubo ($x=0$ e $x=1$).
* **Posizionamento:** Le dimensioni e la posizione dei rettangoli seguono una **distribuzione Beta** ($\alpha=0.5, \beta=0.5$) per garantire varianza nei dati.
* **Profilo di Velocità:** All'inlet viene imposto un **profilo parabolico** di velocità ($U_{max}=1.0$).
* **Pressione:** Imposta a 0 all'outlet.
* **Pareti:** Condizione di *no-slip* (velocità zero) su tutte le altre facce.

### Output del Solver
Il solver calcola la soluzione esatta, che viene poi interpolata su una griglia cartesiana regolare per essere usata dalla rete neurale.
* **Input Tensore:** (3, H, H, H) $\rightarrow$ [Maschera Inlet, Velocità Inlet, Maschera Outlet].
* **Output Tensore:** (4, H, H, H) $\rightarrow$ [Velocità X, Velocità Y, Velocità Z, Pressione].

---

## 🧠 2. Architettura Neurale
Il cuore del sistema (`train_fno3d.py`) è un **Fourier Neural Operator 3D (FNO3d)**, implementato tramite la libreria `neuralop`.

### Perché FNO?
A differenza delle reti classiche, gli FNO imparano un'approssimazione dell'operatore integrale nel dominio delle frequenze. Questo permette:
1.  **Discretization Invariance:** Il modello può essere addestrato a bassa risoluzione e valutato a risoluzione più alta (Super-Resolution).
2.  **Efficienza:** Le convoluzioni spettrali catturano dipendenze meglio delle classiche DNN.

### Iperparametri del Modello
* **Modi di Fourier:** 16 per asse.
* **Canali nascosti (Width):** 64.
* **Input:** 3 canali (geometria e condizioni al contorno).
* **Output:** 4 canali (campo vettoriale velocità + scalare pressione).
* **Embedding Posizionale:** Grid embedding attivato per fornire coordinate spaziali esplicite.

---

## 📉 3. Pipeline di Training
Lo script di training implementa una strategia robusta per garantire la convergenza su problemi fisici 3D.

* **Loss Function:**
    * Principale: **H1 Loss** (Norma di Sobolev). Questa loss penalizza non solo l'errore sui valori puntuali, ma anche l'errore sulle **derivate spaziali**, cruciale per rispettare la fisica del fluido.
    * Validazione: Monitoraggio anche della **LpLoss** (L2 relativa).
* **Ottimizzatore:** AdamW con Weight Decay ($1e^{-3}$) per regolarizzazione.
* **Scheduler:** *Cosine Annealing* per ridurre il learning rate gradualmente fino a fine training.
* **Multi-Resolution Validation:** Il modello viene validato simultaneamente su dataset a diverse risoluzioni ($16^3$, $20^3$, $32^3$) per testare la proprietà di invarianza alla discretizzazione.

---

## 📊 4. Visualizzazione e Analisi Risultati
Lo script `visualize_results.py` fornisce strumenti diagnostici per analizzare qualitativamente le predizioni rispetto al Ground Truth (GT).

Le visualizzazioni generate includono:
1.  **Slice di Magnitudine:** Confronto `GT vs Prediction` della magnitudine della velocità su piani ortogonali (XY, XZ, YZ) a diverse profondità. Include mappe di errore assoluto.
2.  **Campi Vettoriali (Quiver):** Visualizzazione delle frecce di flusso per verificare se la rete ha imparato la direzione corretta del fluido e la formazione del profilo parabolico.
3.  **Campo di Pressione:** Verifica della caduta di pressione dall'inlet all'outlet (gradiente di pressione che guida il flusso).

---

## 📂 Struttura della Repository

```text
neural_operators_3d/
│
├── checkpoints/             # Pesi del modello salvati (fno3d.pt) e indici di test
├── samples-XX/              # Dataset generati (npz) a risoluzione XX^3
├── pvd/                     # File Paraview per visualizzazione 3D nativa
│
├── generate_samples.py      # Generatore fisico (FEniCS -> NPZ)
├── train_fno3d.py           # Script di addestramento (PyTorch/NeuralOp)
├── visualize_results.py     # Script di inferenza e plotting (Matplotlib)
└── old_scripts_examples/    # Prototipi precedenti (2D Darcy, Shallow Water)
