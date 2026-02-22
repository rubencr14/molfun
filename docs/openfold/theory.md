# How OpenFold Works — Theory

---

## Introduction: From Sequence to Structure

**The central goal of OpenFold (and AlphaFold2) is to predict the 3D atomic coordinates of a protein from its amino acid sequence alone.**

The key insight is that this problem naturally decomposes into two complementary sources of information:

- **Evolutionary information** — across millions of years of evolution, homologous proteins have accumulated mutations. Residues that mutate *together* (co-evolve) are likely in physical contact. The pattern of conservation and co-mutation across thousands of related sequences encodes implicit structural constraints.
- **Geometric information** — once we have hypotheses about which residues are close in space, we can reason directly about distances, angles, and rigid-body frames in 3D.

**These two streams of information are processed in parallel and continuously exchange information with each other throughout the network.** Rather than solving structure prediction in a single pass, OpenFold iteratively refines both a set of pairwise geometric hypotheses and an evolutionary profile, letting each inform the other.

**The pipeline can be thought of as a progressive dimensionality build-up:**
1. We start with a **1D primary sequence** — a list of amino acid tokens.
2. We lift it to **2D** — a matrix of pairwise relationships between all residue pairs (distances, orientations, co-evolutionary signals).
3. We finally decode that 2D representation into **3D atomic coordinates** using a module that reasons about rigid-body frames.

---

## The Two Main Representations

All the computation in the Evoformer operates over two tensors that are maintained and refined jointly.

### MSA Representation — Evolutionary Information

```
m : [*, N_seq, N_res, c_m]
```

- `N_seq`: number of homologous sequences in the Multiple Sequence Alignment
- `N_res`: length of the target sequence (number of residues)
- `c_m`: embedding dimension (channel dimension learned by the model)

The MSA is the evolutionary record of the protein family. Each row is one homologous sequence, each column is one residue position. The model is not simply reading token identities — it learns a continuous embedding `c_m` that encodes the *pattern* of variation across sequences.

**Why does this matter?** If residues at positions *i* and *j* are physically in contact, a mutation at *i* that would destabilize the protein is often compensated by a correlated mutation at *j*. This **co-evolutionary signal** is encoded in the columns of the MSA: two columns that co-vary strongly across sequences are almost certainly spatially close. The model learns to extract this signal implicitly through attention across residues.

### Pair Representation — Geometric Information

```
z : [*, N_res, N_res, c_z]
```

- Indices `(i, j)` represent the relationship between residue `i` and residue `j`
- `c_z`: embedding dimension for pairwise features

The pair representation starts from **relative positional encodings** — rough information about how far apart residues are in sequence space. Over the course of the Evoformer, this 2D matrix is progressively refined into a representation that encodes hypotheses about 3D proximity, orientation, and contact probability.

Critically, `z[i,j]` is not simply symmetric: the model can represent *directed* geometric relationships (e.g., the orientation of residue *i* relative to *j* is not the same as *j* relative to *i*).

---

## The Evoformer (Trunk)

The Evoformer is a stack of identical blocks, each of which applies a sequence of attention and multiplicative update operations to `m` and `z`. The key property is that **`m` and `z` update each other within every block**: the pair representation biases the MSA attention, and the MSA updates the pair representation via **outer product mean.**

---

## MSA Attention Mechanisms

### Row-Wise Attention (with Pair Bias)

For each sequence `s` independently, we run standard multi-head self-attention across residue positions:

$$a_{s,ij} = \text{softmax}\!\left(\frac{q_{si} \cdot k_{sj}}{\sqrt{d}} + b_{ij}\right)$$

where `b_ij` is a learned linear projection of the pair representation `z_ij`.

**Why row-wise?** Each row of the MSA is one sequence. Attending along residue positions within a row captures which residues within that sequence tend to co-occur — but crucially, the pair bias `b_ij` couples this to the geometric hypotheses in `z`. This means the evolutionary attention is *geometrically grounded*: the model attends to residue `j` from residue `i` more if the current pair representation suggests they are close.

**This is how information flows from `z` into `m`.**

### Column-Wise Attention

For each residue position `i` independently, we attend across all sequences in the MSA:

$$a_{i,st} = \text{softmax}\!\left(\frac{q_{si} \cdot k_{ti}}{\sqrt{d}}\right)$$

**Why column-wise?** Each column of the MSA contains one amino acid from every homologous sequence at the same position. Attending along the sequence dimension at a fixed residue position captures **evolutionary conservation**: how variable is this position? Is it always the same amino acid (structurally critical) or highly variable (tolerant of substitution)? Column-wise attention allows the model to aggregate information across the entire evolutionary record for each individual residue.

---

## Triangular Multiplicative Updates

The pair representation `z` cannot be updated independently for each `(i,j)` pair without violating geometric consistency — if residue `i` is close to `k`, and `k` is close to `j`, then `i` must be close to `j`. This **triangle inequality** is enforced explicitly by the triangular multiplicative updates.

For each pair `(i,j)`, we aggregate information from all intermediate residues `k` that form a triangle `(i, k, j)`:

### Outgoing Update

$$z_{ij} \mathrel{+}= \text{LayerNorm}\!\left(\sum_k\; g(z_{ik}) \odot z_{ik} \cdot g(z_{jk}) \odot z_{jk}\right)$$

In einsum notation:

```
b i k c, b j k c -> b i j c
```

Both edges leave toward `k`: `i→k` and `j→k`. The triangle has its "tip" at `k`, with the updated edge `(i,j)` as its base. This propagates the constraint: *if `i` relates to `k` in a certain way, and `j` also relates to `k`, this constrains how `i` and `j` must relate*.

### Incoming Update

$$z_{ij} \mathrel{+}= \text{LayerNorm}\!\left(\sum_k\; g(z_{ki}) \odot z_{ki} \cdot g(z_{kj}) \odot z_{kj}\right)$$

In einsum notation:

```
b k i c, b k j c -> b i j c
```

Both edges arrive from `k`: `k→i` and `k→j`. The triangle has its "tip" at `k` but now the edges point inward. This propagates the complementary constraint: *if some third residue `k` relates to both `i` and `j`, this constrains the `(i,j)` relationship*.

Together, the outgoing and incoming updates ensure that all pairwise distances and orientations in `z` remain mutually consistent — a soft enforcement of the triangle inequality before we ever decode 3D coordinates.

---

## Triangular Attention

In addition to multiplicative updates, the pair representation uses **triangular self-attention** — attention that is also structured around triangles but uses query/key/value mechanics.

### Starting Node Attention

For each row `i` of the pair matrix, we attend along the column dimension `j`:

$$a_{ij} = \text{softmax}\!\left(\frac{q_i \cdot k_j}{\sqrt{d}} + b_{ij}\right)$$

where `b_ij` is a linear projection of `z_ij` itself (the pair representation biases its own attention). This aggregates information about residue `i`'s relationship to all other residues, weighted by the current pair features — **attending from the starting node `i` across all neighbors `j`**.

### Ending Node Attention

Symmetrically, for each column `j`, we attend along the row dimension `i` — **attending from the ending node `j`** across all residues `i` that point to `j`. This captures the dual perspective: how does `j` look from the point of view of every other residue.

The triangular attention and the multiplicative updates together ensure that the pair representation `z` converges to a geometrically consistent representation that respects the constraints of 3D space — without ever explicitly computing 3D coordinates during the trunk.

---

## Summary: Information Flow in One Evoformer Block

```
m ──[Row-wise attention, biased by z]──────────────────► m'
m ──[Column-wise attention]────────────────────────────► m'
m ──[Outer product mean]───────────────────────────────► Δz

z ──[Triangular multiplicative update (outgoing)]──────► z'
z ──[Triangular multiplicative update (incoming)]──────► z'
z ──[Triangular attention (starting node)]─────────────► z'
z ──[Triangular attention (ending node)]───────────────► z'
z'──[Pair transition MLP]──────────────────────────────► z''

z biases m attention; m updates z via outer product.
```

After `N` Evoformer blocks, `z` encodes a rich 2D geometric hypothesis about all pairwise residue relationships. The **Structure Module** then decodes this into 3D atomic frames using Invariant Point Attention (IPA), placing each residue's backbone rigid body in 3D space and finally predicting all atom coordinates.

---

## The Structure Module — From 2D to 3D

The Structure Module takes two inputs from the Evoformer:
- `s` — the **single representation** `[*, N_res, c_s]`, obtained by averaging the MSA representation across the sequence dimension. This is a per-residue summary of all evolutionary and geometric information.
- `z` — the **pair representation** `[*, N_res, N_res, c_z]` built throughout the trunk.

The module maintains a set of **rigid-body frames**, one per residue:

$$T_i = (R_i,\; \mathbf{t}_i) \in SE(3)$$

where $R_i \in SO(3)$ is a rotation matrix and $\mathbf{t}_i \in \mathbb{R}^3$ is a translation. At initialisation all frames are placed at the origin with identity rotation. The Structure Module iteratively updates these frames using **Invariant Point Attention (IPA)**.

### Invariant Point Attention (IPA)

Standard attention operates on embeddings in a flat vector space. IPA extends this to operate simultaneously in **embedding space and in 3D space**, while being invariant to the global orientation of the protein.

For each residue `i`, IPA computes queries, keys and values in two ways:

1. **Embedding queries/keys**: standard linear projections of `s_i` (as in a Transformer)
2. **Point queries/keys**: sets of 3D points in the local frame of each residue:

$$\hat{q}_i^{(h)} = T_i \cdot q_i^{(h)},\qquad \hat{k}_j^{(h)} = T_j \cdot k_j^{(h)}$$

The attention logit between residues `i` and `j` is:

$$a_{ij} = w_L \cdot \frac{q_i \cdot k_j}{\sqrt{d}} \;+\; b_{ij} \;-\; w_C \sum_h \|\hat{q}_i^{(h)} - \hat{k}_j^{(h)}\|^2$$

The third term is a **squared distance between 3D query and key points** expressed in global coordinates. Because both points are obtained by applying the current frames $T_i$ and $T_j$ to local vectors, this distance is invariant to any global rotation or translation applied to the entire structure: rotating everything by $R$ maps $\hat{q}_i \mapsto R\hat{q}_i$ and $\hat{k}_j \mapsto R\hat{k}_j$, so the distance $\|\hat{q}_i - \hat{k}_j\|$ is unchanged.

The pair bias `b_ij` (from `z`) is also included, coupling the 3D attention to the learned geometric hypotheses.

**Values** are likewise defined as 3D points in local frames, so the IPA output is a mix of aggregated embeddings and 3D vectors — also invariant to global pose. After IPA, the single representation `s` is updated and a **backbone update** module predicts a rotation and translation delta $\Delta T_i$ that refines each frame. After several IPA iterations, **torsion angles** for side chains are predicted from `s`, and all-atom positions are computed analytically from the backbone frames + torsion angles.

### SE(3)-Equivariance

**The full Structure Module is SE(3)-equivariant:** if the input frames are all rotated and translated by the same rigid transform $T$, the output atom coordinates transform by exactly $T$.

This is a critical property:
- The loss is computed on *relative* coordinates, so there is no preferred absolute orientation.
- The model can never "memorise" a canonical orientation — it must reason about geometry in a coordinate-free way.
- Equivariance means the prediction is the same regardless of how the protein is oriented in the input, which is the physically correct behaviour.

Note that the Evoformer itself is **invariant** (it works on sequences and pair indices with no 3D coordinates), and the Structure Module introduces equivariance only when it starts manipulating explicit 3D frames.

### Output

At the end of the Structure Module we obtain:
- `final_atom_positions` `[*, N_res, 37, 3]` — 3D coordinates for all 37 heavy atoms per residue (atom37 representation)
- `final_atom_mask` `[*, N_res, 37]` — validity mask (not all atom types exist for all residue types)
- `sm` — a dict of intermediate outputs from each Structure Module layer (frames, torsion angles, positions), used for auxiliary losses
- The single `s` and pair `z` representations, passed to the output heads

---

## Loss Function

### FAPE — Frame Aligned Point Error

FAPE is the primary structural loss. The idea is to compare predicted and true atom positions **in many different local reference frames**, making the loss invariant to global rotation and translation while still being sensitive to local errors.

For each backbone frame `i` (true frame $T_i$, predicted frame $\hat{T}_i$), and for each atom `j` with true position $\mathbf{x}_j$ and predicted position $\hat{\mathbf{x}}_j$:

$$\text{FAPE} = \frac{1}{N_{\text{frames}} \cdot N_{\text{atoms}}} \sum_i \sum_j \left\| T_i^{-1}\,\mathbf{x}_j \;-\; \hat{T}_i^{-1}\,\hat{\mathbf{x}}_j \right\|_{\epsilon}$$

where $\|\cdot\|_\epsilon = \sqrt{\|\cdot\|^2 + \epsilon^2}$ is a clamped L2 norm (Huber-like, robust to outliers). The term $T_i^{-1}\,\mathbf{x}_j$ places atom $j$ in the local coordinate system of frame $i$ — asking "where is atom $j$ as seen from residue $i$?".

**Why does this work well?**
- Comparing positions in each local frame makes the loss sensitive to *relative* errors between residues, not just global RMSD.
- A backbone error of 1 Å at residue 50 will contribute to the loss across all `N_res` local frames, not just frame 50 — amplifying the signal.
- FAPE is computed separately for backbone frames (all residues) and side-chain frames (all χ-angle frames), then combined.

In practice, FAPE is clamped at 10 Å to prevent very large errors in poorly-predicted regions from dominating the gradient early in training.

### Full Training Loss

The total loss is a weighted sum of several terms:

| Term | What it trains | Weight |
|---|---|---|
| **FAPE** (backbone) | Backbone frame accuracy | 0.5 |
| **FAPE** (side-chain) | Side-chain frame accuracy | 0.5 |
| **Supervised χ** | Torsion angle accuracy | 1.0 |
| **Distogram** | Pairwise Cβ-distance distribution | 0.3 |
| **pLDDT** | Confidence calibration | 0.01 |
| **Masked MSA** | MSA reconstruction (BERT-style) | 2.0 |
| **Violation** | Steric clash / bond geometry | 0.0 (annealed in) |

The distogram and masked MSA losses act as **auxiliary objectives** that regularise the pair and MSA representations during training — they prevent the Evoformer from collapsing to solutions that only minimise FAPE.

---

## Output Heads

### pLDDT — Per-Residue Confidence

The pLDDT head predicts how confident the model is in its own prediction for each residue. It is a **classification head** over 50 bins in `[0, 1]`:

```
s [*, N_res, c_s]  ──►  Linear  ──►  [*, N_res, 50]  ──►  softmax
```

The expected value of the bin is the reported pLDDT score. During training it is supervised against the true lDDT (local Distance Difference Test) computed from the predicted vs true all-atom structure. **A high pLDDT (> 90) means the model is confident; low pLDDT (< 50) indicates a disordered or uncertain region.**

The pLDDT is also used at inference to colour structure visualisations in AlphaFold DB.

### PAE — Predicted Aligned Error

PAE predicts the expected error in the position of residue `j` when the structure is aligned on residue `i`. It is a pairwise confidence metric:

```
z [*, N_res, N_res, c_z]  ──►  Linear  ──►  [*, N_res, N_res, 64]  ──►  softmax over bins
```

Output: `[*, N_res, N_res]` matrix where entry `(i,j)` is the expected positional error of residue `j` in Å, expressed in the local frame of residue `i`. **Low PAE between two domains means the model is confident about their relative orientation — high PAE means it is not.** PAE is the primary tool for identifying domain boundaries and for multimer inter-chain confidence.

### PDE — Predicted Distance Error

PDE is a lighter variant that predicts the expected error in the *distance* between two residues (rather than the full aligned error). It is used internally in some distogram-related diagnostics and in multimer chain ranking.

---

## Adding a Custom Head — Affinity Example

**Any differentiable function of `s` or `z` can be added as a new output head** without touching the Evoformer or Structure Module. In Molfun this is done by registering a head class:

```python
from molfun.losses import LossFunction, LOSS_REGISTRY

# 1. Define your loss (optional if using a built-in)
@LOSS_REGISTRY.register("mse")
class MSELoss(LossFunction):
    def forward(self, preds, targets=None, batch=None):
        return {"affinity_loss": F.mse_loss(preds, targets.view_as(preds))}


# 2. Define your head — operates on the single representation
class AffinityHead(nn.Module):
    def __init__(self, single_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(single_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trunk_output: TrunkOutput, **_) -> torch.Tensor:
        # Pool over residue dimension: [B, N, c_s] → [B, c_s]
        pooled = trunk_output.single_repr.mean(dim=-2)
        return self.mlp(pooled)   # [B, 1]


# 3. Attach and fine-tune
model = MolfunStructureModel(
    "openfold", config=cfg, weights="weights.pt",
    head="affinity",
    head_config={"single_dim": 384, "hidden_dim": 128},
)
strategy = LoRAFinetune(rank=8, lr_lora=1e-4, lr_head=1e-3, loss_fn="mse")
history  = model.fit(train_loader, val_loader, strategy=strategy, epochs=30)
```

The Evoformer and Structure Module provide a rich representation of the protein's geometry; the affinity head only needs to learn how to *read* that representation for the target task. This is why even a simple MLP on the pooled single representation can be effective when the trunk is well pre-trained.

---

## Recycling Cycles — Iterative Refinement

**A single forward pass through the full model is often not enough.** The Evoformer is processing tokens, not 3D coordinates, so it can miss structural constraints that only become apparent once a rough 3D prediction exists. Recycling solves this by running the full model multiple times, feeding the output of one pass as extra input to the next.

### What Gets Recycled

At the end of cycle `r`, three things are packaged and passed to cycle `r+1`:

1. **Single representation** `s^{(r)}` — the per-residue embedding from the previous Evoformer output, re-injected into the input embedder via a LayerNorm.
2. **Pair representation** `z^{(r)}` — the pairwise embedding, re-injected similarly.
3. **Backbone Cβ positions** — the predicted 3D coordinates are binned into **15 distance bins** (0–20 Å) and added as extra pair features. This gives the Evoformer direct geometric feedback: "the previous pass predicted residues `i` and `j` are ~8 Å apart — use that to refine the pairwise representation."

```
Cycle r:   [Input features]  +  [recycled s, z, distogram]
                │
           [Input Embedder]
                │
           [Evoformer × 48 blocks]  →  s^(r), z^(r)
                │
           [Structure Module]  →  frames^(r), positions^(r)
                │
                └──────► recycle into cycle r+1
```

### Why Recycling Works

The first pass through the Evoformer builds a rough hypothesis from sequence alone. By the time the Structure Module produces rough 3D positions, the pair representation `z` does not yet know what the actual distances are. In the second cycle, the binned distances from cycle 1 are added as features, allowing the Evoformer to **condition its attention on a concrete geometric scaffold** rather than starting from scratch.

**Each recycling cycle refines the prediction:** backbone RMSD typically drops by 1–2 Å per cycle for difficult targets, converging after 3–4 cycles. Training uses a random number of recycling cycles (uniformly sampled from 1 to `N_recycle`) to ensure the model is robust to any number of cycles at inference.

In Molfun, the recycling dimension `R` is the trailing dimension of every input tensor:

```
aatype:             [B, N_res, R]
msa_feat:           [B, N_seq, N_res, 49, R]
backbone_rigid:     [B, N_res, 4, 4, R]
```

The `_strip_recycling_dim` helper in `molfun.helpers.openfold` removes it before passing tensors to loss functions, which expect outputs without that dimension.

---

## Complete Picture

```
                     ┌──────────── Recycling ─────────────────────────┐
                     │                                                  │
 Sequence (1D) ──► Input Embedder ──► m [N_seq,N_res,c_m]             │
                                      z [N_res,N_res,c_z]             │
                            │                                          │
                     ┌──────▼──────────────┐                          │
                     │   Evoformer × 48    │  ◄── recycled s, z,      │
                     │  (MSA + Pair attn)  │      Cβ distance bins    │
                     └──────┬──────────────┘                          │
                            │  s, z                                   │
                     ┌──────▼──────────────┐                          │
                     │  Structure Module   │                          │
                     │  IPA × 8 layers     │                          │
                     └──────┬──────────────┘                          │
                            │                                          │
              ┌─────────────┼──────────────┐              ────────────┘
              │             │              │
        Atom positions    pLDDT head    PAE head    AffinityHead
        [N_res,37,3]     [N_res,50]  [N_res,N_res]    [B,1]
              │
        FAPE loss (training)
```

The result is a system where **sequence evolution informs geometry, geometry informs attention, attention refines geometry, and iterative recycling converges to a self-consistent 3D structure** — all without any hand-crafted physical simulation.

