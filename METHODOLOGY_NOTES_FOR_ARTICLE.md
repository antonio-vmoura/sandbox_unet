# Methodology Notes for the Article — U-Net on ISIC 2018 Task 1 (aligned with YOLO26-seg)

> **Purpose.** Technical foundation for the *Materials and Methods* section of the thesis and article, for the
> U-Net arm of the study. It describes the protocol exactly as implemented in this repository
> (`run_pipeline_unet.sh`, `unet/`) and how it was aligned with the YOLO26-seg pipeline (`sandbox_yolo26`,
> documented in its own `METHODOLOGY_NOTES_FOR_ARTICLE.md`). Items in **[brackets]** must be completed with
> values from the final run. Section 9 lists points to verify or disclose before submission.

---

## 1. Overview and design principles

The U-Net serves as the classical encoder–decoder reference against which the YOLO26-seg family (and SAM) is
compared. The U-Net experiments follow the **same five-phase protocol** as YOLO26 — baseline training,
baseline cross-validation, hyperparameter optimisation (HPO), optimised fine-tuning, and a single evaluation on
a held-out test set with hardware profiling — and were engineered so that every quantity reported for the two
architectures is measured on the same data, with the same definitions, at the same resolution and with the same
software stack. The alignment is summarised below and detailed in Sections 2–7.

| Aspect | Alignment with YOLO26 |
|---|---|
| Images, splits, annotations | Identical: the U-Net data are derived from the YOLO26 dataset (Phase 0) |
| Cross-validation folds | Identical partition (same pool order and splitting algorithm), verified |
| Accuracy metrics | Byte-identical implementation; same ground truth; same evaluation resolution |
| Model selection / budget / precision | Same training budget (120 epochs, patience 25), FP32, seed 0 |
| HPO budget and bookkeeping | 30 trials × 30 epochs (patience 10); identical checkpoint schema |
| Efficiency profiling | Same procedure (derived from the same code), same deployment transformation (Conv–BN fusion), same PyTorch build |
| Output schema | Identical summary tables (columns verified), consumed by the same notebooks |

---

## 2. Pipeline architecture and alignment with YOLO26

### 2.1 Phase 0 — a single source of truth for the data

The data originally used for the U-Net (pre-computed NumPy arrays) had three defects that made a rigorous
comparison impossible: (i) the ground-truth masks were **not binary** — all lesion pixels took fractional values
(maximum ≈ 0.88, thousands of distinct values), so that the reported IoU/Dice compared thresholded predictions
against soft targets and the BCE loss was trained on soft labels; (ii) the test-set arrays were missing; and
(iii) the arrays carried no image identifiers, so the splits could not be aligned with those of YOLO26, which
differ from the official ISIC split. All earlier U-Net results were therefore discarded.

In the present protocol, a deterministic preprocessing step (Phase 0) builds the U-Net data **from the YOLO26
dataset itself** — the same image files, the same polygon annotations and the same train/validation/test split
(2,547 / 100 / 994 images; [state the provenance of the YOLO export, which differs from the official ISIC split
of 2,594 / 100 / 1,000 images]). For each image:

1. The ground-truth mask is obtained by rasterising the YOLO polygons at the image's resolution with the same
   function YOLO26's evaluation uses (Section 2.3).
2. The image is resized to 256 × 256 pixels with area interpolation (the input size of the original U-Net
   baseline).
3. The mask is area-resized to 256 × 256 (each output pixel takes the fraction of lesion pixels it covers) and
   thresholded at 0.5, which keeps it **strictly binary** while placing the boundary at the sub-pixel majority.

Every sample is addressed by its ISIC identifier; Phase 0 verifies that no identifier occurs in two splits and
that all masks take only the values {0, 1}, and records SHA-256 digests of every source file and output array.
The cache is rebuilt only when the source data or the preprocessing parameters change.

### 2.2 The five phases

| Phase | Purpose | Data | Notes |
|---|---|---|---|
| 0. Data cache | Build 256 × 256 arrays from the YOLO26 dataset | all splits | Section 2.1 |
| 1. Baseline | Base setup + default hyperparameters | train (fit), val (selection) | Section 3 |
| 2. Baseline CV | 5-fold CV with the Phase 1 configuration | train ∪ val pool | identical folds to YOLO26 (Section 2.4) |
| 3. HPO | Seeded, fault-tolerant Optuna/TPE search | train (fit), val (fitness) | Section 4 |
| 4. Optimised | Base setup + tuned hyperparameters | train (fit), val (selection) | Section 3 |
| 5. Test set | Accuracy and efficiency of Baseline and Optimised | **test only** | Sections 5–6 |

The test split is used exclusively in Phase 5; before Phase 2, the implementation additionally verifies that no
test identifier is present in the cross-validation pool.

### 2.3 Identical accuracy metrics, ground truth and resolution

The pixel-level metrics (DSC, JSI, ISIC 2018 thresholded JSI, sensitivity, specificity, pixel accuracy; empty
predictions scored 0 and never discarded) are computed by a module that is a **byte-for-byte copy** of
YOLO26's. The U-Net predicts at 256 × 256; its probability map is **upsampled bilinearly to the image's
original resolution (640 × 640)** and thresholded at 0.5, and is scored against the ground truth rasterised from
the same YOLO polygons. U-Net and YOLO26 predictions are thus evaluated on the same pixels, against the same
masks, with the same code. The same evaluation is applied to the held-out folds of Phase 2.

### 2.4 Identical cross-validation folds

The cross-validation pool is formed by the training identifiers followed by the validation identifiers, in the
order in which YOLO26 builds its pool, and is partitioned with YOLO26's algorithm (indices shuffled with
NumPy's `RandomState(0)`, K = 5 contiguous blocks, the first N mod K blocks receiving one extra image; equivalent
to `KFold(n_splits=5, shuffle=True, random_state=0)` without scikit-learn). We verified with YOLO26's own
cross-validation code that **both pipelines produce identical pools and identical folds**, so per-fold results
of the two architectures are paired. Fold-wise metrics are aggregated as mean ± **sample** standard deviation
(ddof = 1). A fingerprinted split manifest prevents the combination of folds from different partitions.

### 2.5 Identical deployment form for profiling (Conv–BN fusion)

At inference, Ultralytics folds every BatchNorm layer of YOLO26 into the preceding convolution
(`model.fuse()`), and YOLO26's parameter counts, GFLOPs and latencies are measured on this fused graph. The
U-Net is profiled in the same deployment form: every BatchNorm is folded into its convolution
(`torch.nn.utils.fusion.fuse_conv_bn_eval`), which leaves the outputs unchanged up to floating-point rounding
(maximum absolute difference of the logits 1.9 × 10⁻⁶ in our verification) and reduces the parameter count from
2,161,649 to 2,158,705.

---

## 3. The PyTorch port of the U-Net

### 3.1 Motivation

The original U-Net was implemented in TensorFlow/Keras. Measuring latency, memory and computational cost of a
TensorFlow model and of PyTorch models (YOLO26, SAM) would confound architectural differences with framework
differences (runtime, kernel selection, allocator behaviour, FLOP counting). The U-Net was therefore ported to
PyTorch and runs on the same pinned stack as YOLO26 (Docker, CUDA 12.1, PyTorch 2.5.1, torchvision 0.20.1).

### 3.2 Architecture and verified equivalence

The port reproduces the Keras model layer by layer: four encoder levels of two 3 × 3 convolutions with
BatchNorm and ReLU (16, 32, 64 and 128 filters), each followed by 2 × 2 max-pooling and dropout; a bottleneck of
256 filters; four decoder levels with a 3 × 3 transposed convolution (stride 2), concatenation with the skip
connection, dropout and a convolution block; and a 1 × 1 convolution with sigmoid output. Framework defaults that
differ were set explicitly to the Keras values:

* BatchNorm momentum 0.99 (Keras convention; 0.01 in PyTorch's convention) and ε = 10⁻³;
* `he_normal` initialisation (truncated normal) for the block convolutions and `glorot_uniform` for the
  transposed convolutions and the output layer, all biases zero;
* the spatial alignment of Keras' `Conv2DTranspose(padding="same")`, which corresponds to the un-padded PyTorch
  transposed convolution cropped to its *first* 2n rows and columns. (The common PyTorch idiom
  `padding=1, output_padding=1` keeps the *last* 2n and is shifted by one pixel; with identical weights it
  produced completely different outputs in our test.)

**Equivalence was verified numerically**: after copying the weights of a randomly initialised Keras model built
with the original, unmodified Keras code (with randomised BatchNorm statistics) into the PyTorch model, the two
outputs were **identical (maximum absolute difference 0.0)**, and the numbers of trainable parameters matched
exactly (2,161,649). The statistics of the PyTorch initialisers were checked against the Keras definitions.

### 3.3 Training procedure

* **Loss:** binary cross-entropy (pixel mean) plus (1 − soft Dice), the Dice coefficient computed over the whole
  batch with smoothing 10⁻⁶ — the original Keras `bce_dice_loss`. The BCE is computed from logits (the
  numerically stable form; Keras additionally clips probabilities to [10⁻⁷, 1 − 10⁻⁷]).
* **Optimiser:** AdamW (decoupled weight decay, equivalent to Keras `Adam(weight_decay=…)`), ε = 10⁻⁷ (Keras
  default), β₂ = 0.999, constant learning rate (as in the original baseline); batch size 16; FP32.
* **Budget and model selection:** at most 120 epochs; after each epoch, the per-image mean JSI on the validation
  set (at 256 × 256) is computed with the definitions of Section 2.3; the checkpoint is replaced only by a
  strictly better epoch (ties keep the earliest epoch); training stops after 25 epochs without improvement.
  Reported validation metrics are always those of the selected epoch.

### 3.4 Strictly binary augmentation (correction of the soft-mask flaw)

The original pipeline augmented images and masks with Keras' `ImageDataGenerator` (rotation ±15°, horizontal
and vertical shifts of ±10 %, zoom of ±10 %, horizontal and vertical flips, reflective padding), interpolating
**both** image and mask bilinearly — which turned the mask borders into soft labels, compounding the
soft-mask defect of the data (Section 2.1). The augmentation was re-implemented jointly on image and mask with the
same transformation family and parameter semantics (rotation angle, independent horizontal/vertical shifts and
zoom factors drawn uniformly, affine transform about the image centre, reflective borders, flips with given
probabilities), with one deliberate difference: **masks are warped with nearest-neighbour interpolation and
therefore remain strictly binary**, while images are interpolated bilinearly. Visual inspection confirmed the
alignment of augmented images and masks.

### 3.5 Determinism and fault tolerance

The random parameters of the augmentation of sample *i* in epoch *e*, and the sample order of epoch *e*, are drawn
from generators seeded with (seed, *e*, *i*) and (seed, *e*); data order and augmentation are therefore pure
functions of the seed and the epoch, independent of the number of data-loader workers. Each data loader owns a
dedicated random generator, so the global random stream used by dropout is likewise independent of the number of
workers (without this, PyTorch draws worker seeds from the global stream, which made results depend on the worker
count). A checkpoint written atomically after every epoch stores the model, optimiser, epoch, early-stopping
state and all random-number-generator states. Consequently, **a run interrupted at any point and resumed is
bit-identical to an uninterrupted run** (verified on CPU and on GPU, and with 0 versus 8 workers); the per-epoch
log is rebuilt from the checkpoint on resumption. Two identical GPU runs were also bit-identical, and PyTorch
reported no operation without a deterministic implementation. (The YOLO26 pipeline can only guarantee a
resumed, not a bit-identical, run, because Ultralytics does not checkpoint the data-loader state.)

---

## 4. Fault-tolerant hyperparameter optimisation (Optuna / TPE)

### 4.1 Search space: learning dynamics and augmentation only

To isolate the effect of hyperparameter optimisation, the **base setup is identical in all phases** —
architecture (width, activation, normalisation), optimiser type, loss, input size, batch size, budget and
numerical precision — and cannot be modified by a tuned configuration or a search space (the implementation
rejects such keys). The search covers only the learning dynamics and the augmentation:

| Hyperparameter | Baseline (Keras default) | Search range |
|---|---|---|
| Initial learning rate `lr0` | 10⁻³ | [10⁻⁴, 10⁻²], log |
| Weight decay | 0 | [10⁻⁶, 10⁻³], log |
| β₁ (AdamW) | 0.9 | [0.80, 0.95] |
| Dropout | 0.1 | [0.10, 0.40] |
| Rotation (degrees) | 15 | [0, 45] |
| Translation (fraction) | 0.1 | [0, 0.20] |
| Zoom (fraction) | 0.1 | [0, 0.30] |
| Horizontal / vertical flip probability | 0.5 / 0.5 | [0, 0.50] |

Baseline = base setup + Baseline column; Optimised = base setup + HPO result. Both variants have identical
architecture and therefore identical computational cost.

**Baseline values outside the search bounds.** The default weight decay of the original baseline is 0, which
cannot be represented on the logarithmic scale on which weight decay is searched (whose lower bound is 10⁻⁶).
The first trial of the search evaluates the Baseline configuration clipped to the bounds, i.e. with weight decay
10⁻⁶ — a negligible amount of regularisation, but formally a different configuration. The Baseline is therefore
not itself a candidate of the search, and the search is not guaranteed to return a configuration at least as good
as the Baseline; the Phase 5 comparison measures the effect of the selected configuration. The logarithmic scale
was retained because the plausible values of weight decay span several orders of magnitude, and because some
regularisation is expected to be selected by the search (the same situation arises for the learning rate and
weight decay of YOLO26; see its notes).

### 4.2 Algorithm and reproducibility

We retained the HPO framework of the original U-Net pipeline: Optuna's Tree-structured Parzen Estimator (TPE),
with SQLite storage, 30 trials of at most 30 epochs (early-stopping patience 10), the first 10 trials being random
(Optuna's default start-up phase), and the fitness of a trial being the validation per-image mean JSI of its best
epoch. Because the sampler's random state is not persisted, a search resumed from storage with a re-created
`TPESampler(seed)` would propose different configurations from an uninterrupted search. We therefore install, before
every proposal *i*, a **fresh TPE sampler seeded with (seed × 1,000,003 + i) mod 2³²**, where *i* is the number of
completed trials (the first proposal is fixed to the clipped Baseline configuration). Since TPE builds its model
only from completed trials and draws all its randomness from its seed, the configuration proposed for trial *i* is
a deterministic function of (seed, *i*, history of completed trials). The proposed parameters are recorded, and a
resumed or retried proposal is verified to be identical; otherwise the search stops. Two independent executions of
the search produced identical trial histories in our verification.

### 4.3 Checkpointing and crash recovery (`hpo_state.json`)

The search state is checkpointed atomically before every trial in `hpo_state.json`, with the same schema as
YOLO26's (status, target / completed / valid trials, best fitness and trial, in-flight trial, failed attempts,
accepted failures, configuration hash, library versions and an event history). Completion is defined by the number
of completed trials. On restart:

* trials left "running" by a crash are marked as interrupted (not counted as failures) and **the same proposal is
  asked again — it receives identical parameters and the same trial folder, so the interrupted training itself
  resumes bit-exactly from its last checkpoint** (in YOLO26 an interrupted trial restarts from its first epoch);
* a trial that raises an error or returns a non-finite fitness is retried with identical parameters, from a clean
  folder, up to two times, and is then recorded as a completed trial with fitness 0 (a legitimately low but finite
  fitness is a valid result, not a failure);
* failures that coincide with an unavailable GPU are not counted; the search exits with code 75, and the
  orchestrator retries after a waiting period, resuming from the last completed trial;
* resumption is refused if the search space, base setup, trial budget, seed, data, or the Optuna/PyTorch versions
  differ from those of the original search.

These behaviours were verified by fault injection (kill during a trial, transient and persistent trial failures,
GPU failure, configuration change, extension of the number of trials) and by killing a real search during the
training of a trial, after which the completed search was identical to an uninterrupted one.

**Asymmetry to disclose.** The U-Net search uses TPE, YOLO26's the Ultralytics genetic algorithm; the budgets
(30 trials × 30 epochs, patience 10) and the bookkeeping are identical, but the search algorithms differ.

---

## 5. Test-set evaluation (Phase 5a)

Every combination of variant (Baseline, Optimised) × numerical precision (FP32, FP16) is evaluated on the 994
test images with batch size 1, as described in Section 2.3 (FP32 is the primary result; FP16 quantifies the
accuracy cost of half precision). Aggregates are the per-image mean (primary), sample standard deviation, median,
interquartile range, a seeded percentile-bootstrap 95 % confidence interval of the mean (2,000 resamples), and
pooled DSC/JSI. YOLO26's Ultralytics-only instance metrics (box and mask mAP, precision, recall, F1) are not
defined for a semantic-segmentation network and are reported as missing. The HPO gain is a paired per-image
comparison (mean difference with bootstrap 95 % CI; two-sided Wilcoxon signed-rank test), as for YOLO26.

---

## 6. Hardware efficiency profiling (Phase 5b)

The efficiency benchmark of the U-Net is derived from the YOLO26 benchmark code; only the model-specific parts
differ. Each configuration (variant × precision) runs in a fresh process on a single GPU with batch size 1:

* **Forward latency** of the fused network (Section 2.5) on a fixed 1 × 3 × 256 × 256 input: 50 warm-up
  iterations, then 500 iterations each timed with a pair of CUDA events and a device synchronisation.
* **End-to-end latency** of the deployed pipeline on a 640 × 640 test image (colour conversion, area resize to
  256 × 256, scaling, host-to-device copy, forward pass, sigmoid, bilinear upsampling to 640 × 640, threshold,
  device-to-host copy): 20 warm-up and 200 timed calls with a monotonic host clock between synchronisations.
* Mean, standard deviation, median, 90th/95th/99th percentiles; FPS = 1000 / mean latency.
* **VRAM:** steady-state peak allocated/reserved during the timed iterations, with the peak counters reset after
  warm-up so that cuDNN autotuning workspaces are excluded (the warm-up peak is reported separately); VRAM of the
  weights alone; the CUDA context is excluded. **RAM:** process resident set size and its peak.
* **Size and cost:** parameters (unfused and fused); GFLOPs as 2 × multiply-accumulate operations (thop), the
  convention used for YOLO26; on-disk size of the checkpoint (stored in FP32, whereas YOLO26 checkpoints are stored
  in FP16 — compare the theoretical FP32/FP16 weight sizes, fused parameters × 4 / × 2 bytes).
* **FP32 vs FP16:** weights and inputs cast to half precision; identical conditions; FP16 accuracy measured, not
  assumed.
* Runs during which another process used the GPU (> 5 % utilisation) are flagged as contended and must be
  repeated on an idle GPU. **[State the GPU, driver and that all reported latencies are uncontended.]**

(In a preliminary, contended measurement, the U-Net — like YOLO26n — was not faster in FP16 than in FP32 at
batch size 1, consistent with kernel-launch overhead dominating small models at batch 1. **[Confirm on an idle
GPU.]**)

---

## 7. The resolution ceiling and the computational-cost comparison

### 7.1 Resolution ceiling of the 256 × 256 U-Net

The U-Net is trained and run at 256 × 256 pixels (the input size of the original baseline), whereas evaluation
takes place at the 640 × 640 resolution of the dataset (identical to YOLO26). Down-sampling to 256 × 256 removes
boundary detail that no prediction at that resolution can recover. To quantify this upper bound, we evaluated an
**oracle** that outputs the reference 256 × 256 mask of each test image through exactly the U-Net inference
pipeline (bilinear upsampling to 640 × 640 and thresholding at 0.5): on the 994 test images it reached a mean
**DSC of 0.9967** (minimum 0.9650; mean JSI 0.9935; pooled DSC 0.9977). This **resolution ceiling** bounds the
attainable accuracy of the 256 × 256 U-Net when scored at 640 × 640; its cost (≈ 0.003 DSC on average, up to
≈ 0.035 for individual images with small or intricate lesions) is small compared with typical inter-model
differences, but it should be stated when the U-Net is compared with models that predict at 640 × 640.
(Nearest-neighbour upsampling of the 256 × 256 mask gives a slightly lower ceiling — DSC 0.9951 on a sample of 199
test images — which is why the pipeline uses bilinear upsampling of the probability map.) The same oracle verification also confirms that the inference and scoring path
is correctly aligned (a one-pixel shift or a flip would have lowered the oracle's DSC drastically).

### 7.2 Computational cost: native vs. resolution-matched GFLOPs

Computational cost depends on the input resolution. At its native 256 × 256 input, the fused U-Net (2.16 M
parameters) requires **6.40 GFLOPs** per image. Evaluated at 640 × 640 — the input resolution of YOLO26 — the same
network requires **40.0 GFLOPs** (the cost of a fully convolutional network grows with the number of pixels,
(640/256)² = 6.25 ×), more than four times the 9.0 GFLOPs of YOLO26n-seg at 640 × 640 (Ultralytics' fused-model
summary), even though YOLO26n-seg has a comparable number of parameters (2.69 M fused). The difference reflects
the architectures: the U-Net applies its convolutions at full input resolution in its first and last stages,
whereas YOLO26 reduces the resolution early in its backbone. We therefore report the U-Net's cost both at its
native resolution (the configuration actually deployed and timed) and at the resolution-matched 640 × 640; a
comparison of computational efficiency between the architectures should use the resolution-matched figure, while
latency is reported for each model at its native input size. **[Report the corresponding latencies.]**

---

## 8. Statistical analysis

Identical to YOLO26: cross-validation mean ± sample standard deviation (ddof = 1); test-set per-image mean with
seeded percentile-bootstrap 95 % CI (2,000 resamples, seed 0); HPO gain as paired per-image difference with
bootstrap CI and two-sided Wilcoxon signed-rank test. **[If claims are made jointly across models/architectures,
correct for multiple comparisons, e.g. Holm–Bonferroni.]** Because the U-Net and YOLO26 are evaluated on the same
test images and the same folds, cross-architecture comparisons can be paired as well.

---

## 9. Points to verify or disclose

1. **Different input resolutions.** U-Net 256 × 256 vs YOLO26 640 × 640 (Section 7); evaluation for both at
   640 × 640.
2. **Evaluation resolution of the dataset export.** The YOLO export stores 640 × 640 images (the original
   dermoscopic images were resized without preserving the aspect ratio); DSC and JSI are ratios of areas and are
   invariant to such a uniform affine rescaling up to rasterisation effects. [State the export settings.]
3. **Split differs from the official ISIC split** (2,547 / 100 / 994 instead of 2,594 / 100 / 1,000) because the
   shared YOLO export is used for all architectures.
4. **Search algorithm asymmetry:** TPE (U-Net) vs genetic algorithm (YOLO26); same budget.
5. **Baseline values outside the search bounds** (weight decay 0 → first trial 10⁻⁶; Section 4.1).
6. **Optimistic bias of validation-based figures** (the validation data select the checkpoint); main claims should
   rest on the Phase 5 test-set figures.
7. **Deliberate differences from the original Keras training:** binary (nearest-neighbour) mask augmentation; a
   fresh shuffle every epoch (Keras' iterator continued across epochs); BCE from logits without Keras' probability
   clipping.
8. **Validation metric resolution:** model selection and the validation/CV metrics from training logs use
   256 × 256 masks; the CV pixel metrics and all test metrics use 640 × 640.
9. **Single network width** (16 base filters, the original baseline); no width scaling as for the YOLO26 sizes.
10. **Contended efficiency runs** must be repeated on an idle GPU; report GPU model and driver.
11. **HPO bookkeeping:** report completed, failed and accepted-failure trials (`hpo_state.json`).

---

## 10. Suggested references

*(Verify bibliographic details before use.)*

* O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015.
* T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization
  Framework," KDD 2019.
* J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl, "Algorithms for Hyper-Parameter Optimization," NeurIPS 2011 (TPE).
* S. Ioffe, C. Szegedy, "Batch Normalization," ICML 2015.
* K. He, X. Zhang, S. Ren, J. Sun, "Delving Deep into Rectifiers," ICCV 2015 (He initialisation).
* X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS 2010.
* I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019 (AdamW).
* N. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018," arXiv:1902.03368, 2019.
* P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset," *Scientific Data* 5, 180161, 2018.
* L. R. Dice (1945); P. Jaccard (1912); F. Wilcoxon (1945); B. Efron & R. J. Tibshirani (1993); S. Holm (1979).
* A. Paszke et al., "PyTorch," NeurIPS 2019.
