# Literature Review, Phase 2: Deep Learning

Team T084 · Phase 2 of "Indoor Location Prediction Using Wi-Fi Signals"

The background of our Phase 2 network, in six short parts. Phase 1's review
([literature_review.md](literature_review.md)) covers fingerprinting, the dataset, the classical
models and fair evaluation. The results we quote are listed, with their place in the paper, in
[literature_matrix_phase2.csv](literature_matrix_phase2.csv); the keys match `report/references.bib`.
Every paper was checked against its publisher record or arXiv page; papers that are not
peer-reviewed are marked as preprints.

## 1. Multi-task learning

- **Caruana (1997)** defines multi-task learning (MTL): related tasks are learned in parallel with
  a shared representation, and what one task learns acts as an inductive bias that helps the
  others. His networks share the hidden layer between tasks.
- **Ruder (2017, arXiv preprint)** calls this *hard parameter sharing*: "sharing the hidden layers
  between all tasks, while keeping several task-specific output layers" (Sect. 3.1). It "greatly
  reduces the risk of overfitting" because the shared layers must work for every task, but it
  "quickly breaks down if tasks are not closely related" (Sect. 6.9).
- **Kendall et al. (2018)** show that a multi-task network's results are "strongly dependent on the
  relative weighting between each task's loss" (abstract), and that the best weights depend on the
  units of each task (e.g. meters). The usual approach is a weighted sum of the task losses
  (their Eq. 1), which is what we use, with the position weight tuned by CV.

Building, floor and position all describe one place, so they are related tasks; this is why the
brief asks for one shared network with three heads.

## 2. Deep networks for Wi-Fi fingerprinting

Phase 1 already reviewed the autoencoder classifier of Nowicki and Wietrzykowski (2017), the
scalable network of Kim et al. (2018) and CNNLoc (Song et al., 2019). The newer work closest to us:

- **Kim (2018), the SIMO network.** One network with a shared hidden layer, a floor head (softmax,
  cross-entropy) and a coordinate head (linear output, squared error): the design the brief asks
  for. Tested on another dataset (TUT, not UJIIndoorLoc); the coordinate loss weight was swept
  and 0.8 worked best; the table reports the best of 20 runs.
- **Laska and Blankenbach (2022)** keep the published UJIIndoorLoc split and average 10 runs. A
  SIMO-style MLP (shared layers 512-512, heads of 256) reached a mean 2D error of 9.20 m, while a
  *single-task* position network reached 8.25 m (Tables 2 and 8): on this dataset, sharing did
  not help the position task by itself. Their own multi-task grid model (m-CEL) reached 7.32 m.
- **Li et al. (2025)** compare one hybrid network trained jointly with linked networks trained in
  stages (a building/floor network first, then a position network started from its layers): same
  floor accuracy (93.34%), mean 3D error 8.45 m (joint) against 8.19 m (staged) (arXiv v1, Table V;
  one comparison, no repeats).
- **Elesawi and Kim (2021)** use recurrent networks for building and floor and a separate fully
  connected network for position: 100%, 95.23% and a 3D error of 8.62 m on the validation file. The
  best setting appears to have been read off the same numbers that are reported as test results.
- **Li and Kim (2026)** is the closest published match to our design: a shared trunk with
  building, floor and coordinate heads, a weighted loss (0.4, 0.8, 12.0) because "the localization
  task requires stronger gradient emphasis to avoid domination by the classification branches",
  AdamW, and not-heard WAPs set to 0. Their fusion settings were tuned on the validation file that
  is also their test set, so their 9.01 (dataset units) is optimistic.

**Why the numbers are hard to compare.** A review of deep fingerprinting (Alhomayani and Mahoor,
2020) names the lack of a common evaluation framework as a major problem, and notes that the deep
network of Kim et al. (99.82% building, 91.27% floor, 9.29 m) "did not prevail over" RTLS@UM, the
best system of the EvAAL 2015 competition (6.20 m), without saying whether both used the same test
data. The same model can look very different under different protocols:
Kargar-Barzi et al. (2024) report 9.52 m on the published split and 2.36 m after merging both
files and splitting them again at random. Published numbers also mix 2D and 3D errors and best-of-
runs results. We therefore compare our network only with our own Phase 1 models, on the same
folds, the same test scans and the same metrics.

## 3. Inputs, missing values and augmentation

- **Scaling.** Kargar-Barzi et al. (2024) min-max scale each reading, (r - r_min) / -r_min with
  r_min = -104 dBm, and map "not heard" to 0, like our scaled RSSI (we use -105 dBm, 1 dB below the
  weakest training reading). Li and Kim (2026) also map missing WAPs to 0.
- **Missingness indicators.** A recent preprint (Eftekharizadeh et al., 2026) gives its network two
  channels per WAP, the normalized RSSI and a 0/1 detection mask, which are our scaled RSSI and
  heard indicators.
- **Hiding WAPs.** The same preprint drops each WAP with probability 0.15 during training. Dropout
  (Srivastava et al., 2014) drops "units (hidden and visible)" and found "dropping out 20% of the
  input units and 50% of the hidden units" often optimal. Our *hidden WAPs* hide 10% of the heard
  WAPs in every training batch, marking them as not heard.

## 4. Training and regularization

- **Layers.** A multilayer perceptron (MLP) is the basic deep feedforward network (Goodfellow et
  al., 2016, Ch. 6); ReLU, max(0, x), is the default hidden activation (Nair and Hinton, 2010;
  Goodfellow et al., 2016, Sect. 6.1).
- **Optimizer.** Adam (Kingma and Ba, 2015) adapts the step size of every weight from running
  averages of its gradients. AdamW (Loshchilov and Hutter, 2019) applies weight decay separately
  from the gradient step: for plain gradient descent L2 regularization and weight decay are the
  same, but for Adam they are not (abstract).
- **Regularization** is "any modification ... intended to reduce its generalization error but not
  its training error" (Goodfellow et al., 2016, Ch. 7). We use dropout, weight decay and hidden
  WAPs, and the multi-task sharing itself acts as one (Ruder, 2017).

## 5. Trees, neural networks and hybrids

- **Trees are strong on tabular data.** On 45 medium-sized tabular datasets, tree-based models
  "remain state-of-the-art" (Grinsztajn et al., 2022), and MLPs are hurt by uninformative
  features. Their benchmark leaves out signal data (every column the same signal from a different
  sensor) and missing values, so a Wi-Fi scan (464 WAP columns, most of them "not heard") falls
  outside it: the result only suggests that a network may not beat our Phase 1 Random Forest, so
  the two must be compared on equal terms.
- **Ensembles of trees and networks.** On tabular data, an ensemble of deep models and XGBoost
  "performs better ... than XGBoost alone" (Shwartz-Ziv and Armon, 2022): a precedent for our blend
  of the Random Forest and the network.
- **Hand-made features in a network.** Wide & Deep learning (Cheng et al., 2016) trains a linear
  model on hand-made features jointly with a deep network and adds their outputs, so that each
  covers the other's weaknesses. Our hybrid network uses the idea in a simpler form: the Phase 1
  features are extra inputs of the network itself, with no separate linear part. The AP-affinity
  columns tell the network where each WAP is usually heard, which it would otherwise have to learn
  from the raw readings.
- **Other hybrids.** Some systems learn a feature space with a network and then match scans with
  weighted kNN in it (Park et al., 2024), or average such a kNN position with the network's own
  (Li and Kim, 2026); neither describes choosing its settings on a separate validation split (Li and
  Kim tuned theirs on the test file), so both results are likely optimistic.

## 6. The gap and how Phase 2 addresses it

1. **Unfair or unclear protocols.** Published deep results on UJIIndoorLoc often choose settings
   on the test file, report the best of many runs or re-split the data at random (Sect. 2). We
   keep Phase 1's protocol: every tuned setting chosen by user-group CV (the network design itself
   is fixed in advance by the brief), the test set used only at the end, and seed checks.
2. **Multi-task or not?** The evidence is mixed: one shared network is the standard design (Kim,
   2018), but a single-task network was better for position on UJIIndoorLoc (Laska and
   Blankenbach, 2022). Our ablation compares the shared network with three separate networks
   under the same inputs, settings and folds.
3. **Hybrids of our own Phase 1 and Phase 2.** We found no study that feeds a network the learned
   location profiles of the WAPs (our AP-affinity), or blends a tuned tree model with a multi-task
   network with weights chosen by user-group CV, with an ablation that shows whether they help.

## References

New keys in `report/references.bib`: ruder2017overview, kendall2018multi, kim2018simo,
laska2022multitask, li2025hierarchical, elesawi2021hierarchical, li2026hybrid,
alhomayani2020deep, kargarbarzi2024edge, eftekharizadeh2026physics (preprint),
srivastava2014dropout, goodfellow2016deep, nair2010rectified, kingma2015adam,
loshchilov2019decoupled, grinsztajn2022tree, shwartzziv2022tabular, cheng2016wide, park2024wknn,
paszke2019pytorch (the PyTorch library). Phase 1 keys used again: he2016wifi,
torres2014ujiindoorloc, caruana1997multitask, nowicki2017loweffort, kim2018scalable,
song2019cnnloc, roberts2017crossvalidation.
