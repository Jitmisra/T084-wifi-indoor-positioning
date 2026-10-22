# Literature Review

Team T084 · Phase 1 of "Indoor Location Prediction Using Wi-Fi Signals"

The background of our project, in six short parts. The results we quote from papers are listed,
with their place in the paper, in [literature_matrix.csv](literature_matrix.csv); the keys in the
reference list match `report/references.bib`.

## 1. Wi-Fi fingerprinting basics

Liu et al. (2007) group indoor positioning methods into triangulation, scene analysis and
proximity, and discuss location fingerprinting in detail because most systems use it. Wi-Fi is
attractive because wireless networks are already installed, and fingerprinting needs no
line-of-sight measurement to the access points (He and Chan, 2016).

A fingerprint is one scan: the received signal strength (RSSI, in dBm) of every access point the
phone hears. Offline, scans are recorded at known positions; these labeled scans form the *radio
map* (our training data). Online, a new scan is located by comparing it with the radio map.

- **RADAR** (Bahl and Padmanabhan, 2000) returns the stored scans closest to the new one in
  Euclidean distance. On one floor of an office building (three base stations, one laptop) its
  median error was 2.94 m; averaging k = 5 neighbors helped a little (2.75 m), and a large k made
  it worse. Its propagation model shows why this works: the signal fades with distance and walls.
- **Horus** (Youssef and Agrawala, 2005) replaced the closest match with a probabilistic model of
  how the signal varies and reported an average error below 0.6 m.

Both were tested on the authors' own small testbeds, so their numbers cannot be compared with each
other or with ours. This is why a shared public dataset is needed.

## 2. The UJIIndoorLoc dataset

UJIIndoorLoc (Torres-Sospedra et al., 2014) was presented as the first public Wi-Fi fingerprinting
database covering several buildings and floors, recorded by more than 20 users with 25 devices. The
UCI release has 19,937 training and 1,111 validation scans from three buildings of Universitat Jaume
I, each with 520 WAP readings (100 = not heard), position, floor, building, user, phone and time.

The validation file was recorded months later, partly by users and phones that were not part of
the training collection (Torres-Sospedra et al., 2015). This makes it a realistic test, so we use it
as our test set. In our copy, training scans are from 30 May to 20 June 2013, test scans from
19 September to 8 October 2013, and 9 of the 11 test phones never appear in training.

We could not read the 2014 paper itself (closed access). Later papers quote 89.92% as its baseline
(Kim et al., 2018; Song et al., 2019). The same authors report this number in their 2015 paper:
1-NN with Euclidean distance gets building *and* floor right for 89.92% of the validation scans,
with a mean 2D error of 7.90 m measured only over those scans (Torres-Sospedra et al., 2015).

## 3. Classical models

**kNN** predicts from the k most similar training scans. With unlimited data, the error of the
1-nearest-neighbor rule is at most twice the lowest error any rule can reach, the Bayes error
(Cover and Hart, 1967). kNN needs no training, but every WAP enters every distance, even an
irrelevant one, and results depend on how the readings are transformed and on the distance used.

The most systematic kNN study on UJIIndoorLoc (Torres-Sospedra et al., 2015) compared 51 distance
and similarity measures and four ways of transforming the readings. Their best setting got building
and floor right for 95.2% of the validation scans, with a mean 2D error of 6.19 m over those scans
only, but it was chosen on the same file on which it is reported, so it is optimistic.
One of their transformations, *powed*, shifts and scales a reading to the range 0 to 1 and raises
it to the power e, so strong readings count much more than weak ones, which come from far-away WAPs
and fluctuate more. We use this formula only as the weight inside AP-affinity. A comparison of
several classifiers on UJIIndoorLoc found kNN the most suitable, with bagged or boosted decision
trees close to it (Bozkurt et al., 2015; we read only the abstract).

**Random Forest** (Breiman, 2001): many decision trees, each grown on a random resample of the
scans and trying a random subset of features at each split; the trees vote (building, floor) or
average (position). Breiman proved that adding trees does not cause overfitting, so we fix 300
trees and do not tune their number.

**XGBoost** (Chen and Guestrin, 2016): gradient-boosted trees, added one at a time, each correcting
the errors of the trees before it, with a regularized objective (a penalty on complex trees). We use
scikit-learn (Pedregosa et al., 2011) for kNN and Random Forest, and the XGBoost library.

## 4. Neural networks on UJIIndoorLoc

- **Nowicki and Wietrzykowski (2017)** used a stacked autoencoder (SAE, a network that learns a
  compressed summary of the scan) and a classifier with one output per building-floor pair: 92% of
  the validation scans got building and floor right, against 75% for their simple 1-NN; positions
  were left for future work. Coding "not heard" as +100 instead of −110 dBm (same network) lowered
  the validation accuracy from 91.1% to 84.5%, which supports our choice of −110 dBm.
- **Kim et al. (2018)** trained one network (an SAE and a multi-label classifier) with building,
  floor and location outputs: 99.82% building and 91.27% floor accuracy and a 9.29 m error. The
  three tasks share one network, so this is a multi-task design and the closest precedent for our
  Phase 2. Their split is unclear (70:30 in the text, 0.90 in a table).
- **CNNLoc** (Song et al., 2019) combined an SAE with a one-dimensional convolutional network and
  reports 100% building and 96.03% floor accuracy and an 11.78 m error on the validation file. These
  are the best values of different runs, not of one model: the run with 96.03% floor accuracy has an
  18.10 m error, and the run with 11.78 m has 94.68% floor accuracy.

The deep models do well on building and floor, but their position errors (9.29 m and 11.78 m) are
larger than those of kNN on the validation file (6.19 to 7.90 m). The error definitions differ, so
this is only a rough comparison, but it gives no evidence that deep models locate scans better. A
new network must therefore be compared with strong, tuned classical models.

## 5. Fair evaluation

- **Leakage** is "the introduction of information about the data mining target that should not be
  legitimately available to mine from" (Kaufman et al., 2012). Our risks: learning kept WAPs or
  AP-affinity profiles from scans that are later scored, cleaning the test file, and choosing a
  model by its test score.
- **Grouped data.** When data have structure (in time, in space or in groups), random
  cross-validation can seriously underestimate the error; holding out whole blocks is "nearly
  universally more appropriate" for predicting new data, although it can force extrapolation
  (Roberts et al., 2017).
- **Evidence on UJIIndoorLoc.** One network got building and floor right for 98.2% of a random split
  of the training file but for 91.1% of the validation file (Nowicki and Wietrzykowski, 2017), and
  the best kNN setting above was chosen on the file it reports (Torres-Sospedra et al., 2015).

Our answer is user-group cross-validation (CV): whole users and their phones are held out (users who
shared a phone form one group), everything learned from data is refitted inside each training fold,
and the test set is used once, at the end. Our own check agrees: for the Baseline RF, random CV
promised a mean 2D error of 3.67 m; the test set gave 12.52 m. (Mean 2D error = straight-line
distance between predicted and true position; our errors are in the dataset's Web-Mercator meters,
and one such meter is about 0.77 m on the ground.)

## 6. The gap and how we address it

**Phase 1.** We found no published study on UJIIndoorLoc that compares tuned kNN, Random Forest and
XGBoost while making every choice with user-group CV and using the test file only once. Phase 1
does this, and adds AP-affinity: the training scans that hear a WAP tell us roughly where it is
(building, floor, location), and a scan is then described by the WAPs it hears.

**Phase 2.** In Phase 1, every method uses three separate models (building, floor, position) that
share nothing. Multi-task learning "improves generalization by using the domain information
contained in the training signals of related tasks as an inductive bias", by learning the tasks in
parallel with a shared representation (Caruana, 1997). Building, floor and position describe one
location, so they are related tasks. Kim et al. (2018) show that one shared network works on
UJIIndoorLoc, but not that it beats strong tree models on position.

Our Phase 1 results point to this gap. No model won every task: on the test set, Random Forest had
the lowest mean 2D error (7.42 m) and XGBoost the best floor accuracy (95.2%). AP-affinity, a
hand-made way of sharing location knowledge between the tasks, cut the Baseline RF's CV mean 2D
error from 15.87 m to 11.38 m. Phase 2 will train one multi-task neural network (shared layers,
three output heads) and compare it with Phase 1 on the same split, test set and metrics.

## References

- Bahl, P., and Padmanabhan, V. N. (2000). RADAR: An in-building RF-based user location and tracking system. *Proc. IEEE INFOCOM 2000*, vol. 2, pp. 775–784. doi:10.1109/INFCOM.2000.832252 · `bahl2000radar`
- Bozkurt, S., Elibol, G., Gunal, S., and Yayan, U. (2015). A comparative study on machine learning algorithms for indoor positioning. *Proc. INISTA 2015*, pp. 1–8. doi:10.1109/INISTA.2015.7276725 · `bozkurt2015comparative`
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. doi:10.1023/A:1010933404324 · `breiman2001random`
- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41–75. doi:10.1023/A:1007379606734 · `caruana1997multitask`
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. KDD '16*, pp. 785–794. doi:10.1145/2939672.2939785 · `chen2016xgboost`
- Cover, T., and Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. doi:10.1109/TIT.1967.1053964 · `cover1967nearest`
- He, S., and Chan, S.-H. G. (2016). Wi-Fi fingerprint-based indoor positioning: Recent advances and comparisons. *IEEE Communications Surveys & Tutorials*, 18(1), 466–490. doi:10.1109/COMST.2015.2464084 · `he2016wifi`
- Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15:1–15:21. doi:10.1145/2382577.2382579 · `kaufman2012leakage`
- Kim, K. S., Lee, S., and Huang, K. (2018). A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting. *Big Data Analytics*, 3(1), Art. no. 4. doi:10.1186/s41044-018-0031-2 · `kim2018scalable`
- Liu, H., Darabi, H., Banerjee, P., and Liu, J. (2007). Survey of wireless indoor positioning techniques and systems. *IEEE Transactions on Systems, Man, and Cybernetics, Part C*, 37(6), 1067–1080. doi:10.1109/TSMCC.2007.905750 · `liu2007survey`
- Nowicki, M., and Wietrzykowski, J. (2017). Low-effort place recognition with WiFi fingerprints using deep learning. In *Automation 2017*, AISC vol. 550, pp. 575–584. Springer. doi:10.1007/978-3-319-54042-9_57 · `nowicki2017loweffort`
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12(85), 2825–2830. · `pedregosa2011scikit`
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. doi:10.1111/ecog.02881 · `roberts2017crossvalidation`
- Song, X., et al. (2019). A novel convolutional neural network based indoor localization framework with WiFi fingerprinting. *IEEE Access*, 7, 110698–110709. doi:10.1109/ACCESS.2019.2933921 · `song2019cnnloc`
- Torres-Sospedra, J., Montoliu, R., Martínez-Usó, A., Avariento, J. P., Arnau, T. J., Benedito-Bordonau, M., and Huerta, J. (2014). UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems. *Proc. IPIN 2014*, pp. 261–270. doi:10.1109/IPIN.2014.7275492 · `torres2014ujiindoorloc`
- Torres-Sospedra, J., Montoliu, R., Trilles, S., Belmonte, Ó., and Huerta, J. (2015). Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems. *Expert Systems with Applications*, 42(23), 9263–9278. doi:10.1016/j.eswa.2015.08.013 · `torres2015comprehensive`
- Youssef, M., and Agrawala, A. (2005). The Horus WLAN location determination system. *Proc. MobiSys '05*, pp. 205–218. doi:10.1145/1067170.1067193 · `youssef2005horus`
