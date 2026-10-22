# Dataset card: UJIIndoorLoc

Wi-Fi scans recorded with phones in three buildings of Universitat Jaume I (Castellón, Spain): the
signal strength (RSSI, in dBm) of 520 Wi-Fi access points (WAPs), plus where, when and by whom.

- **Source and license:** UCI Machine Learning Repository, [dataset 310](https://archive.ics.uci.edu/dataset/310/ujiindoorloc),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (free to share and reuse if you credit the authors).
- **Citation:** J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor
  database for WLAN fingerprint-based indoor localization problems," in *Proc. IPIN 2014*, Busan,
  South Korea, pp. 261–270, doi: [10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492).

## Files and columns (`raw/UJIndoorLoc/`, the same 529 columns in both files)

| File | Scans | Recorded | Our use |
| --- | ---: | --- | --- |
| `trainingData.csv` | 19,937 | 30 May to 20 June 2013 | training: cleaning, cross-validation, final fit |
| `validationData.csv` | 1,111 | 19 Sep. to 8 Oct. 2013 | our **test set**: used once, at the end; never cleaned |

| Column | Meaning |
| --- | --- |
| `WAP001` … `WAP520` | RSSI of each WAP in dBm (−104 to 0 in training); **100 = not heard** |
| `LONGITUDE`, `LATITUDE` | position in Web-Mercator meters (one such meter is about 0.77 m on the ground) |
| `FLOOR`, `BUILDINGID` | floor (0–3 in buildings 0 and 1, 0–4 in building 2) and building (0, 1 or 2) |
| `SPACEID`, `RELATIVEPOSITION` | id of the space (office, corridor or classroom); 1 = inside it, 2 = in front of its door (both 0 in the test file) |
| `USERID`, `PHONEID`, `TIMESTAMP` | who recorded the scan (users 1–18; 0 in the test file), with which phone, and when (Unix time); users and phones define our CV groups |

## How we prepare the data

- **Not heard.** 100 is a code, not a signal. In both files we replace it with −110 dBm, weaker than
  the weakest real reading (−104 dBm), so "not heard" simply means "very weak".
- **Cleaning, training file only.** Three rules, in this order ([cleaning.csv](../results/metrics/cleaning.csv)):

| Step | Scans removed | Scans left |
| --- | ---: | ---: |
| raw training file | 0 | 19,937 |
| no WAP heard (the scan says nothing about where it was taken) | 76 | 19,861 |
| exact duplicate row (the first copy is kept) | 634 | 19,227 |
| a reading of −30 dBm or stronger (implausible; most come from one phone) | 508 | 18,719 |

- **The test file is never cleaned:** all 1,111 test scans are scored; a real system cannot choose its scans.
- **Kept WAPs:** only the WAPs heard in the training data are used (464 after cleaning).

## Download and check

`make data` runs [download_data.py](../scripts/download_data.py): it downloads the UCI zip only if a
file is missing, then checks both files against the SHA-256 checksums in [config.py](../src/indoorloc/config.py)
and prints `OK` or `FAILED` for each.
