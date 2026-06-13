# Music Instrument Classification using MedleyDB

This repository contains the implementation for our Advanced Technology Integration 2026 project at Universidad de Jaén.

The project focuses on automatic music instrument classification using isolated instrument stems from the MedleyDB dataset. The system reads MedleyDB metadata files, extracts handcrafted audio features from WAV stems, trains several machine learning models, evaluates their performance, and performs additional metadata analysis.

## Project Information

Course: Advanced Technology Integration 2026
University: Universidad de Jaén
Project: Music Instrument Classification
Dataset: MedleyDB
Notebook: MIC_Project_MercurioNieves.ipynb

## Project Motivation

Music instrument classification is an important task in Music Information Retrieval. It can be useful for automatic music tagging, audio indexing, music search, music recommendation, and analysis of multitrack recordings.

In this project, we focus on classifying individual isolated stems from MedleyDB. Each stem usually contains one active instrument, so the task is treated as a single-label classification problem.

## Research Question

The main research question is:

Can handcrafted audio features extracted from isolated MedleyDB stems be used to classify musical instruments using traditional machine learning models?

## Dataset

The project uses the MedleyDB dataset, which contains professionally recorded multitrack songs with isolated instrument stems and metadata annotations.

The dataset is not included in this repository because of its large size and licensing conditions. The user must download MedleyDB separately and store it locally.

Expected local dataset structure:

```text
MedleyDB/
    Audio/
        SongName/
            SongName_METADATA.yaml
            SongName_MIX.wav
            SongName_STEMS/
                SongName_STEM_01.wav
                SongName_STEM_02.wav
```

In our local experiment, the dataset path was:

```text
C:\MDB\download_2026-06-03_18-52-53\MedleyDB
```

In the notebook, the user must edit `MEDLEYDB_PATH` in Cell 2 to point to the local MedleyDB folder.

Example:

```python
RUN_MODE = 'local'
MEDLEYDB_PATH = r'C:\MDB\download_2026-06-03_18-52-53\MedleyDB'
```

## Instrument Classes

The selected instrument classes were:

```text
clean electric guitar
acoustic guitar
piano
drum set
electric bass
violin
male singer
female singer
trumpet
tenor saxophone
```

## Methodology

The system follows this pipeline:

1. Read MedleyDB YAML metadata files.
2. Extract instrument labels for each stem.
3. Select stems belonging to the chosen instrument classes.
4. Skip near-silent stems using an RMS threshold of 0.01.
5. Load each WAV stem as mono audio.
6. Resample audio to 22050 Hz.
7. Extract a 296-dimensional feature vector from each stem.
8. Use a song-level train and test split to prevent data leakage.
9. Normalize features using StandardScaler fitted only on the training set.
10. Train and compare five machine learning models.
11. Evaluate models using accuracy, F1-macro, F1-micro, classification report, and confusion matrix.
12. Perform additional metadata analysis using genre distribution and instrument co-occurrence.

No data augmentation was applied. Data augmentation is suggested as future work.

## Feature Extraction

Each stem is represented using 296 features:

```text
7 spectral features x mean and standard deviation = 14
13 MFCC coefficients x mean and standard deviation = 26
128 mel-spectrogram bands x mean and standard deviation = 256
Total = 296 features
```

Audio settings:

```text
Sample rate: 22050 Hz
N FFT: 2048
Hop length: 512
MFCC coefficients: 13
Mel bands: 128
Clip duration: 5 seconds
Silence threshold: RMS 0.01
```

## Models Tested

The following models were trained and compared:

```text
KNN, k=5
Random Forest
SVM Linear
SVM RBF
MLP Neural Network
```

KNN with k=5 was used as the baseline model.

## Final Dataset After Filtering

After selecting the target instruments and applying silent stem filtering, the final dataset used for classification was:

```text
Metadata rows loaded: 856
Songs loaded: 122
Genres found: 9
Instruments found: 82
Total usable stems: 179
Classes: 10
Train stems: 144
Test stems: 35
Feature size: 296
```

The train and test split was performed at song level. This means that stems from the same song were not allowed to appear in both the training and test sets.

Leakage check result:

```text
Overlap between train and test songs: 0
Leakage check: PASS
```

## Final Results

Model comparison:

| Model              | Accuracy | F1-macro | F1-micro |
| ------------------ | -------: | -------: | -------: |
| KNN, k=5           |    45.7% |    0.395 |    0.457 |
| Random Forest      |    57.1% |    0.478 |    0.571 |
| SVM Linear         |    77.1% |    0.700 |    0.771 |
| SVM RBF            |    62.9% |    0.553 |    0.629 |
| MLP Neural Network |    54.3% |    0.468 |    0.543 |

Best model:

```text
SVM Linear
Accuracy: 77.1%
F1-macro: 0.700
Cross-validation F1: 0.539 ± 0.049
```

The SVM Linear model achieved the best performance. It clearly improved over the KNN baseline, which achieved 45.7% accuracy.

## Error Analysis

The confusion matrix showed that the model performed well for several classes, especially:

```text
drum set
electric bass
female singer
piano
```

The weaker classes were:

```text
clean electric guitar
violin
trumpet
tenor saxophone
```

The main limitation was class imbalance after silent stem filtering. Some classes had very few usable examples, especially trumpet and tenor saxophone. This made them harder to evaluate and classify.

Some errors are also related to shared timbral characteristics. For example, clean electric guitar, electric bass, and violin can share harmonic content, which may cause confusion when using summary audio features such as MFCCs and mel-spectrogram statistics.

## Additional Metadata Analysis

In addition to the main classification experiment, the notebook includes exploratory metadata analysis using the MedleyDB YAML files.

This analysis was added to better understand the musical context of the selected instrument classes.

The additional analysis includes:

```text
1. Instrument distribution across genres
2. Instrument co-occurrence matrix
```

The genre distribution analysis shows how the selected instrument classes are distributed across the available MedleyDB genre annotations.

The co-occurrence matrix shows which instruments commonly appear together in the same multitrack songs. This helps explain the musical context of the dataset and supports discussion about overlapping instruments in real music mixtures.

Top instrument co-occurrences found in the dataset:

```text
drum set + electric bass: 54 songs
drum set + male singer: 33 songs
electric bass + male singer: 32 songs
clean electric guitar + drum set: 31 songs
clean electric guitar + electric bass: 29 songs
piano + drum set: 21 songs
clean electric guitar + male singer: 18 songs
acoustic guitar + electric bass: 16 songs
piano + electric bass: 15 songs
drum set + female singer: 15 songs
```

These results show that rhythm-section instruments such as drums and bass frequently appear together, which matches typical band arrangements.

## Output Files

The notebook saves the following files in the `MIC_Project_Output` folder:

```text
X_features.npy
y_labels.npy
groups_songs.npy
scaler.pkl
best_model.pkl
class_distribution.png
confusion_matrix_best.png
confusion_matrix_knn.png
feature_importance.png
model_comparison.png
instrument_genre_distribution.png
cooccurrence_matrix.png
spectrogram_acoustic_guitar.png
spectrogram_clean_electric_guitar.png
spectrogram_drum_set.png
spectrogram_electric_bass.png
spectrogram_piano.png
```

The most important figures for the report and presentation are:

```text
class_distribution.png
model_comparison.png
confusion_matrix_best.png
feature_importance.png
instrument_genre_distribution.png
cooccurrence_matrix.png
```

## Software Requirements

The project was implemented using Python 3 and Jupyter Notebook.

Required libraries:

```text
notebook
ipykernel
librosa
scikit-learn
seaborn
pyyaml
tqdm
joblib
soundfile
matplotlib
pandas
numpy
```

## How to Reproduce the Results

### 1. Clone or download the repository

Download this repository to your computer and open the project folder.

Expected project files:

```text
MIC_Project_MercurioNieves.ipynb
README.md
```

### 2. Prepare the MedleyDB dataset

Download the MedleyDB dataset separately. The dataset is not included in this repository because it is large and has its own licensing conditions.

The expected folder structure is:

```text
MedleyDB/
    Audio/
        SongName/
            SongName_METADATA.yaml
            SongName_MIX.wav
            SongName_STEMS/
                SongName_STEM_01.wav
                SongName_STEM_02.wav
```

### 3. Create and activate a virtual environment

Open Command Prompt inside the project folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
```

If `python` does not work, use:

```bash
py -m venv .venv
.venv\Scripts\activate
```

### 4. Install the required dependencies

Run:

```bash
pip install notebook ipykernel librosa scikit-learn seaborn pyyaml tqdm joblib soundfile matplotlib pandas numpy
```

If there is an error related to `pkg_resources`, run:

```bash
pip install --force-reinstall "setuptools==80.9.0"
```

### 5. Register the Jupyter kernel

Run:

```bash
python -m ipykernel install --user --name mic_project --display-name "Python (MIC Project)"
```

### 6. Open Jupyter Notebook

Run:

```bash
jupyter notebook
```

Then open:

```text
MIC_Project_MercurioNieves.ipynb
```

### 7. Set the dataset path

In Cell 2, set:

```python
RUN_MODE = 'local'
MEDLEYDB_PATH = r'C:\MDB\download_2026-06-03_18-52-53\MedleyDB'
```

Change `MEDLEYDB_PATH` if your MedleyDB folder is in a different location.

### 8. Run the notebook

Run the cells in order.

Important note:

```text
Cell 7 performs feature extraction and may take time.
After features are saved, future sessions can start from Cell 8.
```

The notebook will reproduce:

```text
Dataset filtering
Feature extraction
Train and test split
Model training
Model evaluation
Confusion matrices
Feature importance
Genre distribution analysis
Instrument co-occurrence analysis
```

### 9. Check the output folder

After running the notebook, the results will be saved in:

```text
MIC_Project_Output/
```

Important output files include:

```text
X_features.npy
y_labels.npy
groups_songs.npy
scaler.pkl
best_model.pkl
class_distribution.png
model_comparison.png
confusion_matrix_best.png
confusion_matrix_knn.png
feature_importance.png
instrument_genre_distribution.png
cooccurrence_matrix.png
```

These files reproduce the reported results and figures used in the manuscript and presentation.

## Important Notes

The MedleyDB audio dataset is not uploaded to this repository.

The repository only includes the notebook, README file, and project implementation. The dataset must be downloaded separately and placed locally by the user.

The main classification experiment focuses on isolated stems. Overlapping instrument mixtures are discussed as a limitation and future extension.

Future work could include:

```text
polyphonic instrument recognition
multi-label classification
genre-based analysis
co-occurrence analysis
data augmentation
deep learning models
longer audio segments
more balanced class selection
```

## References

[1] R. M. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello, “MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research,” Proceedings of the 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, 2014.

[2] S. Doh, “Music Instrument Classification using Machine Learning,” GitHub repository, 2019.

[3] V. S. Kadandale, “Musical Instrument Recognition in Multi-Instrument Audio Contexts,” Master’s thesis, Universitat Pompeu Fabra, Barcelona, 2018.

[4] E. J. Humphrey, S. Durand, and B. McFee, “OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition,” Proceedings of the 19th International Society for Music Information Retrieval Conference, Paris, France, 2018.

[5] J. C. Brown, “Computer identification of musical instruments using pattern recognition with cepstral coefficients as features,” Journal of the Acoustical Society of America, 1999.

[6] MedleyDB, “MedleyDB Downloads,” official dataset download page.

[7] MedleyDB, “MedleyDB Description,” official dataset description page.
