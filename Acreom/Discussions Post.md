[https://github.com/SemiSim](https://github.com/SemiSim)

Directory Structure: Dual Repository

1. Repository for Matrix Builder:

 [https://github.com/SemiSim/llps_feature_builder](https://github.com/SemiSim/llps_feature_builder)
```bash
semisim-root/
├── setup.py
├── config.yaml               #points at 'data' , 'fasta'
├── README.md
├── split_data/    #split matrix output files generate here
│   ├── #features.csv    #the full, unsplit matrix
│   ├── #features_X.csv  #feature matrix w/o label column 
│   ├── #features_y.csv  #feature matrix w/ ONLY the llps_label values column 
│   ├── #X_train.csv     #feature only training set
│   ├── #y_train.csv     #corresponding training labels 
│   ├── #X_test.csv      #feature only test set
│   ├── #y_test.csv      #corresponding training labels   
│             
├── llps_feature_builder/      #matrix code package
│   ├── __init__.py      #exposes top-level API = clean & self-documenting
│   ├── main.py          
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── pondr.py
│   │   ├── saps.py
│   │   ├── plaac.py
│   │   ├── string.py
│   │   └── biogrid.py
│   ├── builder/
│   │   ├── __init__.py
│   │   └── matrix_builder.py
│   ├── utils/
│   │   ├── __init__.py
│   └── └── config_loader.py
│
├── data/
│   ├── pondr/
│   ├── plaac/
│   ├── saps/
│   ├── biogrid/
│   ├── string/
│   └── llpsdb.csv    #manually curated 468 x 50 file
│
├── fasta/      #FASTA files for NetSurfP predictions
│   ├── p_<proteinID>_fasta.txt       #wildtype  
│   ├── p_<proteinID>_mut_fasta.txt   #mutant number
│   ├── p_<proteinID>_iso_fasta.txt   #isoform
└── └── ...
```

with all these files in place & installed via:
```bash
pip install -e .

```

you can run this via CLI:
```bash
llps-feature-builder

```

✦  and it will:
✧ Read `config.yaml`
✧ Build and save the full feature matrix (including NetSurfP 3.0 predictions)
✧Split into train/test sets (80% of dataset = training ; 20% of dataset = testing)

✦ Matrix generation software is modular & fully customizable:
✧ If you wanted to train a model on only proteins relevant to Alzheimer's disease, for example, you could easily eliminate all irrelevant protein entries from the directory's 'data' file (many Tau, a/b Synuclein, and dozens of mutations for each are already loaded in the dataset provided)

✦ If you'd like to customize the directory structure, a script has already been created for you to automate generation of your new config.yaml file (see `README.md` for more information)

2. Model

[https://github.com/SemiSim/SemiSim-MU-/tree/main/mainfiles](https://github.com/SemiSim/SemiSim-MU-/tree/main/mainfiles)

Setup
```bash
#!/usr/bin/env python
# setup script for SemiSim
from setuptools import setup, find_packages
from codecs import open
from os import path
here = path.abspath(path.dirname(__file__))
# Get long description from README if available
try:
    with open(path.join(here, '[README.md](http://README.md)'), encoding='utf-8') as f:
        long_description = [f.read](http://f.read)()
except FileNotFoundError:
    long_description = 'A deep learning model for predicting liquid-liquid phase separation (LLPS) using sequence and structural features.'
setup(
    name='semisim',
    version='0.1.0',
    description='SemiSim: A deep learning framework for LLPS prediction using sequence and NetSurfP features',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='[https://github.com/SemiSim/SemiSim-MU-](https://github.com/SemiSim/SemiSim-MU-)',
    author='Moriah Miles',
    [author_email='your.email@example.com](mailto:author_email='your.email@example.com)',  # replace with your actual email
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit-learn',
        'matplotlib',  # optional
    ],
    extras_require={
        'dev': ['jupyter', 'pytest']
    },
    tests_require=['pytest'],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
)
```

Install
```bash
==================
Installation Guide
==================
Where to get SemiSim
---

The SemiSim source code and Jupyter notebook are available at  
``_  
You can clone the repository using Git or download the ZIP file directly from the site.
Requirements
---

SemiSim requires the following Python packages:
- `numpy `_
- `pandas `_
- `scikit-learn `_
- `tensorflow `_
You may also optionally install:
- `matplotlib` or `seaborn` for visualizations
Unix/OSX Installation

---

After installing Python (>=3.7), open a terminal and run:
.. code-block:: bash
    $ git clone https://github.com/SemiSim/SemiSim-MU-.git
    $ cd SemiSim-MU-
    $ pip install -r requirements.txt
    $ jupyter notebook
If `requirements.txt` is not available, install manually:
.. code-block:: bash
    $ pip install numpy pandas scikit-learn tensorflow
Windows Installation
---

Open Anaconda Prompt or Command Prompt and run:
.. code-block:: bash
    > git clone https://github.com/SemiSim/SemiSim-MU-.git
    > cd SemiSim-MU-
    > pip install -r requirements.txt
    > jupyter notebook
If Git is not installed, you can also download the ZIP directly from the GitHub page, extract it, and run the notebook manually.
```

Limitations of SemiSim Model
1. **Sequence Length Truncation** The model limits sequences to a maximum length (e.g., 100 residues), which may exclude critical regions involved in LLPS, especially for large or multi-domain proteins.
2. **Limited Feature Scope** Only a subset of biochemically relevant features (like NetSurfP-derived structure probabilities) is used. Missing additional descriptors (e.g., electrostatics, charge distribution, disorder-to-order transitions) might limit performance.
3. **Class Imbalance and Oversampling** The model uses **upsampling** to balance the dataset, which may lead to overfitting on minority class examples, rather than learning genuine biophysical distinctions.
4. **Black-Box Behavior** LSTM-based models offer little interpretability. It’s difficult to extract *why* a sequence was predicted to undergo LLPS — a challenge in explaining findings biologically.
5. **Monte Carlo Layer Limited** The intended **Monte Carlo simulation** layer for energy sampling is limited, meaning SemiSim currently lacks physical modeling of droplet energetics or molecular fluctuation — key elements in phase behavior.
6. **No Experimental Validation Loop** There is no comparison to real-world in vitro/in vivo LLPS observations beyond the labeled data. Thus, generalization to unseen biological systems is untested at this time.
7. **Model Generalization** The model is trained on **LLPSDB**, which may be biased toward well-studied systems. It may struggle with rare, atypical, or conditionally phase-separating proteins.
