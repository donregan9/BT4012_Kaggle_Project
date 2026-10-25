# BT4012 Bitcoin Transaction Detection

## Dataset placement

The dataset is not included in this repository. Create a folder named
`data` in the repository root and place the four competition files inside it:

```text
BT4012_Kaggle_Project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── txs_edgelist.csv
│   └── sample_submission.csv
├── e1157135.ipynb
└── requirements.txt
```

The notebook reads the files from this relative path:

```python
DATA_DIR = Path("data")
```

Run the notebook from the repository root so that the `data/` directory is
found correctly. The generated prediction file is:

```text
submission_improved_temporal_graph_ensemble.csv
```
