# Dataset

The MLRSNet dataset is not included in this ZIP because the submission guide says to exclude large datasets.

Download MLRSNet from Mendeley Data:

https://data.mendeley.com/datasets/7j9bv9vwsx/3

After downloading, place the extracted files in the project root as:

```text
train_data/
|-- Images/
`-- labels/
```

Then run:

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\python.exe src/dataset_splitter.py
```
