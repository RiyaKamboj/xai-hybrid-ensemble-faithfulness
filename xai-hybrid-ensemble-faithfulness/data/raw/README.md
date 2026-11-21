# Raw data

The raw files are not redistributed here. Download each zip into this folder under the file name shown. `scripts/01_verify_data.py`, `07a_prepare_study2.py` and `07b_prepare_cdc.py` check the SHA-256 fingerprints and stop if a file differs.

| Dataset | Source | File name here | SHA-256 of the zip |
|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | UCI id 17, https://archive.ics.uci.edu/dataset/17 | `breast+cancer+wisconsin+diagnostic.zip` | `bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af` |
| Diabetic Retinopathy Debrecen | UCI id 329, https://archive.ics.uci.edu/dataset/329 | `diabetic+retinopathy+debrecen.zip` | `64ee2dbaffc69dab77cc0fb7458fa93cb21a20f15938cafacdc195a5e2226b2c` |
| Heart Disease (Cleveland) | UCI id 45, https://archive.ics.uci.edu/dataset/45 | `heart+disease.zip` | `b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc` |
| ILPD (Indian Liver Patient) | UCI id 225, https://archive.ics.uci.edu/dataset/225 | `ilpd+indian+liver+patient+dataset.zip` | `60e18fbfe7f84daf58c178ea280ba5fe8db7e2507aabce01b57425b3b822cf0b` |
| Parkinsons | UCI id 174, https://archive.ics.uci.edu/dataset/174 | `parkinsons.zip` | `9baccfbcdac15d6d9347b37c2a352fb31d65c52e367df5a47d7543a55c75b0ab` |
| Mammographic Mass | UCI id 161, https://archive.ics.uci.edu/dataset/161 | `mammographic+mass.zip` | `4c5158166aa7f1503e14d95be4c95fe66a734474f614441ede226314b93db6e7` |
| Cardiotocography | UCI id 193, https://archive.ics.uci.edu/dataset/193 | `cardiotocography.zip` | `ba2853564e282779907242b7bd8a83265d9fc4e79bb008573967daa16677bb6c` |
| CDC Diabetes Health Indicators (BRFSS 2015) | Kaggle release "Diabetes Health Indicators Dataset" (three CSV files; the same data is listed as UCI id 891) | `cdc_diabetes_health_indicators.zip` | `ecbd65727a3b9b730bfd3ba547a181767522f1188fb16077e885945c1030f046` |

**UCI licence.** The UCI datasets are distributed under CC BY 4.0.

**CDC sample.** The CDC data come from the public Behavioral Risk Factor Surveillance System. The 5,000-respondent analysis sample is defined by `data/processed/cdc_diabetes_source_rows.csv`, which lists row indices into `diabetes_binary_health_indicators_BRFSS2015.csv`.

**Not analysed.** `diabetes130.zip` (UCI id 296; SHA-256 `f82ac129…`) was downloaded but has not been analysed.
