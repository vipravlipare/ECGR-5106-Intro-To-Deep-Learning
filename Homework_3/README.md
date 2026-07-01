# Homework 3

Homework 3 is for sequence-to-sequence machine translation using the provided `vast_english_french.txt` dataset.

This folder includes the report, notebooks, dataset, and saved results for all three problems.

## Files

- `Homework 3 Report - ECGR 5106 - Viprav Lipare.pdf`: Final PDF report.
- `Homework3_Problem1_Baseline_GRU_Seq2Seq.ipynb`: Problem 1 baseline GRU encoder-decoder for English to French translation.
- `Homework3_Problem2_Attention_GRU_Seq2Seq.ipynb`: Problem 2 GRU encoder-decoder with attention for English to French translation.
- `Homework3_Problem3_Reversed_Translation_GRU_Attention.ipynb`: Problem 3 reversed French to English translation using both baseline and attention models.
- `vast_english_french.txt`: Dataset used for all three problems.
- `Results_Problem_1/`: Saved Problem 1 plots, summaries, split file, and validation results.
- `Results_Problem_2/`: Saved Problem 2 plots, attention maps, summaries, and validation results.
- `Results_Problem_3/`: Saved Problem 3 plots, attention maps, summaries, comparison files, and validation results.

## Notes

The same 80/20 train-validation split is used across the homework so the models can be compared fairly. The notebooks save the main metrics, loss curves, sample translations, and comparison plots used in the report.
