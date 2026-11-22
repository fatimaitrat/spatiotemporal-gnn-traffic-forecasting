# Traffic forecasting with fixed, adaptive and hybrid graphs (LargeST San Diego)

Forecast the next four 15-minute traffic-flow intervals (1 hour) for 716 San Diego sensors from the previous eight (2 hours),
comparing a temporal-only model with spatio-temporal GNNs using a fixed road graph, a learned adaptive graph and a hybrid graph.

## Results (test set, Apr-Dec 2021; vehicles per 15 min; one run, seed 42)
| Model | Adjacency | Params | Val MAE | Val RMSE | Test MAE | Test RMSE | Train time (s) |
|---|---|---|---|---|---|---|---|
| Temporal-only | None | 10,788 | 57.70 | 93.38 | 65.93 | 104.20 | 259.7 |
| STGNN | Fixed | 10,788 | 52.41 | 84.77 | 60.79 | 95.87 | 506.7 |
| STGNN | Adaptive | 25,108 | 44.57 | 72.64 | 53.17 | 83.92 | 501.2 |
| STGNN | Hybrid | 25,109 | 44.36 | 72.22 | 53.47 | 84.39 | 500.4 |

## Reproduce
1. Open `traffic_forecasting_stgnn.ipynb` in Google Colab with a GPU runtime (we used a Tesla T4).
2. Create a Kaggle API token. The first cell asks for it via a hidden prompt and downloads the LargeST dataset
   (`kaggle datasets download liuxu77/largest`, about 7 GB; CC BY-NC 4.0, not redistributed here).
3. Run all cells top to bottom. The notebook extracts the 716 District-11 sensors for 2020-2021, resamples to 15 minutes
   (sum of counts), handles missing values, builds the windows, and trains the four models (about 8 min each on a T4).

## Configuration
Seed 42; hidden size 32 (head 64); embedding dimension 10; Adam, lr 1e-3; batch size 64; at most 20 epochs with early stopping
(patience 4 on validation MAE); masked MAE loss; splits: train Jan-Dec 2020, validation Jan-Mar 2021, test Apr-Dec 2021.

## Notes
- A skip connection was added to the graph layer, H = Z + ReLU(A Z W). The no-skip result is in `results/results_q4_noskip.json`.
- GPU training is not bit-for-bit deterministic; repeated runs with the same seed differ slightly in the second decimal.
- Data: LargeST (Liu et al., NeurIPS 2023), https://github.com/liuxu77/LargeST
