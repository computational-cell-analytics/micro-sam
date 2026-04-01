## Time benchmarking

### GPU (NVIDIA Quadro RTX5000 - 16GB VRAM)
(in seconds, if not specified)

| Model | Experiment           | Inference Time (for 5 runs)       | Chosen |
|-------|----------------------|-----------------------------------|--------|
| vit_t | Precompute embeddings | 0.039, 0.039, 0.039, 0.040, 0.041 | 0.039s |
| vit_b | Precompute embeddings | 0.202, 0.203, 0.205, 0.204, 0.205 | 0.202s |
| vit_l | Precompute embeddings | 0.482, 0.486, 0.479, 0.488, 0.483 | 0.479s |
| vit_h | Precompute embeddings | 0.876, 0.874, 0.895, 0.884, 0.885 | 0.874s |
| vit_t | AMG (w/o embeddings) | 3.392, 3.377, 3.375, 3.376, 3.376 | 3.375s  |
| vit_b | AMG (w/o embeddings) | 3.378, 3.380, 3.381, 3.391, 3.392 | 3.378s  |
| vit_l | AMG (w/o embeddings) | 3.374, 3.379, 3.376, 3.376, 3.384 | 3.374s  |
| vit_h | AMG (w/o embeddings) | 3.362, 3.366, 3.367, 3.360, 3.362 | 3.360s  |
| vit_t | AMG (w. embeddings)  | 3.411, 3.416, 3.423, 3.442, 3,418 | 3.411s  |
| vit_b | AMG (w. embeddings)  | 3.578, 3.593, 3.592, 3.592, 3.589 | 3.578s  |
| vit_l | AMG (w. embeddings)  | 3.892, 3.901, 3.889, 3.889, 3.901 | 3.889s  |
| vit_h | AMG (w. embeddings)  | 4.265, 4.268, 4.270, 4.272, 4.251 | 4.251s  |
| vit_t | AIS (w/o embeddings) | 0.260, 0.259, 0.267, 0.261, 0.260 | 0.259s  |
| vit_b | AIS (w/o embeddings) | 0.259, 0.255, 0.259, 0.258, 0.257 | 0.255s  |
| vit_l | AIS (w/o embeddings) | 0.248, 0.248, 0.249, 0.247, 0.250 | 0.247s  |
| vit_h | AIS (w/o embeddings) | 0.249, 0.251, 0.249, 0.250, 0.249 | 0.249s  |
| vit_t | AIS (w. embeddings)  | 0.298, 0.301, 0.301, 0.301, 0.307 | 0.298s  |
| vit_b | AIS (w. embeddings)  | 0.465, 0.465, 0.462, 0.464, 0.463 | 0.462s  |
| vit_l | AIS (w. embeddings)  | 0.747, 0.744, 0.749, 0.749, 0.751 | 0.744s  |
| vit_h | AIS (w. embeddings)  | 1.132, 1.131, 1.145, 1.141, 1.142 | 1.131s  |
| vit_t | Point (w/o embeddings) (in ms) | 9.65, 9.70, 9.64, 9.66, 9.62 | 9.62ms |
| vit_b | Point (w/o embeddings) (in ms) | 9.68, 9.70, 9.65, 9.89, 9.72 | 9.65ms |
| vit_l | Point (w/o embeddings) (in ms) | 9.39, 9.46, 9.46, 9.45, 9.47 | 9.39ms |
| vit_h | Point (w/o embeddings) (in ms) | 9.70, 9.60, 9.67, 9.65, 9.69 | 9.60ms |
| vit_t | Point (w. embeddings) (in ms)  | 9.64, 9.72, 9.66, 9.72, 9.67 | 9.64ms |
| vit_b | Point (w. embeddings) (in ms)  | 10.90, 10.95, 11.04, 10.99, 10.98 | 10.90ms |
| vit_l | Point (w. embeddings) (in ms)  | 12.74, 12.74, 12.76, 12.94, 12.78 | 12.74ms |
| vit_h | Point (w. embeddings) (in ms)  | 15.68, 15.69, 15.74, 15.85, 15.72 | 15.68ms |
| vit_t | Box (w/o embeddings) (in ms) | 8.76, 8.59, 8.69, 8.56, 8.62  | 8.56ms |
| vit_b | Box (w/o embeddings) (in ms) | 8.44, 8.63, 8.59, 8.54, 8.54 | 8.44ms |
| vit_l | Box (w/o embeddings) (in ms) | 8.59, 8.59, 8.74, 8.59, 8.59 | 8.59ms |
| vit_h | Box (w/o embeddings) (in ms) | 8.70, 8.66, 8.68, 8.64, 8.68 | 8.64ms |
| vit_t | Box (w. embeddings) (in ms) | 8.84, 8.78, 8.75, 8.78, 8.67 | 8.67ms |
| vit_b | Box (w. embeddings) (in ms) | 10.07, 9.74, 9.76, 9.86, 9.69 | 9.69ms |
| vit_l | Box (w. embeddings) (in ms) | 12.05, 11.95, 11.79, 11.89, 13.09 | 11.79ms |
| vit_h | Box (w. embeddings) (in ms) | 15.41, 14.99, 14.86, 14.86, 14.84 | 14.84ms |

### CPU (medium partition - 64GB RAM)

| Model | Experiment           | Inference Time (for 5 runs)       | Chosen |
|-------|----------------------|-----------------------------------|--------|
| vit_t | Precompute embeddings | 0.29, 0.28, 0.28, 0.28, 0.27 | 0.27s |
| vit_b | Precompute embeddings | 1.48, 1.47, 1.46, 1.48, 1.48 | 1.46s |
| vit_l | Precompute embeddings | 3.67, 3.69, 3.67, 3.71, 3.67 | 3.67s |
| vit_h | Precompute embeddings | 5.07, 5.08, 5.18, 5.39, 5.15 | 5.07s |
| vit_t | AMG (w/o embeddings) | 18.91, 19.97, 18.82, 18.81, 18.86 | 18.81s |
| vit_b | AMG (w/o embeddings) | 19.00, 18.97, 19.01, 19.10, 19.99 | 18.97s |
| vit_l | AMG (w/o embeddings) | 18.80, 18.78, 19.01, 19.10, 19.99 | 18.78s |
| vit_h | AMG (w/o embeddings) | 18.92, 18.89, 18.90, 19.02, 20.02 | 18.89s |
| vit_t | AMG (w. embeddings)  | 19.13, 19.09, 19.14, 19.33, 20.28 | 19.09s |
| vit_b | AMG (w. embeddings)  | 20.39, 20.39, 20.38, 20.67, 21.60 | 20.38s |
| vit_l | AMG (w. embeddings)  | 22.39, 22.35, 22.47, 22.56, 22.59 | 22.35s |
| vit_h | AMG (w. embeddings)  | 24.09, 24.04, 24.15, 24.30, 23.55 | 23.55s |
| vit_t | AIS (w/o embeddings) | 1.25, 1.24, 1.23, 1.25, 1.29 | 1.23s |
| vit_b | AIS (w/o embeddings) | 1.24, 1.24, 1.24, 1.29, 1.25 | 1.24s |
| vit_l | AIS (w/o embeddings) | 1.23, 1.23, 1.23, 1.24, 1.29 | 1.23s |
| vit_h | AIS (w/o embeddings) | 1.23, 1.23, 1.22, 1.30, 1.24 | 1.22s |
| vit_t | AIS (w. embeddings)  | 1.47, 1.49, 1.49, 1.49, 1.57 | 1.47s |
| vit_b | AIS (w. embeddings)  | 2.68, 2.67, 2.68, 2.70, 2.83 | 2.67s |
| vit_l | AIS (w. embeddings)  | 4.84, 4.84, 4.83, 4.91, 5.02 | 4.83s |
| vit_h | AIS (w. embeddings)  | 6.30, 6.30, 6.30, 6.39, 6.61 | 6.30s |
| vit_t | Point (w/o embeddings) (in ms) | 28.51, 27.61, 28.29, 27.93, 28.14 | 27.61ms |
| vit_b | Point (w/o embeddings) (in ms) | 28.25, 28.34, 29.11, 28.87, 28.49 | 28.25ms |
| vit_l | Point (w/o embeddings) (in ms) | 28.09, 28.51, 28.83, 28.35, 28.05 | 28.05ms |
| vit_h | Point (w/o embeddings) (in ms) | 28.09, 28.87, 29.18, 28.76, 28.64 | 28.09ms |
| vit_t | Point (w. embeddings) (in ms) | 28.99, 28.94, 29.38, 30.20, 30.62 | 28.94ms |
| vit_b | Point (w. embeddings) (in ms) | 38,39, 37.92, 37.54, 39.88, 39.96| 37.54ms |
| vit_l | Point (w. embeddings) (in ms) | 52.73 ,55.87, 53.71, 53.23, 53.22 | 52.73ms |
| vit_h | Point (w. embeddings) (in ms) | 63.74, 63.73, 63.77, 67.44, 63.97 | 63.73ms |
| vit_t | Box (w/o embeddings) (in ms) | 25.97, 26.45, 26.42, 25.97, 25.83 | 25.97ms |
| vit_b | Box (w/o embeddings) (in ms) | 27.16, 26.72, 26.62, 26.40, 26.52 | 26.40ms |
| vit_l | Box (w/o embeddings) (in ms) | 26.87, 26.37, 28.56, 31.09, 28.09 | 26.37ms |
| vit_h | Box (w/o embeddings) (in ms) | 27.47, 27.66, 27.32, 27.26, 27.46 | 27.26ms |
| vit_t | Box (w. embeddings) (in ms) | 28.46, 28.02, 27.68, 27.73, 27.68 | 27.68ms |
| vit_b | Box (w. embeddings) (in ms) | 36.97, 36.43, 36.34, 39,45, 41.12 | 36.34ms |
| vit_l | Box (w. embeddings) (in ms) | 51.57, 51.68, 52.53, 54.81, 52.04 | 51.57ms |
| vit_h | Box (w. embeddings) (in ms) | 60.79, 61.69, 62.09, 65.46, 61.39 | 60.79ms |


> Setting: For AIS / AMG (ran for first 5 images per cell-type, and minimum of 5 runs taken), for interactive (ran for first 5 images per cell-type, time per prompt taken into account and minimum of 5 runs taken)