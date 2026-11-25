# Module 3 — Customer Behavioural Segmentation Report

Clustering was performed on 30,000 customer snapshots using 8 standardized behavioral features (log-transformed `orders_last_180d` and `avg_order_value`), strictly excluding `no_purchase_next_90d` so post-hoc risk profiling remains leak-free.

## Algorithm Comparison
| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Noise % |
| --- | --- | --- | --- | --- | --- |
| K-Means++ (k=4) | 4 | 0.1083 | 2.1823 | 3786.5 | 0.00% |
| Hierarchical Ward (k=4) | 4 | 0.1309 | 2.0601 | 3732.1 | 0.00% |
| DBSCAN (eps=1.8, min_s=30) | 1 | -1.0000 | N/A | N/A | 7.17% |

## K-Means++ Elbow & Validity Sweep (k=2..7)
| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| --- | --- | --- | --- | --- |
| 2 | 201524.0 | 0.1538 | 2.2012 | 5727.4 |
| 3 | 184474.3 | 0.1367 | 2.1831 | 4514.5 |
| 4 | 174077.2 | 0.1083 | 2.1823 | 3786.5 |
| 5 | 164804.2 | 0.1093 | 2.0012 | 3421.5 |
| 6 | 157051.7 | 0.1064 | 2.1310 | 3168.3 |
| 7 | 150985.2 | 0.1018 | 2.0085 | 2947.8 |

## K-Means++ (k=4) Cluster Personas & Post-Hoc 90-Day Non-Purchase Risk
| Cluster Persona | Customers | Share | Tenure (mo) | Orders 180d | Avg AOV | Returns 180d | Post-Hoc 90d Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0: New / Low-Order High-Risk | 9221 | 30.7% | 11.3 | 0.94 | $81.35 | 0.24 | 49.9% |
| C1: Tenured Loyalists (Zero Returns) | 7037 | 23.5% | 40.6 | 5.39 | $83.80 | 0.01 | 28.1% |
| C2: High-Return Active Buyers | 6640 | 22.1% | 37.1 | 4.82 | $82.98 | 1.23 | 33.0% |
| C3: Core Engaged Repeat Buyers | 7102 | 23.7% | 39.6 | 5.19 | $82.16 | 0.0 | 23.4% |
