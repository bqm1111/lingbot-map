# Local registration audit -- result tables

## 0006: relative pose and reprojection (five-frame clips)

GT motion anchor<-frame: translation median 10.03 m, rotation median 3.11 deg

| offset | S trans err m | L trans err m | S rel | L rel | S dir deg | L dir deg | S rot deg | L rot deg | S reproj px med/p90 | L reproj px med/p90 | G reproj px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -1 | 3.64 | 0.58 | 0.837 | 0.133 | 31.8 | 6.0 | 1.49 | 0.47 | 123.2 / 465 | 32.2 / 71 | 0.00 |
| -2 | 4.62 | 1.12 | 0.545 | 0.126 | 18.9 | 5.9 | 1.94 | 0.66 | 123.9 / 446 | 47.8 / 116 | 0.00 |
| -3 | 6.09 | 1.65 | 0.468 | 0.123 | 16.1 | 5.9 | 2.39 | 0.86 | 130.5 / 479 | 58.4 / 148 | 0.00 |
| -4 | 7.46 | 2.20 | 0.432 | 0.124 | 13.0 | 5.9 | 2.53 | 1.00 | 144.5 / 512 | 69.3 / 177 | 0.00 |
| all | 5.23 | 1.32 | 0.530 | 0.126 | 18.6 | 5.9 | 2.03 | 0.72 | 128.5 / 470 | 46.5 / 124 | 0.00 |

Endpoint error after mapping into the anchor camera (median m): pose only S / L / G = 4.66 / 1.08 / 0.00; LingBot depth + pose: S 5.26, L 1.56, Lp 2.36, G 2.40, GLd 1.48

| stream position | n pairs | S reproj px | L reproj px | S rel trans | L rel trans | S rot deg | L rot deg |
|---|---|---|---|---|---|---|---|
| early (0-299) | 396 | 84.3 | 44.3 | 0.305 | 0.120 | 1.86 | 0.89 |
| middle (300-899) | 800 | 115.5 | 46.0 | 0.453 | 0.124 | 1.97 | 0.72 |
| late (>=900) | 1164 | 164.8 | 47.6 | 0.674 | 0.131 | 2.13 | 0.65 |

## 0003: relative pose and reprojection (five-frame clips)

GT motion anchor<-frame: translation median 16.32 m, rotation median 2.22 deg

| offset | S trans err m | L trans err m | S rel | L rel | S dir deg | L dir deg | S rot deg | L rot deg | S reproj px med/p90 | L reproj px med/p90 | G reproj px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -1 | 1.65 | 1.09 | 0.257 | 0.157 | 8.7 | 6.0 | 0.95 | 0.33 | 57.1 / 257 | 40.4 / 100 | 0.00 |
| -2 | 3.82 | 2.21 | 0.248 | 0.158 | 7.8 | 5.9 | 1.10 | 0.49 | 76.3 / 324 | 54.4 / 140 | 0.00 |
| -3 | 4.61 | 3.25 | 0.209 | 0.153 | 7.5 | 5.9 | 1.63 | 0.74 | 77.1 / 254 | 62.4 / 148 | 0.00 |
| -4 | 6.37 | 4.27 | 0.199 | 0.147 | 7.4 | 5.9 | 1.77 | 0.97 | 69.0 / 341 | 60.8 / 164 | 0.00 |
| all | 4.00 | 2.36 | 0.212 | 0.155 | 7.5 | 5.9 | 1.21 | 0.56 | 67.7 / 287 | 49.9 / 126 | 0.00 |

Endpoint error after mapping into the anchor camera (median m): pose only S / L / G = 2.26 / 1.54 / 0.00; LingBot depth + pose: S 3.81, L 2.16, Lp 2.55, G 2.49, GLd 2.51

| stream position | n pairs | S reproj px | L reproj px | S rel trans | L rel trans | S rot deg | L rot deg |
|---|---|---|---|---|---|---|---|
| early (0-299) | 260 | 67.7 | 49.9 | 0.212 | 0.155 | 1.21 | 0.56 |

## 0007: relative pose and reprojection (five-frame clips)

GT motion anchor<-frame: translation median 20.43 m, rotation median 0.95 deg

| offset | S trans err m | L trans err m | S rel | L rel | S dir deg | L dir deg | S rot deg | L rot deg | S reproj px med/p90 | L reproj px med/p90 | G reproj px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -1 | 10.41 | 1.61 | 1.287 | 0.179 | 41.2 | 5.9 | 1.23 | 0.28 | 219.9 / 1013 | 46.5 / 125 | 0.00 |
| -2 | 11.41 | 3.27 | 0.680 | 0.174 | 22.4 | 5.8 | 1.25 | 0.43 | 228.1 / 1019 | 53.6 / 163 | 0.00 |
| -3 | 14.87 | 4.98 | 0.617 | 0.172 | 21.8 | 5.9 | 1.57 | 0.55 | 354.6 / 2277 | 51.5 / 172 | 0.00 |
| -4 | 16.79 | 6.53 | 0.505 | 0.170 | 18.2 | 5.9 | 1.65 | 0.67 | 542.3 / 2653 | 52.7 / 140 | 0.00 |
| all | 13.17 | 3.25 | 0.709 | 0.173 | 23.9 | 5.9 | 1.34 | 0.45 | 253.6 / 1287 | 49.7 / 140 | 0.00 |

Endpoint error after mapping into the anchor camera (median m): pose only S / L / G = 12.46 / 1.70 / 0.00; LingBot depth + pose: S 13.33, L 2.33, Lp 2.68, G 2.37, GLd 2.87

| stream position | n pairs | S reproj px | L reproj px | S rel trans | L rel trans | S rot deg | L rot deg |
|---|---|---|---|---|---|---|---|
| early (0-299) | 396 | 143.9 | 69.8 | 0.488 | 0.191 | 1.00 | 0.50 |
| middle (300-899) | 364 | 378.7 | 39.6 | 1.113 | 0.149 | 2.07 | 0.42 |

## 0010: relative pose and reprojection (five-frame clips)

GT motion anchor<-frame: translation median 12.65 m, rotation median 1.21 deg

| offset | S trans err m | L trans err m | S rel | L rel | S dir deg | L dir deg | S rot deg | L rot deg | S reproj px med/p90 | L reproj px med/p90 | G reproj px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -1 | 4.45 | 0.91 | 0.830 | 0.163 | 24.3 | 5.8 | 0.96 | 0.30 | 129.5 / 751 | 34.5 / 86 | 0.00 |
| -2 | 5.90 | 1.81 | 0.505 | 0.158 | 14.7 | 5.8 | 1.26 | 0.42 | 129.4 / 664 | 46.5 / 119 | 0.00 |
| -3 | 7.64 | 2.68 | 0.441 | 0.157 | 15.0 | 5.8 | 1.39 | 0.52 | 159.2 / 573 | 57.6 / 140 | 0.00 |
| -4 | 8.27 | 3.56 | 0.381 | 0.155 | 12.1 | 5.8 | 1.57 | 0.60 | 171.0 / 704 | 67.3 / 157 | 0.00 |
| all | 6.75 | 1.97 | 0.480 | 0.157 | 16.0 | 5.8 | 1.25 | 0.45 | 142.2 / 671 | 47.1 / 118 | 0.00 |

Endpoint error after mapping into the anchor camera (median m): pose only S / L / G = 6.02 / 1.49 / 0.00; LingBot depth + pose: S 7.33, L 2.22, Lp 2.60, G 2.95, GLd 2.73

| stream position | n pairs | S reproj px | L reproj px | S rel trans | L rel trans | S rot deg | L rot deg |
|---|---|---|---|---|---|---|---|
| early (0-299) | 396 | 87.1 | 48.5 | 0.301 | 0.156 | 1.12 | 0.54 |
| middle (300-899) | 400 | 245.2 | 45.6 | 0.879 | 0.160 | 1.50 | 0.38 |

## Depth (anchor frame, median over clips)

| drive | AbsRel S vs LiDAR | AbsRel L vs LiDAR | AbsRel S vs L | corr S,L | L/S canonical ratio | L/S scale | S metric/LiDAR | L metric/LiDAR |
|---|---|---|---|---|---|---|---|---|
| 0006 | 0.123 | 0.081 | 0.202 | 0.991 | 0.805 | 1.317 | 0.885 | 0.934 |
| 0003 | 0.098 | 0.100 | 0.267 | 0.990 | 0.735 | 1.335 | 0.907 | 0.903 |
| 0007 | 0.087 | 0.129 | 0.172 | 0.991 | 0.863 | 1.095 | 0.920 | 0.876 |
| 0010 | 0.142 | 0.124 | 0.192 | 0.990 | 0.823 | 1.226 | 0.862 | 0.880 |

## 0006: five-frame maps (590 clips; anchor-relative S vs stream-frame S Jaccard median 0.664; LiDAR/S depth ratio median 1.130)

| condition | raw vs past-5 LiDAR IoU / P / R | dilated vs past-5 IoU | raw vs future target IoU | dilated vs target IoU / P / R | occupied on LiDAR-free | conflicts / clip | multi-view NN m (-4,-3,-2,-1) | frozen head vs target IoU / P / R |
|---|---|---|---|---|---|---|---|---|
| S | 0.054 / 0.142 / 0.080 | 0.082 | 0.025 | 0.052 / 0.090 / 0.108 | 0.320 | 8222 | 1.17 / 1.09 / 1.04 / 0.93 | 0.109 / 0.146 / 0.300 |
| L | 0.090 / 0.187 / 0.146 | 0.109 | 0.040 | 0.061 / 0.087 / 0.173 | 0.223 | 11800 | 0.69 / 0.63 / 0.52 / 0.33 | 0.110 / 0.139 / 0.349 |
| Lp | 0.068 / 0.198 / 0.093 | 0.101 | 0.027 | 0.057 / 0.114 / 0.101 | 0.317 | 6027 | 0.41 / 0.28 / 0.29 / 0.33 | 0.116 / 0.155 / 0.319 |
| G | 0.061 / 0.211 / 0.080 | 0.107 | 0.025 | 0.058 / 0.130 / 0.096 | 0.305 | 4755 | 1.30 / 1.31 / 0.90 / 0.49 | 0.118 / 0.160 / 0.310 |
| GLd | 0.079 / 0.165 / 0.131 | 0.107 | 0.038 | 0.062 / 0.088 / 0.171 | 0.226 | 12490 | 1.50 / 1.24 / 0.92 / 0.50 | 0.109 / 0.139 / 0.337 |
| Gs (oracle) | 0.072 / 0.291 / 0.088 | 0.133 | 0.031 | 0.071 / 0.159 / 0.113 | 0.192 | 4840 | 1.68 / 1.38 / 0.96 / 0.57 | 0.120 / 0.165 / 0.309 |

| condition | early (0-299) | middle (300-899) | late (>=900) |
|---|---|---|---|
| S raw past-5 IoU (median per clip) | 0.063 | 0.053 | 0.048 |
| L raw past-5 IoU (median per clip) | 0.083 | 0.092 | 0.088 |
| Lp raw past-5 IoU (median per clip) | 0.059 | 0.065 | 0.066 |
| G raw past-5 IoU (median per clip) | 0.063 | 0.057 | 0.056 |
| GLd raw past-5 IoU (median per clip) | 0.086 | 0.082 | 0.075 |
| Gs raw past-5 IoU (median per clip) | 0.076 | 0.074 | 0.066 |

## 0003: five-frame maps (65 clips; anchor-relative S vs stream-frame S Jaccard median 0.650; LiDAR/S depth ratio median 1.102)

| condition | raw vs past-5 LiDAR IoU / P / R | dilated vs past-5 IoU | raw vs future target IoU | dilated vs target IoU / P / R | occupied on LiDAR-free | conflicts / clip | multi-view NN m (-4,-3,-2,-1) | frozen head vs target IoU / P / R |
|---|---|---|---|---|---|---|---|---|
| S | 0.046 / 0.147 / 0.063 | 0.071 | 0.024 | 0.050 / 0.089 / 0.103 | 0.331 | 6431 | 0.73 / 0.65 / 0.62 / 0.82 | 0.100 / 0.119 / 0.384 |
| L | 0.043 / 0.087 / 0.078 | 0.061 | 0.025 | 0.042 / 0.056 / 0.144 | 0.325 | 12163 | 0.87 / 0.81 / 0.92 / 0.60 | 0.088 / 0.099 / 0.445 |
| Lp | 0.052 / 0.180 / 0.068 | 0.079 | 0.026 | 0.053 / 0.099 / 0.101 | 0.293 | 5827 | 0.26 / 0.21 / 0.30 / 0.51 | 0.101 / 0.119 / 0.399 |
| G | 0.035 / 0.169 / 0.042 | 0.074 | 0.017 | 0.047 / 0.105 / 0.078 | 0.288 | 3413 | 2.20 / 2.32 / 1.58 / 0.90 | 0.106 / 0.127 / 0.385 |
| GLd | 0.035 / 0.076 / 0.062 | 0.057 | 0.021 | 0.039 / 0.053 / 0.128 | 0.326 | 10719 | 2.55 / 2.10 / 1.64 / 0.86 | 0.089 / 0.101 / 0.436 |
| Gs (oracle) | 0.037 / 0.213 / 0.043 | 0.087 | 0.021 | 0.055 / 0.125 / 0.089 | 0.204 | 3094 | 2.93 / 2.34 / 1.64 / 0.98 | 0.104 / 0.126 / 0.373 |

| condition | early (0-299) |
|---|---|
| S raw past-5 IoU (median per clip) | 0.039 |
| L raw past-5 IoU (median per clip) | 0.041 |
| Lp raw past-5 IoU (median per clip) | 0.051 |
| G raw past-5 IoU (median per clip) | 0.024 |
| GLd raw past-5 IoU (median per clip) | 0.034 |
| Gs raw past-5 IoU (median per clip) | 0.030 |

## 0007: five-frame maps (190 clips; anchor-relative S vs stream-frame S Jaccard median 0.624; LiDAR/S depth ratio median 1.087)

| condition | raw vs past-5 LiDAR IoU / P / R | dilated vs past-5 IoU | raw vs future target IoU | dilated vs target IoU / P / R | occupied on LiDAR-free | conflicts / clip | multi-view NN m (-4,-3,-2,-1) | frozen head vs target IoU / P / R |
|---|---|---|---|---|---|---|---|---|
| S | 0.033 / 0.086 / 0.051 | 0.053 | 0.022 | 0.044 / 0.064 / 0.125 | 0.377 | 7008 | 3.25 / 2.79 / 2.36 / 2.26 | 0.109 / 0.126 / 0.443 |
| L | 0.049 / 0.099 / 0.089 | 0.069 | 0.031 | 0.053 / 0.072 / 0.162 | 0.349 | 10870 | 1.25 / 1.08 / 0.98 / 0.64 | 0.109 / 0.121 / 0.529 |
| Lp | 0.042 / 0.101 / 0.067 | 0.069 | 0.027 | 0.056 / 0.091 / 0.128 | 0.439 | 7548 | 0.56 / 0.35 / 0.50 / 0.64 | 0.115 / 0.131 / 0.493 |
| G | 0.042 / 0.133 / 0.058 | 0.074 | 0.025 | 0.059 / 0.109 / 0.112 | 0.386 | 5142 | 2.78 / 2.82 / 2.08 / 1.07 | 0.120 / 0.138 / 0.489 |
| GLd | 0.043 / 0.098 / 0.072 | 0.067 | 0.028 | 0.051 / 0.072 / 0.148 | 0.335 | 8639 | 3.05 / 2.49 / 1.80 / 1.02 | 0.111 / 0.124 / 0.522 |
| Gs (oracle) | 0.046 / 0.155 / 0.061 | 0.086 | 0.027 | 0.066 / 0.123 / 0.123 | 0.315 | 5058 | 2.90 / 3.07 / 2.12 / 1.12 | 0.121 / 0.139 / 0.479 |

| condition | early (0-299) | middle (300-899) |
|---|---|---|
| S raw past-5 IoU (median per clip) | 0.031 | 0.026 |
| L raw past-5 IoU (median per clip) | 0.039 | 0.056 |
| Lp raw past-5 IoU (median per clip) | 0.040 | 0.036 |
| G raw past-5 IoU (median per clip) | 0.029 | 0.044 |
| GLd raw past-5 IoU (median per clip) | 0.033 | 0.039 |
| Gs raw past-5 IoU (median per clip) | 0.034 | 0.045 |

## 0010: five-frame maps (199 clips; anchor-relative S vs stream-frame S Jaccard median 0.638; LiDAR/S depth ratio median 1.160)

| condition | raw vs past-5 LiDAR IoU / P / R | dilated vs past-5 IoU | raw vs future target IoU | dilated vs target IoU / P / R | occupied on LiDAR-free | conflicts / clip | multi-view NN m (-4,-3,-2,-1) | frozen head vs target IoU / P / R |
|---|---|---|---|---|---|---|---|---|
| S | 0.040 / 0.087 / 0.069 | 0.056 | 0.024 | 0.041 / 0.053 / 0.152 | 0.312 | 10815 | 2.43 / 2.27 / 1.71 / 1.36 | 0.098 / 0.115 / 0.390 |
| L | 0.059 / 0.109 / 0.115 | 0.068 | 0.034 | 0.046 / 0.056 / 0.208 | 0.272 | 14038 | 2.42 / 2.06 / 1.53 / 0.59 | 0.094 / 0.107 / 0.427 |
| Lp | 0.051 / 0.113 / 0.086 | 0.068 | 0.029 | 0.047 / 0.064 / 0.148 | 0.316 | 9928 | 0.93 / 0.67 / 0.87 / 0.61 | 0.101 / 0.118 / 0.410 |
| G | 0.044 / 0.111 / 0.068 | 0.071 | 0.026 | 0.048 / 0.068 / 0.136 | 0.298 | 7727 | 2.44 / 1.95 / 1.34 / 0.73 | 0.105 / 0.125 / 0.401 |
| GLd | 0.049 / 0.092 / 0.094 | 0.067 | 0.029 | 0.043 / 0.053 / 0.196 | 0.260 | 13948 | 2.67 / 2.13 / 1.58 / 0.80 | 0.094 / 0.108 / 0.413 |
| Gs (oracle) | 0.048 / 0.157 / 0.065 | 0.087 | 0.029 | 0.054 / 0.079 / 0.148 | 0.167 | 7567 | 2.63 / 2.02 / 1.48 / 0.87 | 0.107 / 0.129 / 0.387 |

| condition | early (0-299) | middle (300-899) |
|---|---|---|
| S raw past-5 IoU (median per clip) | 0.043 | 0.032 |
| L raw past-5 IoU (median per clip) | 0.057 | 0.065 |
| Lp raw past-5 IoU (median per clip) | 0.043 | 0.053 |
| G raw past-5 IoU (median per clip) | 0.040 | 0.046 |
| GLd raw past-5 IoU (median per clip) | 0.042 | 0.053 |
| Gs raw past-5 IoU (median per clip) | 0.038 | 0.049 |

## Semantic sanity check (0006, verified same-point pairs inside clips)

| condition | landmarks | two-frame same-voxel support | class agreement | local-control agreement | feature cosine | local-control cosine | cosine margin |
|---|---|---|---|---|---|---|---|
| S | 140,505 | 0.008 | 0.516 | 0.489 | 0.903 | 0.901 | +0.002 |
| L | 140,505 | 0.066 | 0.711 | 0.621 | 0.937 | 0.918 | +0.019 |
| Lp | 140,505 | 0.027 | 0.749 | 0.693 | 0.946 | 0.928 | +0.019 |
| G | 140,505 | 0.011 | 0.732 | 0.678 | 0.939 | 0.924 | +0.014 |
| GLd | 140,505 | 0.024 | 0.690 | 0.631 | 0.933 | 0.921 | +0.012 |
