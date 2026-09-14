# Region semantic consistency -- results

Regions: cubic blocks of side s voxels (0.2 m) on the raw-LiDAR occupied grid, 26-connected components inside a block; s = {'small': 3, 'medium': 5, 'large': 9, 'xlarge': 15}. Groups: interleaved frame-disjoint A/B/C ({'A': [1, 4, 7, 10, 13, 16, 19], 'B': [2, 5, 8, 11, 14, 17, 20], 'C': [3, 6, 9, 12, 15, 18]}). Units carrying both A and B are scored; R1 = within-anchor shuffle stratified by observation count; R2 = local shuffle within (z bin 4 vox, xy cell 20 vox).

## eligible population (unobserved at t -> future-observed -> LiDAR-occupied)

**0006** -- 2,018,736 voxels in population

| representation | A&B units | coverage A&B (% pop.) | coverage any (%) | top-1 A-B [CI] | cosine | JS | A-C | B-C | 3-way | R0 prior | R1 shuffle | R2 local | true-R1 | true-R2 | classes >=200 | dominant share | entropy | median occ. voxels | median extent (m) | mean frames A+B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voxel | 18,051 | 0.89 | 30.3 | 0.561 [0.50,0.61] | 0.622 | 0.268 | 0.514 | 0.530 | 0.398 (2592) | 0.195 | 0.403 | 0.548 | +0.159 | +0.014 | 8 | 0.33 (vegetation) | 1.90 | 1 | 0.2 | 2.07 |
| small | 10,557 | 3.02 | 42.2 | 0.557 [0.51,0.61] | 0.619 | 0.268 | 0.528 | 0.530 | 0.399 (2223) | 0.195 | 0.390 | 0.542 | +0.167 | +0.015 | 8 | 0.34 (vegetation) | 1.92 | 6 | 0.6 | 2.11 |
| medium | 8,626 | 5.64 | 50.6 | 0.552 [0.50,0.60] | 0.616 | 0.269 | 0.527 | 0.524 | 0.387 (2231) | 0.196 | 0.399 | 0.536 | +0.153 | +0.016 | 8 | 0.34 (vegetation) | 1.93 | 14 | 1.0 | 2.13 |
| large | 7,103 | 11.82 | 62.2 | 0.539 [0.49,0.58] | 0.611 | 0.269 | 0.527 | 0.521 | 0.370 (2273) | 0.201 | 0.385 | 0.527 | +0.154 | +0.012 | 8 | 0.35 (vegetation) | 1.91 | 38 | 1.8 | 2.19 |
| xlarge | 6,079 | 21.66 | 72.9 | 0.525 [0.49,0.56] | 0.602 | 0.275 | 0.520 | 0.519 | 0.365 (2354) | 0.204 | 0.380 | 0.495 | +0.146 | +0.030 | 6 | 0.36 (vegetation) | 1.91 | 80 | 3.0 | 2.25 |

Matched (voxels with their own A&B evidence, n = 18,051): own top-1 0.561, region-inherited small 0.569, medium 0.573, large 0.572, xlarge 0.575; own cosine 0.622, region small 0.635, medium 0.643, large 0.654, xlarge 0.668

**train pooled** -- 4,204,467 voxels in population

| representation | A&B units | coverage A&B (% pop.) | coverage any (%) | top-1 A-B [CI] | cosine | JS | A-C | B-C | 3-way | R0 prior | R1 shuffle | R2 local | true-R1 | true-R2 | classes >=200 | dominant share | entropy | median occ. voxels | median extent (m) | mean frames A+B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voxel | 28,288 | 0.67 | 22.8 | 0.698 [0.60,0.76] | 0.726 | 0.195 | 0.700 | 0.703 | 0.675 (4880) | 0.256 | 0.613 | 0.697 | +0.085 | +0.001 | 10 | 0.38 (building) | 1.65 | 1 | 0.2 | 2.07 |
| small | 16,778 | 2.08 | 31.2 | 0.680 [0.58,0.75] | 0.712 | 0.203 | 0.683 | 0.690 | 0.663 (3855) | 0.242 | 0.583 | 0.680 | +0.097 | -0.000 | 7 | 0.35 (building) | 1.71 | 5 | 0.6 | 2.12 |
| medium | 13,222 | 3.71 | 37.4 | 0.672 [0.58,0.74] | 0.706 | 0.207 | 0.663 | 0.672 | 0.636 (3481) | 0.232 | 0.573 | 0.673 | +0.099 | -0.001 | 7 | 0.33 (building) | 1.76 | 11 | 1.0 | 2.16 |
| large | 10,521 | 7.34 | 46.3 | 0.658 [0.58,0.72] | 0.698 | 0.211 | 0.650 | 0.660 | 0.592 (3278) | 0.221 | 0.539 | 0.665 | +0.119 | -0.007 | 6 | 0.32 (vegetation) | 1.82 | 27 | 1.8 | 2.21 |
| xlarge | 8,668 | 12.79 | 55.4 | 0.647 [0.58,0.71] | 0.693 | 0.214 | 0.646 | 0.649 | 0.560 (2906) | 0.217 | 0.531 | 0.658 | +0.116 | -0.010 | 6 | 0.34 (vegetation) | 1.84 | 45 | 2.6 | 2.26 |

Matched (voxels with their own A&B evidence, n = 28,288): own top-1 0.698, region-inherited small 0.701, medium 0.702, large 0.706, xlarge 0.711; own cosine 0.726, region small 0.733, medium 0.738, large 0.749, xlarge 0.762

## all future-observed LiDAR-occupied voxels

**0006** -- 2,393,660 voxels in population

| representation | A&B units | coverage A&B (% pop.) | coverage any (%) | top-1 A-B [CI] | cosine | JS | A-C | B-C | 3-way | R0 prior | R1 shuffle | R2 local | true-R1 | true-R2 | classes >=200 | dominant share | entropy | median occ. voxels | median extent (m) | mean frames A+B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voxel | 23,778 | 0.99 | 30.4 | 0.561 [0.50,0.61] | 0.618 | 0.271 | 0.515 | 0.536 | 0.435 (3792) | 0.179 | 0.386 | 0.544 | +0.175 | +0.017 | 10 | 0.31 (vegetation) | 1.98 | 1 | 0.2 | 2.08 |
| small | 13,175 | 3.23 | 42.3 | 0.552 [0.50,0.60] | 0.615 | 0.271 | 0.523 | 0.530 | 0.404 (2799) | 0.181 | 0.371 | 0.535 | +0.180 | +0.017 | 8 | 0.32 (vegetation) | 1.99 | 6 | 0.6 | 2.11 |
| medium | 10,530 | 5.95 | 50.8 | 0.546 [0.50,0.59] | 0.611 | 0.273 | 0.523 | 0.527 | 0.394 (2709) | 0.183 | 0.374 | 0.524 | +0.172 | +0.022 | 8 | 0.33 (vegetation) | 1.99 | 13 | 1.0 | 2.14 |
| large | 8,491 | 12.40 | 62.5 | 0.537 [0.49,0.58] | 0.608 | 0.272 | 0.522 | 0.522 | 0.372 (2633) | 0.189 | 0.376 | 0.522 | +0.161 | +0.015 | 8 | 0.34 (vegetation) | 1.96 | 33 | 1.8 | 2.19 |
| xlarge | 7,128 | 22.32 | 73.3 | 0.518 [0.48,0.56] | 0.596 | 0.279 | 0.512 | 0.518 | 0.355 (2665) | 0.191 | 0.367 | 0.490 | +0.152 | +0.028 | 8 | 0.34 (vegetation) | 1.97 | 65 | 3.0 | 2.25 |

Matched (voxels with their own A&B evidence, n = 23,778): own top-1 0.561, region-inherited small 0.564, medium 0.571, large 0.569, xlarge 0.569; own cosine 0.618, region small 0.632, medium 0.640, large 0.653, xlarge 0.667

**train pooled** -- 4,945,659 voxels in population

| representation | A&B units | coverage A&B (% pop.) | coverage any (%) | top-1 A-B [CI] | cosine | JS | A-C | B-C | 3-way | R0 prior | R1 shuffle | R2 local | true-R1 | true-R2 | classes >=200 | dominant share | entropy | median occ. voxels | median extent (m) | mean frames A+B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voxel | 34,417 | 0.70 | 23.0 | 0.682 [0.58,0.75] | 0.713 | 0.203 | 0.682 | 0.681 | 0.650 (5428) | 0.241 | 0.585 | 0.680 | +0.097 | +0.003 | 11 | 0.38 (building) | 1.71 | 1 | 0.2 | 2.06 |
| small | 19,745 | 2.18 | 31.7 | 0.665 [0.56,0.74] | 0.699 | 0.212 | 0.661 | 0.668 | 0.642 (4283) | 0.227 | 0.557 | 0.662 | +0.107 | +0.002 | 10 | 0.34 (building) | 1.77 | 5 | 0.6 | 2.12 |
| medium | 15,458 | 3.91 | 38.0 | 0.656 [0.56,0.73] | 0.694 | 0.215 | 0.642 | 0.655 | 0.614 (3885) | 0.218 | 0.543 | 0.653 | +0.113 | +0.002 | 8 | 0.31 (building) | 1.81 | 11 | 1.0 | 2.15 |
| large | 12,101 | 7.72 | 47.1 | 0.641 [0.56,0.71] | 0.686 | 0.219 | 0.632 | 0.644 | 0.573 (3628) | 0.209 | 0.519 | 0.647 | +0.122 | -0.006 | 7 | 0.30 (vegetation) | 1.86 | 24 | 1.8 | 2.21 |
| xlarge | 9,962 | 13.45 | 56.2 | 0.634 [0.56,0.70] | 0.684 | 0.220 | 0.629 | 0.637 | 0.542 (3246) | 0.205 | 0.506 | 0.638 | +0.128 | -0.004 | 7 | 0.31 (vegetation) | 1.88 | 39 | 2.4 | 2.26 |

Matched (voxels with their own A&B evidence, n = 34,417): own top-1 0.682, region-inherited small 0.686, medium 0.688, large 0.691, xlarge 0.695; own cosine 0.713, region small 0.721, medium 0.726, large 0.737, xlarge 0.749

## Repeatability by number of contributing frames (A+B), drive 0006, eligible population

| representation | 2 | 3 | 4-5 | >=6 |
|---|---|---|---|---|
| voxel | 0.565 (n=16945) | 0.501 (n=1019) | 0.517 (n=87) | - |
| small | 0.563 (n=9590) | 0.508 (n=847) | 0.422 (n=116) | 0.250 (n=4) |
| medium | 0.559 (n=7642) | 0.511 (n=846) | 0.418 (n=134) | 0.500 (n=4) |
| large | 0.546 (n=6028) | 0.517 (n=871) | 0.448 (n=192) | 0.083 (n=12) |
| xlarge | 0.530 (n=4873) | 0.519 (n=945) | 0.478 (n=247) | 0.143 (n=14) |

## Repeatability by observations pooled (A+B), drive 0006

| representation | 2 | 3-4 | 5-8 | 9-16 | 17-32 | >32 |
|---|---|---|---|---|---|---|
| voxel | 0.565 (n=16945) | 0.504 (n=1100) | 0.333 (n=6) | - | - | - |
| small | 0.537 (n=1989) | 0.547 (n=2529) | 0.554 (n=3335) | 0.582 (n=2252) | 0.591 (n=440) | 0.750 (n=12) |
| medium | 0.514 (n=1163) | 0.528 (n=1416) | 0.540 (n=1977) | 0.564 (n=2209) | 0.599 (n=1526) | 0.570 (n=335) |
| large | 0.499 (n=745) | 0.526 (n=909) | 0.506 (n=1161) | 0.540 (n=1392) | 0.547 (n=1516) | 0.587 (n=1380) |
| xlarge | 0.502 (n=562) | 0.482 (n=660) | 0.480 (n=785) | 0.526 (n=1002) | 0.541 (n=1115) | 0.556 (n=1955) |

## Per drive (A-B top-1, R1 in brackets)

| representation | 0003 | 0007 | 0010 | 0006 |
|---|---|---|---|---|
| voxel | 0.902 [0.879] n=2921 | 0.492 [0.486] n=6124 | 0.732 [0.613] n=19243 | 0.561 [0.403] n=18051 |
| small | 0.878 [0.835] n=1760 | 0.485 [0.471] n=4102 | 0.721 [0.584] n=10916 | 0.557 [0.390] n=10557 |
| medium | 0.874 [0.836] n=1467 | 0.494 [0.472] n=3297 | 0.706 [0.567] n=8458 | 0.552 [0.399] n=8626 |
| large | 0.841 [0.784] n=1147 | 0.506 [0.468] n=2655 | 0.686 [0.526] n=6719 | 0.539 [0.385] n=7103 |
| xlarge | 0.830 [0.781] n=992 | 0.521 [0.471] n=2130 | 0.663 [0.511] n=5546 | 0.525 [0.380] n=6079 |

## Per class (A-B top-1; class = argmax of the A vector; classes with >= 200 units at voxel level)

**0006**

| class | voxel | small | medium | large | xlarge |
|---|---|---|---|---|---|
| vegetation | 0.729 (n=5934) | 0.718 (n=3612) | 0.711 (n=2963) | 0.699 (n=2514) | 0.692 (n=2185) |
| man-made structure | 0.557 (n=3928) | 0.556 (n=2091) | 0.561 (n=1702) | 0.534 (n=1412) | 0.506 (n=1189) |
| building | 0.478 (n=2852) | 0.490 (n=1627) | 0.508 (n=1304) | 0.485 (n=1098) | 0.496 (n=937) |
| road | 0.632 (n=2075) | 0.628 (n=1168) | 0.610 (n=917) | 0.575 (n=680) | 0.550 (n=551) |
| car | 0.462 (n=836) | 0.521 (n=512) | 0.525 (n=427) | 0.493 (n=351) | 0.417 (n=312) |
| sidewalk | 0.097 (n=649) | 0.134 (n=411) | 0.135 (n=347) | 0.129 (n=263) | <200 |
| other ground | 0.278 (n=632) | 0.224 (n=437) | 0.220 (n=368) | 0.245 (n=290) | 0.230 (n=261) |
| fence | 0.317 (n=504) | 0.308 (n=305) | 0.255 (n=255) | 0.303 (n=211) | <200 |

**train pooled**

| class | voxel | small | medium | large | xlarge |
|---|---|---|---|---|---|
| building | 0.863 (n=10837) | 0.866 (n=5902) | 0.857 (n=4319) | 0.853 (n=3142) | 0.826 (n=2429) |
| vegetation | 0.745 (n=8790) | 0.740 (n=5292) | 0.740 (n=4224) | 0.740 (n=3418) | 0.747 (n=2925) |
| road | 0.652 (n=4127) | 0.606 (n=2575) | 0.610 (n=2115) | 0.607 (n=1708) | 0.587 (n=1398) |
| other ground | 0.201 (n=1240) | 0.187 (n=815) | 0.207 (n=658) | 0.175 (n=525) | 0.203 (n=413) |
| car | 0.452 (n=778) | 0.465 (n=600) | 0.499 (n=525) | 0.481 (n=478) | 0.484 (n=409) |
| man-made structure | 0.257 (n=674) | 0.346 (n=405) | 0.365 (n=359) | 0.380 (n=324) | 0.366 (n=276) |
| parking | 0.127 (n=371) | 0.138 (n=268) | 0.117 (n=223) | <200 | <200 |
| sidewalk | 0.157 (n=357) | <200 | <200 | <200 | <200 |
| pole | 0.081 (n=308) | <200 | <200 | <200 | <200 |
| other vehicle | 0.242 (n=248) | <200 | <200 | <200 | <200 |

## Observed semantic quality against benchmark labels

Not measured: the gate's protocol never reads a KITTI-360 human semantic label on the source drives (the SSCBench-KITTI-360 label was shown to contradict its own LiDAR in Gate 8C-0), and SemanticKITTI / Occ3D are untouched in this step by constraint.

Verification: 10/10 checks pass.

