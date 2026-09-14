# Direct semantic-correspondence audit -- results

## Correspondence construction and rejection

| drive | source frames | landmarks | candidate (X, B) | behind / out of range | outside image | no LiDAR reference | occluded | in front of reference | accepted loose | accepted strict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0003 | 21 | 31,500 | 418,500 | 253,406 | 16,401 | 2,190 | 32,424 | 15,280 | 98,799 | 71,119 |
| 0007 | 58 | 87,000 | 1,203,000 | 805,671 | 26,751 | 10,393 | 121,957 | 30,155 | 208,073 | 148,509 |
| 0010 | 61 | 91,500 | 1,264,500 | 705,969 | 49,563 | 16,937 | 149,620 | 34,887 | 307,524 | 241,956 |
| 0006 | 356 | 534,000 | 7,446,000 | 3,774,544 | 300,526 | 108,241 | 1,148,706 | 131,100 | 1,982,883 | 1,597,989 |

## Same-point similarity and controls (strict visibility)

| drive | pairs | feature cos same | C0 random | C1 same frame | C3 same class | C2 local (0.3-1.5 m) | C2 same-view local | Δ vs C2 | AUC vs C2 | AUC vs C3 | AUC vs C0 | prob cos same / C2 | prob AUC vs C2 | top-1 same / C2 / same-view local | refined-label top-1 same |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0003 | 71,104 | 0.950 | 0.856 | 0.879 | 0.921 | 0.943 | 0.981 | +0.006 | 0.547 | 0.709 | 0.904 | 0.892 / 0.872 | 0.525 | 0.846 / 0.825 / 0.878 | 0.889 |
| 0007 | 148,443 | 0.949 | 0.846 | 0.864 | 0.918 | 0.940 | 0.978 | +0.008 | 0.555 | 0.715 | 0.903 | 0.886 / 0.863 | 0.527 | 0.794 / 0.774 / 0.842 | 0.831 |
| 0010 | 241,900 | 0.949 | 0.853 | 0.871 | 0.929 | 0.942 | 0.981 | +0.006 | 0.551 | 0.683 | 0.901 | 0.849 / 0.821 | 0.538 | 0.680 / 0.647 / 0.785 | 0.728 |
| 0006 | 1,597,606 | 0.942 | 0.841 | 0.855 | 0.918 | 0.932 | 0.974 | +0.009 | 0.561 | 0.689 | 0.897 | 0.877 / 0.833 | 0.553 | 0.729 / 0.675 / 0.783 | 0.740 |

Previous mapper-defined same-voxel pairs (0006): feature cos 0.918 vs local 0.915 (AUC 0.52); prob cos 0.762 vs 0.748 (AUC 0.514); top-1 0.568 vs 0.551.

## Viewpoint dependence, drive 0006 (strict)

**baseline**

| bin | n | feature cos same | C2 local | AUC vs C2 | prob cos same | top-1 same | top-1 C2 |
|---|---|---|---|---|---|---|---|
| <5 m | 457,130 | 0.970 | 0.955 | 0.618 | 0.940 | 0.814 | 0.729 |
| 5-15 m | 777,559 | 0.945 | 0.936 | 0.560 | 0.884 | 0.733 | 0.680 |
| >15 m | 362,917 | 0.900 | 0.897 | 0.518 | 0.783 | 0.615 | 0.597 |

**viewing-angle change**

| bin | n | feature cos same | C2 local | AUC vs C2 | prob cos same | top-1 same | top-1 C2 |
|---|---|---|---|---|---|---|---|
| <10 deg | 720,052 | 0.964 | 0.954 | 0.589 | 0.920 | 0.798 | 0.738 |
| 10-30 deg | 805,914 | 0.928 | 0.920 | 0.559 | 0.853 | 0.690 | 0.640 |
| >30 deg | 71,640 | 0.870 | 0.866 | 0.522 | 0.719 | 0.473 | 0.453 |

**depth in B**

| bin | n | feature cos same | C2 local | AUC vs C2 | prob cos same | top-1 same | top-1 C2 |
|---|---|---|---|---|---|---|---|
| 3-10 m | 320,030 | 0.956 | 0.942 | 0.606 | 0.919 | 0.777 | 0.702 |
| 10-20 m | 726,196 | 0.951 | 0.940 | 0.574 | 0.897 | 0.757 | 0.692 |
| 20-35 m | 423,700 | 0.926 | 0.920 | 0.526 | 0.842 | 0.685 | 0.653 |
| 35-50 m | 127,680 | 0.907 | 0.901 | 0.508 | 0.772 | 0.599 | 0.578 |

**visibility residual**

| bin | n | feature cos same | C2 local | AUC vs C2 | prob cos same | top-1 same | top-1 C2 |
|---|---|---|---|---|---|---|---|
| <0.1 m | 588,019 | 0.945 | 0.935 | 0.573 | 0.885 | 0.723 | 0.664 |
| 0.1-0.3 m | 679,968 | 0.943 | 0.934 | 0.562 | 0.881 | 0.739 | 0.684 |
| 0.3 m-loose | 384,820 | 0.927 | 0.920 | 0.536 | 0.855 | 0.732 | 0.692 |

**stream offset B-A**

| bin | n | feature cos same | C2 local | AUC vs C2 | prob cos same | top-1 same | top-1 C2 |
|---|---|---|---|---|---|---|---|
| -12 | 10,555 | 0.880 | 0.879 | 0.510 | 0.734 | 0.537 | 0.523 |
| -8 | 58,976 | 0.882 | 0.880 | 0.511 | 0.734 | 0.563 | 0.550 |
| -6 | 110,939 | 0.897 | 0.894 | 0.516 | 0.782 | 0.611 | 0.594 |
| -4 | 180,572 | 0.918 | 0.913 | 0.530 | 0.832 | 0.676 | 0.645 |
| -3 | 235,689 | 0.930 | 0.923 | 0.542 | 0.857 | 0.713 | 0.672 |
| -2 | 311,765 | 0.945 | 0.936 | 0.563 | 0.885 | 0.744 | 0.688 |
| -1 | 394,469 | 0.967 | 0.953 | 0.610 | 0.929 | 0.798 | 0.717 |
| 1 | 161,777 | 0.969 | 0.955 | 0.617 | 0.936 | 0.796 | 0.713 |
| 2 | 66,118 | 0.953 | 0.944 | 0.571 | 0.898 | 0.728 | 0.676 |
| 3 | 34,979 | 0.943 | 0.938 | 0.557 | 0.877 | 0.690 | 0.656 |
| 4 | 20,245 | 0.937 | 0.933 | 0.550 | 0.857 | 0.650 | 0.621 |
| 6 | 7,497 | 0.923 | 0.922 | 0.536 | 0.826 | 0.621 | 0.602 |
| 8 | 2,415 | 0.915 | 0.915 | 0.529 | 0.810 | 0.584 | 0.574 |
| 12 | 1,610 | 0.934 | 0.928 | 0.565 | 0.873 | 0.734 | 0.702 |

## Per class, drive 0006 (class = A's per-view top-1; >= 300 pairs)

| class | n | top-1 same | top-1 C2 | feature cos | feature AUC vs C2 | prob AUC vs C2 |
|---|---|---|---|---|---|---|
| road | 485,610 | 0.906 | 0.867 | 0.954 | 0.570 | 0.551 |
| vegetation | 466,226 | 0.880 | 0.825 | 0.946 | 0.550 | 0.545 |
| sidewalk | 135,746 | 0.585 | 0.456 | 0.939 | 0.605 | 0.624 |
| other ground | 110,998 | 0.254 | 0.200 | 0.932 | 0.566 | 0.579 |
| other structure | 97,929 | 0.303 | 0.279 | 0.926 | 0.561 | 0.545 |
| building | 92,967 | 0.740 | 0.696 | 0.955 | 0.567 | 0.557 |
| car | 69,227 | 0.685 | 0.653 | 0.895 | 0.535 | 0.530 |
| fence | 47,072 | 0.586 | 0.497 | 0.930 | 0.562 | 0.574 |
| parking | 26,868 | 0.409 | 0.309 | 0.945 | 0.581 | 0.597 |
| other object | 16,085 | 0.127 | 0.087 | 0.886 | 0.577 | 0.568 |
| other vehicle | 13,628 | 0.454 | 0.434 | 0.895 | 0.541 | 0.532 |
| terrain | 12,373 | 0.301 | 0.252 | 0.921 | 0.553 | 0.563 |
| truck | 10,900 | 0.728 | 0.715 | 0.910 | 0.555 | 0.523 |
| pole | 8,490 | 0.208 | 0.069 | 0.902 | 0.599 | 0.627 |
| traffic sign | 1,822 | 0.333 | 0.188 | 0.900 | 0.594 | 0.619 |
| person | 1,177 | 0.319 | 0.213 | 0.876 | 0.603 | 0.614 |
| bicycle | 396 | 0.258 | 0.197 | 0.866 | 0.572 | 0.550 |

## Pixel-neighbourhood sensitivity, drive 0006 (81,476 pairs)

| perturbation of the B sample | 1 - cosine |
|---|---|
| 2px | 0.0004 |
| 5px | 0.0025 |
| 1tok | 0.0103 |
| cross-view, verified same point | 0.0584 |
| cross-view, local control point | 0.0676 |

A one-token (11 px) shift explains 18 % of the cross-view feature change.

## Mapper association vs direct correspondence, drive 0006, |offset| <= 3 stream frames

297,949 verified pairs replayed through the mapper's frozen inputs (LingBot depth/poses, frozen scale 15.52).

- mapper has any frame-B observation in the 0.2 m voxel of X: **5.5 %** (frame A: 13.0 %; both: 1.1 %)
- when it has one, its B pixels lie 84 px (median; p25 47, p90 301) from the true projection u_B; within 1 token (11 px) 3.7 %, beyond 30 px 87.8 %
- LingBot depth at u_B vs LiDAR depth: median |error| 1.43 m (11.5 % relative); relative error > 10 % on 56 % of pairs
- LingBot relative pose alone (GT camera-frame point at A carried to B): reprojection error median 93 px, p90 499 px
- LingBot endpoint of the true pixel vs X in the mapper's world: median 5.0 m, p90 10.6 m; > 0.2 m (one voxel) on 100.0 %
- class agreement on the 3,407 pairs where the mapper has observations in both frames: direct same point 0.692, mapper same voxel 0.534; feature cos 0.945 vs 0.902

| stream position of A | pairs | mapper has B obs | pose reproj err (px) | endpoint 3D err (m) | depth rel err | mapper pixel offset (px) | direct top-1 | direct top-1 (both) | mapper top-1 (both) |
|---|---|---|---|---|---|---|---|---|---|
| 0-100 | 17,965 | 9.1 % | 30 | 1.9 | 11 % | 54 | 0.751 | 0.735 | 0.646 |
| 100-300 | 33,923 | 5.8 % | 59 | 3.2 | 12 % | 64 | 0.755 | 0.771 | 0.570 |
| 300-600 | 52,356 | 6.1 % | 59 | 3.2 | 13 % | 49 | 0.756 | 0.679 | 0.596 |
| 600-1000 | 71,287 | 5.3 % | 101 | 5.0 | 10 % | 89 | 0.805 | 0.746 | 0.476 |
| 1000-2000 | 122,418 | 4.7 % | 156 | 6.7 | 12 % | 140 | 0.749 | 0.527 | 0.320 |

Verification: 12/12 checks pass.

