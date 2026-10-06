# Paper decision after the final semantic-method experiment (rule frozen in `source_predeclaration.md`)

- **Source gate:** SOURCE FAIL. On KITTI-360 0006 completed_unobserved, B4 is 8.22 mIoU vs B0 8.80 and B3 9.38. S1–S4 fail; S5 and S6 pass.
- **Targets:** not opened.
- **Rule:** PAPER C includes SOURCE FAIL.

What is established:
- **Generative field.** It reaches identities transport cannot: +3.6 mIoU on candidate-unsupported voxels. It loses more on candidate-supported voxels (−13.1) and favours frequent classes. The net effect is negative.
- **Candidate transport (B0, B3).** Sub-point, positive gains; teacher-limited semantics (final zero-shot evaluation).
- **Geometry.** The causal single-camera geometry is the strong verified result: better than OccAny's released geometry under the same evaluator on both target datasets.

**PAPER C**

**Next action:** close semantic-completion method development and write the paper around the causal single-camera zero-shot geometry result, reporting the semantic-completion experiments (B0, B3, B4) as limitations.
