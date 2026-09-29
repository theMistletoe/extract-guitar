# exp032_R_r3_c6 (aborted)

The validation part finished (41/41 clips, per_clip.csv: mean SDR 6.503 dB, median 5.33,
real 4.06 / synthetic 8.62, retention 0.655, SIR 10.9, SAR 9.0), but writing the
target-song output failed because the disk filled up (LibsndfileError), so no metrics.json,
target candidate or Champion/Challenger decision were recorded. The member cache for this
run was pruned while freeing disk. The experiment was re-run from scratch as the next
experiment id with the same pipeline (configs/pipelines/R_r3_c6.yaml); use that record.
