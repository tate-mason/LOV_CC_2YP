# Historical files

This folder holds superseded drafts, exploratory implementations, older source
revisions, proposals, and legacy outputs, organized on 2026-11-17.

Paths beneath `old/` preserve the original repository paths. For example,
`old/Code/full_draft_0901/estimation.py` originally lived at
`Code/full_draft_0901/estimation.py`.

See [manifest.csv](manifest.csv) for every original path, destination, size, and
classification reason. Source hashes are recorded for copied-source comparisons.
No files were deleted. Restore a file by moving it to its recorded original path,
checking first that a newer file does not already occupy that location.

Exact copies of retained active source and generated TeX/bytecode artifacts were
placed in [waste/](../waste/README.md), rather than here. Some historical source
snapshots therefore span both manifests; consult both when restoring a revision.
Archived documents retain historical path references and may need restoration
before rebuilding. Historical audit/change logs in the active tree intentionally
retain their original provenance paths.

Current code is in `Code/final_simple`, the seminar deck is in
`Code/final_draft/seminar_pres`, and the November 8 writeup remains at
`Code/final_draft/writeup/1108_writeup`. Legacy `final_draft` modules referenced by
current validation/build scripts were retained. Estimate inputs and referenced
aggregate results were also retained.

Environments and third-party replication code were left in place. Files listed
in [retained_for_review.csv](retained_for_review.csv) were not opened or moved:
their filenames identify protected data or do not establish that diagnostic
contents are safe aggregate output. Empty original directories may remain.

The research-server-only data boundary still applies to archived code. Moving a
script here does not authorize local empirical execution.
