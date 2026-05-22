# OpenSearch ml-commons 3.7 Release Test Research

This folder contains research and test plans for the OpenSearch ml-commons 3.7 release.

## Scope

The 3.7 release covers all changes merged to `main` since the **3.5.0** release. Because the
`main` branch at the time of writing is the 3.7 development line, the relevant changes were
shipped in two waves:

- **3.5.0 → 3.6.0** (the 3.6 release, already cut on branch `3.6`)
- **3.6.0 → 3.7.0** (new in 3.7, currently on `main`)

We intentionally include both waves: any user upgrading from 3.5 to 3.7 will pick up everything
from both, so QA must validate them as a single delta.

## Source ranges

- 3.5 baseline commit: `581b229c8` (tag `3.5.0.0`)
- 3.6 baseline commit: `4f7054ff0` (tag `3.6.0.0`)
- 3.7 head: tip of `upstream/main` at research time (`1d3eb9afe`)

Total commits in the 3.5 → 3.7 delta: **71** (48 in 3.6 wave + 23 in 3.7 wave).

## Files

- `README.md` — this overview
- `changes-3.6.md` — features, enhancements, bug fixes, infra in the 3.5 → 3.6 wave
- `changes-3.7.md` — features, enhancements, bug fixes, infra in the 3.6 → 3.7 wave
- `test-plan.md` — concrete test items grouped by area, with priorities and entry points

## How this was generated

- Commit list: `git log --oneline 3.5.0.0..upstream/main`
- Cross-referenced against the auto-generated 3.5 and 3.6 release notes under
  `release-notes/` for category labels.
- Per-commit file impact: `git show --stat <sha>` for the larger items.

## How to use

1. Start with `test-plan.md` to drive QA — it lists test areas grouped by feature/fix.
2. Use `changes-3.6.md` and `changes-3.7.md` to find the underlying PR for any given test item
   (PR numbers are linked by `#NNNN`).
3. For each PR, the GitHub description is usually the most precise spec; the commit message is
   the next best source.
