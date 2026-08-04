# Loop status

**Cadence:** every **5 minutes** · job `019fcbd740be`  
**Updated:** 2026-08-04 09:33 UTC

## Locked science
- canonical **540/540** · IF **mixed** · Adult/DP α=0.1 **5/6** · LSAC/DP degenerate

## This wave
- **Restored** `_archive` trees after hard-delete / "do not restore" commits (hard rule: archive over delete)
- **`docs/ARCHIVE_POLICY.md`** + INDEX
- Promoted core design docs back to live `docs/*.md` from `docs/reference/`
- Report-live figs at `figures/` root
- UTKFace REAL **42/90** ({'dp': 30, 'if': 12}) — **no paper claim**

## Counts
| Area | Count |
|------|------:|
| docs live `*.md` | 17 |
| experiments `*.py` | 17 |
| scripts active | 4 |
| docs/_archive entries | ~70 |
| UTKFace REAL | **42 / 90** |

## Attention
Concurrent agent commits (`f9b219b` re-purge) conflict with archive-over-delete. Do not re-delete archives.

## Reproduce
`make install && make data && make test && make validate && make paper && make report`
