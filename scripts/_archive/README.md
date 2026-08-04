# Archived scripts

Not needed for default reproduce (`make install test validate paper report`).
Live operational set is only under `scripts/` (see `docs/CLEAN_TREE.md`).

| File | Why archived |
|------|----------------|
| auto_pipeline.py / v2 | One-off orchestration superseded by Makefile + agent_h_finalize + run_if_parallel |
| finalize_experiments.py | Lambda-era finalize; obsolete after 540 + `make full` |
| finalize_if_sweep.sh | One-shot IF sweep finalize (done) |
| run_if_rerun_cluster.sh | Cluster IF re-run (grid complete at 540) |
| agent_h_reconcile_prose.py | One-off prose inventory after IF merge |
| agent_h_finalize.sh | Historical copy; **live** copy in `scripts/` |
| watch_sweep_readonly.sh | May appear here only if re-archived; **live** path is `scripts/watch_sweep_readonly.sh` |
| monitor_if_then_regen.sh | IF watch → regen (post-meeting) |
| canonical_watcher*.py, lambda_watcher.py, watch_* | Progress watchers |
| agent_data_refresher.py, data_refresher_loop.sh | Periodic refresh loops |
| auto_complete.sh, finish_everything_when_ready.sh, final_delivery_orchestrator.sh | Full auto orchestrators |
| grok_final_delivery_orchestrator.py | Grok-side poller |
| delayed_poll.py, quick_poll_loop.sh | Misc pollers |
| utkface_watchdog.sh* | UTKFace local watchdog |

- watch_sweep_readonly.sh — IF sweep complete (180/180); obsolete poller
- finalize_experiments.py — superseded by agent_h_finalize.sh
