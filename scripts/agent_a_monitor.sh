#!/bin/bash
LOGFILE="logs/agent_a_status.log"
echo "=== Agent A Monitor (lambda priority) started $(date) ===" >> $LOGFILE
LAMBDA_LOG="logs/lambda_grid_run.log"
while true; do
  echo "=== POLL at $(date) ===" >> $LOGFILE
  /usr/local/bin/python3 /tmp/count_results.py >> $LOGFILE 2>&1
  tail -3 $LAMBDA_LOG 2>/dev/null | sed 's/^/  LOG: /' >> $LOGFILE
  ps aux | grep -E '[r]un_lambda_lr_grid.py' | awk '{print "  PS:", $2, $3"%", $(NF-1)}' >> $LOGFILE 2>&1 || echo "  no ps" >> $LOGFILE
  sleep 180
done
