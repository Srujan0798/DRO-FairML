#!/usr/bin/env python3
import json, collections, subprocess, time, os

def poll_and_log():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("results/canonical_tau1.json") as f:
            data = json.load(f)
        n = len(data)
        ds = collections.Counter(r.get("dataset", "unknown") for r in data)
        has_credit_lsac = any(r.get("dataset") in ("credit", "lsac") for r in data)
        last_ds = [r.get("dataset") for r in data[-3:]]
        last_row = {k: data[-1].get(k) for k in ["dataset","alpha","seed","attack","method","radii_mode","k_inner","tau"]} if data else {}
        
        # ps check
        ps_can = subprocess.getoutput("ps -p 21531 -o pid,stat,%cpu,etime,command 2>/dev/null | cat")
        ps_emp = subprocess.getoutput("ps aux | grep -E 'empirical|run_canonical.*--radii_mode empirical' | grep -v grep | cat")
        
        # check empirical json
        emp_n = 0
        try:
            with open("results/canonical_tau1_empirical.json") as f:
                emp_data = json.load(f)
            emp_n = len(emp_data)
        except:
            pass
        
        msg = f"[{timestamp}] CANONICAL: {n}/540 | DATASETS: {dict(ds)} | HAS_CREDIT_LSAC={has_credit_lsac} | LAST3: {last_ds}"
        print(msg)
        print(f"  LAST_ROW: {last_row}")
        print(f"  PID21531: {ps_can.strip()}")
        print(f"  EMPIRICAL_PROC: {ps_emp.strip() or 'None'} | EMP_JSON_ROWS: {emp_n}")
        
        # write to watcher log
        with open("logs/canonical_watcher.log", "a") as logf:
            logf.write(msg + "\n")
            logf.write(f"  LAST_ROW: {last_row}\n")
            logf.write(f"  PID21531: {ps_can.strip()}\n")
            logf.write(f"  EMPIRICAL_PROC: {ps_emp.strip() or 'None'} | EMP_JSON_ROWS: {emp_n}\n")
            logf.write("---\n")
        
        return n, ds, has_credit_lsac, emp_n
    except Exception as e:
        err = f"[{timestamp}] ERROR: {e}"
        print(err)
        with open("logs/canonical_watcher.log", "a") as logf:
            logf.write(err + "\n")
        return 0, {}, False, 0

if __name__ == "__main__":
    poll_and_log()
