# ONE LINE COMMANDS FOR FRIEND

## SETUP (One Time)
```bash
cd /data/srujan.sai && git clone https://github.com/Srujan0798/DRO-FairML.git && cd DRO-FairML && pip install numpy torch scikit-learn pandas
```

## DAILY WORK
```bash
cd /data/srujan.sai/DRO-FairML && git pull origin main && export PYTHONPATH=/data/srujan.sai/DRO-FairML && python3 scripts/test_fairness_pgd.py
```

## PUSH CODE TO GITHUB (For review)
```bash
cd /data/srujan.sai/DRO-FairML && git add . && git commit -m "YOUR NAME: what you did today" && git push origin main
```

## WRITE REPORT
```bash
cd /data/srujan.sai/DRO-FairML && echo "DATE: $(date)" >> REPORTS/$(date +%Y%m%d).txt && echo "RESULTS:" >> REPORTS/$(date +%Y%m%d).txt && cat results/latest.txt >> REPORTS/$(date +%Y%m%d).txt 2>/dev/null || echo "No results yet" >> REPORTS/$(date +%Y%m%d).txt
```

---

## WHAT FRIEND DOES:
1. Login to http://flair2.iitgn.ac.in:8000/hub/login (user: srujan.sai, pass: ss#081)
2. Open Terminal
3. Copy ONE command above and paste
4. When done, PUSH code: `git add . && git commit -m "Friend: did X" && git push origin main`
5. Write report in REPORTS folder

## WHAT SRUJAN DOES:
- Review pushed code
- Make changes/improvements
- Push back to GitHub
- Friend pulls and continues

---

# AUTO RUNNER (Does everything automatically)
```bash
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh
```

This runs experiments, saves results, and commits automatically every day.