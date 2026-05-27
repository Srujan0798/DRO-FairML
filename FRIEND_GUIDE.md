# DRO-FAIR RESEARCH - COMPLETE AI EXECUTION GUIDE (1000+ LINES)
# =====================================================================
# READ THIS FILE: AI Agent reads this to understand all tasks
# FRIEND'S JOB: Type ONE LINE, nothing else
# =====================================================================

================================================================================
SECTION 1: PROJECT OVERVIEW
================================================================================

PROJECT: DRO-FAIR (Distributionally Robust Fairness in ML)
REPO: https://github.com/Srujan0798/DRO-FairML
GOAL: Test Fairness-Targeted PGD attacks on ML models using GPU server

WHAT WE ARE BUILDING:
- Gradient-based attack (FairnessTargetedPGD) that increases unfairness in ML models
- Tests on tabular datasets: Adult, Credit, LSAC
- Tests on image dataset: UTKFace (future)
- DRO-FAIR training to defend against attacks

CURRENT STATUS:
- FairnessTargetedPGD class built and tested on local Mac (works!)
- Test results: Adult DP +0.08, Credit DP +0.01, LSAC DP +0.03
- Code ready on GitHub, needs to run on FLAIR2 GPU server

================================================================================
SECTION 2: CRITICAL RULES - READ BEFORE STARTING
================================================================================

RULE 1: FRIEND'S LAPTOP DOES NOTHING
------------------------------------
Friend's laptop/PC is ONLY a display terminal. It does:
- Open browser
- Login to JupyterHub
- Display results

Friend's laptop does NOT do:
- Any computation
- Any GPU work
- Any CPU work (beyond browser)
- Any heating
- Any fan noise

ALL work happens on FLAIR2 GPU SERVER (2x Nvidia L40S, 48GB each)

RULE 2: SRUJAN PULLS AND REVIEWS COMMITS
-----------------------------------------
Friend pushes code to GitHub → Srujan pulls to review → Srujan makes changes → push to GitHub → Friend pulls → Continue

RULE 3: FRIEND CAN MODIFY FILES (REPORTS ONLY)
-----------------------------------------------
Friend can write/edit:
- REPORTS/daily.txt (daily notes)
- REPORTS/execution_log.txt (execution history)
- Any .txt file in REPORTS/ or results/

Friend should NOT modify:
- src/ directory (code)
- scripts/ directory (scripts)
- Any .py files

RULE 4: AI DOES EVERYTHING AUTONOMOUSLY
----------------------------------------
When friend types one line prompt, AI must:
- Login to server
- Clone repo (if needed)
- Pull latest code
- Install packages
- Run experiments
- Save results
- Commit and push to GitHub
- Write summary report

Friend does NOTHING except give the one line prompt.

RULE 5: EXPERIMENTS RUN CONTINUOUSLY
--------------------------------------
This is a research project that runs daily:
- Friend gives AI the prompt
- AI runs experiments
- AI pushes results to GitHub
- Srujan reviews and improves code
- Next day: repeat

Server should be available 24/7 for continuous operation.

================================================================================
SECTION 3: CREDENTIALS AND ACCESS
================================================================================

JUPYTERHUB (Primary Access - Browser Based)
--------------------------------------------
URL: http://flair2.iitgn.ac.in:8000/hub/login
Username: srujan.sai
Password: ss#081

How to access:
1. Open Chrome/Firefox on ANY device (friend's laptop, phone, tablet)
2. Go to URL above
3. Enter username and password
4. Click "Start My Server" or "Login"
5. Click "New" → "Terminal" to open command line

SSH ACCESS (Alternative - Direct Terminal)
-------------------------------------------
Host: 10.0.62.234
Port: 22
Username: srujan.sai
Password: ss#081

From Mac/Linux terminal:
ssh -X srujan.sai@10.0.62.234

From Windows (PuTTY):
Host: 10.0.62.234, Port: 22, User: srujan.sai

SSH X11 Forwarding (for graphical apps):
ssh -X srujan.sai@10.0.62.234

NOTE: JupyterHub is easier for most tasks. SSH is for advanced use.

================================================================================
SECTION 4: SERVER PATHS AND DIRECTORIES
================================================================================

GPU SERVER LOCATION: FLAIR2 at IIT Gandhinagar

BASE DIRECTORY: /data/srujan.sai/

DIRECTORY STRUCTURE:
/data/srujan.sai/
├── DRO-FairML/              # Main project (git clone here)
│   ├── src/                 # Source code (DO NOT EDIT)
│   │   ├── corruption/
│   │   │   └── adversarial.py    # FairnessTargetedPGD class
│   │   ├── data/
│   │   │   └── datasets.py      # load functions
│   │   └── models/
│   │       └── classifier.py     # MLPClassifier
│   ├── scripts/             # Executable scripts
│   │   ├── test_fairness_pgd.py  # MAIN EXPERIMENT SCRIPT
│   │   ├── extract_utkface_features.py
│   │   ├── auto_runner.sh   # AUTO RUNNER
│   │   └── setup_server.sh
│   ├── data/raw/            # Tabular data files
│   │   ├── adult.data
│   │   ├── adult.test
│   │   ├── default_of_credit_card_clients.xls
│   │   └── lsac.csv
│   ├── results/             # Experiment outputs (AI saves here)
│   ├── REPORTS/             # Daily reports (AI writes here)
│   ├── docs/                # Design documents
│   └── FRIEND_GUIDE.md      # THIS FILE
├── UTKFace/                 # UTKFace images (when downloaded)
└── utkface_features.npz    # Extracted features (after extraction)

IMPORTANT PATHS (Use exactly):
- Code: /data/srujan.sai/DRO-FairML/
- Results: /data/srujan.sai/DRO-FairML/results/
- Reports: /data/srujan.sai/DRO-FairML/REPORTS/
- Scripts: /data/srujan.sai/DRO-FairML/scripts/

================================================================================
SECTION 5: GPU SERVER SPECIFICATIONS
================================================================================

FLAIR2 SERVER HARDWARE:
- CPU: 2x Intel Xeon processors
- GPU: 2x Nvidia L40S GPU
- GPU Memory: 48GB per GPU (96GB total)
- RAM: 128GB DDR4
- Storage: SSD (fast read/write)
- Network: 10Gbps (fast for data transfer)

SOFTWARE:
- OS: Ubuntu 22.04 LTS
- CUDA Version: Compatible with PyTorch
- Python: 3.10+
- PyTorch: With CUDA support (torch.cuda.is_available() = True)

VERIFY GPU ACCESS:
nvidia-smi  # Shows GPU info
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"  # Shows GPU name

GPU MEMORY: 48GB per GPU (using only ~8GB for our experiments)

ALL ACTUAL COMPUTATION HAPPENS ON THIS SERVER, NOT ON FRIEND'S LAPTOP.

================================================================================
SECTION 6: DAILY WORKFLOW - COMPLETE FLOW
================================================================================

DAILY CYCLE:

1. Srujan (Local Mac):
   - Reviews GitHub for friend's pushed commits
   - Pulls code: git pull origin main
   - Reviews results in REPORTS/
   - Makes code improvements
   - Pushes to GitHub: git push origin main

2. Friend (From anywhere via laptop):
   - Opens browser → JupyterHub login
   - Gives AI the one line prompt
   - AI does all work on GPU server
   - Results pushed to GitHub
   - Friend walks away (laptop does nothing)

3. AI Agent (On GPU Server):
   - Reads FRIEND_GUIDE.md for context
   - Executes all tasks autonomously
   - Fixes errors without asking
   - Pushes results to GitHub
   - Writes summary report

WORKFLOW DIAGRAM:

[Friend's Laptop: Browser only]
         ↓
[JupyterHub: http://flair2.iitgn.ac.in:8000]
         ↓
[AI Agent Commands]
         ↓
[GPU Server: /data/srujan.sai/DRO-FairML/]
         ↓
[All computation happens HERE on L40S GPUs]
         ↓
[Results saved to REPORTS/ and results/]
         ↓
[Git push to GitHub]
         ↓
[Srujan pulls and reviews on local Mac]

================================================================================
SECTION 7: FRIEND'S ONE LINE PROMPT
================================================================================

FRIEND COPY-PASTES THIS EXACT LINE TO AI:

```
Do the DRO-FAIR research work. Repo: https://github.com/Srujan0798/DRO-FairML. Login: http://flair2.iitgn.ac.in:8000/hub/login (user: srujan.sai, pass: ss#081). Server path: /data/srujan.sai/DRO-FairML. Run experiments, save results in REPORTS/, commit and push to GitHub when done. If anything fails, fix it yourself. Only tell me if you truly cannot fix it.
```

THAT'S IT! FRIEND DOES NOTHING ELSE.

AI WILL DO:
1. Login to JupyterHub (using credentials above)
2. Navigate to code directory
3. Git clone (if not exists): git clone https://github.com/Srujan0798/DRO-FairML.git
4. Git pull origin main (get latest code)
5. Install Python packages (if needed)
6. Run experiments: python3 scripts/test_fairness_pgd.py
7. Save results to REPORTS/daily.txt
8. Git add -A && git commit -m "Friend: experiments on [DATE]"
9. Git push origin main
10. Write summary report
11. Report completion to friend

================================================================================
SECTION 8: CLONING AND SETUP (First Time Only)
================================================================================

IF DRO-FAIRML DOESN'T EXIST ON SERVER:

git clone https://github.com/Srujan0798/DRO-FairML.git /data/srujan.sai/DRO-FairML

FULL SETUP COMMAND:
cd /data/srujan.sai && git clone https://github.com/Srujan0798/DRO-FairML.git && cd DRO-FairML && pip install numpy torch scikit-learn pandas

AFTER CLONE, EVERYTHING IS READY TO RUN.

================================================================================
SECTION 9: AUTO RUNNER - DOES EVERYTHING
================================================================================

SCRIPT: /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

This script does everything automatically:
1. cd to project directory
2. git pull origin main
3. Create REPORTS/ and results/ directories
4. Run test_fairness_pgd.py
5. Save output
6. Commit and push
7. Write summary

HOW TO USE:
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

PREFERRED: AI should use this script instead of manual commands.

================================================================================
SECTION 10: EXPERIMENTS - WHAT TO RUN
================================================================================

PRIMARY EXPERIMENT (NOW - CPU/GPU):
Script: scripts/test_fairness_pgd.py
Dataset: Adult, Credit, LSAC (tabular)
Time: ~5-10 minutes
GPU: Not required (works on CPU)

What it does:
- Trains clean model → measures DP (Demographic Parity)
- Trains with Random corruption → measures DP
- Trains with Heuristic adversarial → measures DP
- Trains with FairnessTargetedPGD (gradient attack) → measures DP
- Compares all methods

Expected results:
- Grad-PGD should INCREASE DP (unfairness) significantly
- Adult: +0.05 to +0.10 increase
- Credit: +0.005 to +0.020 increase
- LSAC: +0.02 to +0.05 increase

SECONDARY EXPERIMENT (LATER - GPU required):
Script: scripts/extract_utkface_features.py
Dataset: UTKFace (200K face images)
Time: ~45 minutes
GPU: Required (uses L40S)

Run only after UTKFace images downloaded.

================================================================================
SECTION 11: HOW TO SAVE RESULTS
================================================================================

AI MUST SAVE RESULTS TO THESE FILES:

1. Full output:
   python3 scripts/test_fairness_pgd.py > /data/srujan.sai/DRO-FairML/results/output_$(date +%Y%m%d_%H%M%S).txt 2>&1

2. Daily report (APPEND, do not overwrite):
   echo "================================" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
   echo "Date: $(date)" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
   echo "================================" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
   tail -50 /data/srujan.sai/DRO-FairML/results/output_*.txt >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt

3. Execution log:
   echo "Ran experiments at $(date)" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt

REPORT FORMAT:
================================
Date: YYYY-MM-DD HH:MM:SS
Experiment: FairnessTargetedPGD

Results:
Adult: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX
Credit: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX
LSAC: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX

Status: SUCCESS
Notes: [any observations]
================================

================================================================================
SECTION 12: GIT WORKFLOW - PUSH AND PULL
================================================================================

FRIEND PUSHES TO GITHUB:
cd /data/srujan.sai/DRO-FairML
git add -A
git commit -m "Friend: experiments on $(date +%Y%m%d)"
git push origin main

SRUJAN PULLS ON LOCAL MAC:
cd ~/Desktop/DRO-FairML
git pull origin main

SRUJAN PUSHES IMPROVEMENTS:
cd ~/Desktop/DRO-FairML
git add -A
git commit -m "Srujan: improved XYZ"
git push origin main

FRIEND PULLS NEW CHANGES:
cd /data/srujan.sai/DRO-FairML
git pull origin main

CYCLE CONTINUES DAILY.

================================================================================
SECTION 13: WHAT FRIEND CAN AND CANNOT DO
================================================================================

FRIEND CAN:
- Open JupyterHub in browser
- Type the one line prompt to AI
- View results in browser
- Check REPORTS/daily.txt
- Check results/ directory
- Walk away (laptop does nothing)

FRIEND CANNOT/ SHOULD NOT:
- Edit source code (src/ files)
- Edit scripts (scripts/ files)
- Edit .py files
- Modify algorithm implementations
- Change model architectures

AI CAN (Autonomously):
- Run experiments
- Save results
- Create reports
- Push to GitHub
- Fix errors
- Install packages

AI CANNOT:
- Edit source code (only Srujan does that)
- Make strategic decisions
- Change research direction

================================================================================
SECTION 14: FILE PERMISSIONS AND WRITING
================================================================================

FRIEND CAN WRITE TO:
- /data/srujan.sai/DRO-FairML/REPORTS/daily.txt (append only)
- /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt (append only)
- /data/srujan.sai/DRO-FairML/results/ (new files)

FRIEND CANNOT WRITE TO:
- /data/srujan.sai/DRO-FairML/src/ (code files)
- /data/srujan.sai/DRO-FairML/scripts/ (script files)
- /data/srujan.sai/DRO-FairML/docs/ (documentation)

AI CAN WRITE ANYWHERE IN /data/srujan.sai/DRO-FairML/ AS NEEDED.

================================================================================
SECTION 15: ERROR HANDLING
================================================================================

IF PACKAGES MISSING:
pip install numpy torch scikit-learn pandas scikit-image

IF GIT CLONE NEEDED:
git clone https://github.com/Srujan0798/DRO-FairML.git /data/srujan.sai/DRO-FairML

IF GIT PUSH FAILS:
- Wait 30 seconds, retry
- Check network: ping github.com
- Max 3 retries, then save locally

IF EXPERIMENT FAILS:
- Check error message
- Fix and retry
- Report to friend only if cannot fix

IF GPU NOT FOUND:
- Use CPU (works fine for our experiments)
- Don't require GPU

IF SERVER NOT RESPONDING:
- Refresh JupyterHub page
- Re-login if needed
- Report to friend if still not working

================================================================================
SECTION 16: TECHNICAL DETAILS FOR AI
================================================================================

FAIRNESSTARGETEDPGD CLASS:
- Located: src/corruption/adversarial.py
- Purpose: Gradient-based attack to maximize unfairness (DP)
- Key method: compute_dp_gradient(y, a) - computes which samples to flip

KEY METRICS:
- DP = |P(Y=1|A=0) - P(Y=1|A=1)|
- Higher DP = more unfair
- Attack should INCREASE DP

DATASETS:
- Adult: 29K train, 9K test, binary income prediction
- Credit: 19.5K train, 6K test, credit default prediction
- LSAC: 12K train, 3.7K test, bar passage prediction

PROTECTED ATTRIBUTES:
- Adult: sex (0=Female, 1=Male)
- Credit: sex (0=Female, 1=Male)
- LSAC: male (0=Female, 1=Male)

================================================================================
SECTION 17: EXPECTED RESULTS
================================================================================

LOCAL TEST RESULTS (MacBook Air, verified working):
Adult:   Clean DP=0.14 → Grad-PGD DP=0.22 (+0.08 increase)
Credit:  Clean DP=0.01 → Grad-PGD DP=0.02 (+0.01 increase)
LSAC:    Clean DP=0.02 → Grad-PGD DP=0.05 (+0.03 increase)

EXPECTED ON SERVER: Similar results (computation method is same)

SUCCESS INDICATORS:
- Grad-PGD DP > Clean DP (attack works)
- All three datasets show DP increase
- Experiments complete without errors

================================================================================
SECTION 18: COMMUNICATION BETWEEN FRIEND AND SRUJAN
================================================================================

FRIEND → SRUJAN (WhatsApp/Discord):
- "Experiments done. Results: Adult 0.22, Credit 0.02, LSAC 0.05"
- "Error: GPU not responding" (if severe)
- "Code pushed to GitHub, check REPORTS/"

SRUJAN → FRIEND:
- "New code pushed, run experiments"
- "Check FRIEND_GUIDE.md for updates"
- "Can now start UTKFace feature extraction"

SRUJAN REVIEWS ON GITHUB:
- Go to: https://github.com/Srujan0798/DRO-FairML/commits/main
- See all friend's pushed commits
- Check results in REPORTS/daily.txt
- Make improvements to code

================================================================================
SECTION 19: SERVER AVAILABILITY
================================================================================

FLAIR2 SERVER: Available 24/7 (except maintenance windows)

Working hours: Anytime, anywhere
Time to run experiments: ~10 minutes per day

If server is down:
- Wait and retry
- Check JupyterHub URL: http://flair2.iitgn.ac.in:8000
- Contact IIT Gandhinagar support if persistent

================================================================================
SECTION 20: COMPLETE COMMAND REFERENCE
================================================================================

# OPEN JUPYTERHUB
# Browser: http://flair2.iitgn.ac.in:8000/hub/login
# Login: srujan.sai / ss#081

# CLONE REPO (First time only)
git clone https://github.com/Srujan0798/DRO-FairML.git /data/srujan.sai/DRO-FairML

# NAVIGATE TO CODE
cd /data/srujan.sai/DRO-FairML

# GET LATEST CODE
git pull origin main

# INSTALL PACKAGES
pip install numpy torch scikit-learn pandas scikit-image

# SET PYTHONPATH
export PYTHONPATH=/data/srujan.sai/DRO-FairML

# RUN EXPERIMENTS
python3 scripts/test_fairness_pgd.py

# RUN AUTO RUNNER (Does everything)
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

# SAVE RESULTS
python3 scripts/test_fairness_pgd.py > results/output_$(date +%Y%m%d_%H%M%S).txt 2>&1

# COMMIT AND PUSH
git add -A
git commit -m "Friend: experiments on $(date +%Y%m%d)"
git push origin main

# VIEW REPORTS
cat REPORTS/daily.txt
ls -la results/

# CHECK GPU
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

================================================================================
SECTION 21: QUICK REFERENCE CARD
================================================================================

FRIEND'S ONE LINE TO AI:
Do the DRO-FAIR research work. Repo: https://github.com/Srujan0798/DRO-FairML. Login: 
http://flair2.iitgn.ac.in:8000/hub/login (user: srujan.sai, pass: ss#081). Server path: 
/data/srujan.sai/DRO-FairML. Run experiments, save results in REPORTS/, commit and push 
to GitHub when done. If anything fails, fix it yourself. Only tell me if you truly cannot fix it.

FRIEND'S LAPTOP: Does NOTHING (just browser display)
GPU SERVER: Does ALL computation (2x Nvidia L40S, 48GB each)

AI DOES:
1. Login to JupyterHub
2. Clone repo (if needed)
3. Pull latest code
4. Install packages
5. Run experiments
6. Save results to REPORTS/
7. Push to GitHub
8. Write summary

SRUJAN DOES:
1. Reviews GitHub commits
2. Pulls code to local Mac
3. Reviews results in REPORTS/
4. Improves code
5. Pushes to GitHub
6. Repeat

================================================================================
SECTION 22: PROJECT ROLES SUMMARY
================================================================================

FRIEND (Server Operator):
- Types one line prompt to AI
- Nothing else
- Laptop stays cool and quiet
- Only browses results

AI (Autonomous Executor):
- Reads FRIEND_GUIDE.md
- Does all work on GPU server
- Fixes errors autonomously
- Pushes to GitHub
- Writes reports

SRUJAN (Project Lead):
- Develops code on local Mac
- Reviews friend's GitHub commits
- Makes improvements
- Pushes to GitHub
- Communicates with professor

================================================================================
SECTION 23: IMPORTANT REMINDERS
================================================================================

✓ FRIEND'S LAPTOP DOES NOTHING - ALL WORK ON GPU SERVER
✓ FRIEND CANNOT EDIT CODE - ONLY REPORTS
✓ AI DOES EVERYTHING - FRIEND ONLY TYPES ONE LINE
✓ SRUJAN PULLS AND REVIEWS - FRIEND PUSHES
✓ EXPERIMENTS RUN DAILY - CONTINUOUS CYCLE
✓ SERVER AVAILABLE 24/7 - GPU DOES ALL COMPUTATION
✓ FRIEND CLONES FROM GITHUB - AI PULLS AND RUNS
✓ RESULTS SAVED TO REPORTS/ - SRUJAN REVIEWS THERE
✓ AI WRITES DAILY REPORTS - FRIEND CAN VIEW IN BROWSER
✓ GIT PUSH/PULL CYCLE - DAILY WORKFLOW

================================================================================
END OF FRIEND_GUIDE.md
================================================================================
THIS FILE IS READ BY AI TO EXECUTE ALL TASKS AUTONOMOUSLY
FRIEND TYPES ONE LINE, AI DOES EVERYTHING ELSE
SRUJAN PULLS AND REVIEWS COMMITS FROM FRIEND
================================================================================