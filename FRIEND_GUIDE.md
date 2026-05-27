# DRO-FAIR RESEARCH PROJECT - COMPLETE AI AGENT EXECUTION GUIDE
# =====================================================================
# THIS FILE IS READ BY AI - FRIEND TYPES ONE LINE: "do your work"
# AI MUST UNDERSTAND FULL CONTEXT AND EXECUTE ALL TASKS AUTONOMOUSLY
# =====================================================================

================================================================================
SECTION 1: PROJECT BACKGROUND AND CONTEXT
================================================================================

PROJECT NAME: DRO-FAIR (Distributionally Robust Fairness in Machine Learning)

RESEARCH DOMAIN: Adversarial Machine Learning, Fairness in ML Systems

PROBLEM STATEMENT:
We are researching how adversarial attacks can corrupt training data to make 
machine learning models produce unfair predictions across protected groups 
(such as gender, race, age). We have built a "FairnessTargetedPGD" attack that 
uses gradient-based optimization to maximally increase unfairness, and we need 
to test it on real datasets using GPU compute resources.

WHY THIS RESEARCH MATTERS:
- Modern ML systems make decisions in hiring, lending, criminal justice, healthcare
- These systems must treat all groups fairly regardless of protected attributes
- Adversarial attacks could corrupt data to make systems discriminate
- Understanding attack strength helps build better defenses
- This extends our academic course project on DRO-FAIR framework

RESEARCH QUESTIONS:
1. How effective is gradient-based attack compared to random/heuristic attacks?
2. Can adversarial corruption significantly increase Demographic Parity violation?
3. Does the attack work across different datasets (Adult, Credit, LSAC)?
4. Can we defend against such attacks using DRO-FAIR training?

================================================================================
SECTION 2: TECHNICAL FOUNDATION
================================================================================

FAIRNESS METRIC: Demographic Parity (DP)
----------------------------------------
Definition: DP = |P(Y=1|A=0) - P(Y=1|A=1)|

Where:
- Y = model's prediction (binary: 0 or 1)
- A = protected attribute (binary: 0 or 1, e.g., gender: 0=Female, 1=Male)
- P(Y=1|A=0) = probability of positive prediction when protected attr = 0
- P(Y=1|A=1) = probability of positive prediction when protected attr = 1

Interpretation:
- DP = 0 means perfectly fair (same prediction rates across groups)
- DP > 0 means unfair (one group gets more positive predictions)
- DP close to 1 means highly unfair (one group rarely gets positive predictions)

Example:
- If 80% of males get positive prediction but only 40% of females do:
- DP = |0.80 - 0.40| = 0.40 (significant unfairness)

INDIVIDUAL FAIRNESS (IF) - Alternative Metric
----------------------------------------------
Definition: IF = E[|h(x_i) - h(x_j)||a_i == a_j, similar(x_i, x_j)]

Where:
- h(x) = model prediction for input x
- similar(x_i, x_j) = inputs are similar in feature space
- a_i == a_j = same protected group

Interpretation: Similar individuals should get similar predictions regardless
of protected attribute. Harder to compute than DP.

FAIRNESSTARGETEDPGD ATTACK - Core Innovation
-------------------------------------------
This is a gradient-based adversarial attack that maximizes unfairness.

How it works:
1. Given trained model h, compute DP = |P(Y=1|A=0) - P(Y=1|A=1)|
2. Compute gradient: d(DP)/d(y_i) for each training sample i
3. The gradient tells how much flipping sample i's label would increase DP
4. Select top-k samples with largest positive gradient
5. Flip their labels (0→1 or 1→0)
6. Retrain model on corrupted data

Mathematical formulation:
- For group g ∈ {0, 1}, group rate p_g = mean(y[g])
- DP = |p_0 - p_1|
- d(DP)/d(y_i) depends on which group sample i belongs to

Gradient computation rules:
- If p_0 >= p_1 (group 0 has higher rate):
  - Group 0: flip 0→1 increases p_0 → grad = +1
  - Group 0: flip 1→0 decreases p_0 → grad = -1
  - Group 1: flip 1→0 decreases p_1 → grad = +1
  - Group 1: flip 0→1 increases p_1 → grad = -1
- If p_1 > p_0 (group 1 has higher rate):
  - Opposite: flip samples that widen the gap

Attack parameters:
- alpha: fraction of samples to corrupt (0.2 = 20%)
- pgd_steps: number of PGD iterations (default: 5)
- coordinated: if True, 70% of corruption targets minority group

COMPARISON WITH OTHER ATTACKS:
1. Random Corruption: Flip labels randomly
   - Baseline comparison
   - Expected effect: minimal DP change
   
2. Heuristic Adversarial: Use rules to flip labels
   - E.g., flip minority samples 0→1 if minority has lower rate
   - Simple but suboptimal
   
3. FairnessTargetedPGD (OUR): Use exact gradient
   - Computes d(DP)/d(y_i) analytically
   - Flips samples that maximally increase unfairness
   - Expected: highest DP increase

================================================================================
SECTION 3: DATASETS AND THEIR STRUCTURE
================================================================================

ADULT CENSUS INCOME DATASET
---------------------------
Source: UCI Machine Learning Repository
Task: Predict if person earns >$50K/year (binary classification)
Protected attribute: sex (1=Male, 0=Female)

Data characteristics:
- ~48,000 samples (train + test combined)
- 11 features after preprocessing
- Features: age, workclass, education-num, marital-status, occupation,
  relationship, race, capital-gain, capital-loss, hours-per-week, native-country
- Binary labels: income >50K (1) or <=50K (0)

Split used:
- Training: 29,393 samples (80%)
- Validation: 6,784 samples (15% of train)
- Testing: 9,045 samples (20%)

Expected results:
- Clean model: DP ~0.14 (baseline unfairness)
- After attack: DP ~0.22 (significant increase)

CREDIT CARD DEFAULT DATASET
----------------------------
Source: UCI Machine Learning Repository
Task: Predict if person will default on credit card (binary)
Protected attribute: SEX (1=Male, 0=Female)

Data characteristics:
- 30,000 samples
- 23 features (credit limit, payment history, bill amounts, etc.)
- Binary labels: default (1) or not (0)

Split used:
- Training: 19,500 samples (65%)
- Validation: 4,500 samples (15%)
- Testing: 6,000 samples (20%)

Expected results:
- Clean model: DP ~0.01 (nearly fair)
- After attack: DP ~0.02 (small increase)

LSAC BAR PASSAGE DATASET
------------------------
Source: Law School Admissions Council
Task: Predict if person will pass bar exam (binary)
Protected attribute: male (1=Male, 0=Female)

Data characteristics:
- ~17,000 samples
- Features: LSAT score, undergraduate GPA, race, gender, etc.
- Binary labels: pass_bar (1) or fail (0)

Split used:
- Training: 12,149 samples (65%)
- Validation: 2,804 samples (15%)
- Testing: 3,739 samples (20%)

Expected results:
- Clean model: DP ~0.02 (fair)
- After attack: DP ~0.05 (notable increase)

UTKFACE DATASET (FUTURE WORK)
-----------------------------
Source: UTKFace (Face age, gender, race dataset)
Task: Predict gender from face image
Protected attribute: actual gender (for fairness measurement)

Data characteristics:
- 200,000+ face images
- Filename format: {age}_{gender}_{race}_{date}.jpg.chip.jpg
- Gender: 0=Female, 1=Male
- Race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
- Age: 0-116 years

For fairness: predict gender from face, measure DP across actual gender

Current status: NOT YET RUN - requires GPU for feature extraction

================================================================================
SECTION 4: CODE ARCHITECTURE AND FILE STRUCTURE
================================================================================

REPOSITORY INFORMATION:
- GitHub URL: https://github.com/Srujan0798/DRO-FairML
- Branch: main
- Local server path: /data/srujan.sai/DRO-FairML/

DIRECTORY STRUCTURE:
/data/srujan.sai/DRO-FairML/
├── src/
│   ├── __init__.py
│   ├── corruption/
│   │   ├── __init__.py
│   │   └── adversarial.py      # FairnessTargetedPGD class
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py          # Dataset loaders
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py        # MLPClassifier
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py           # Fairness metrics
├── scripts/
│   ├── test_fairness_pgd.py     # MAIN EXPERIMENT SCRIPT
│   ├── extract_utkface_features.py  # CNN feature extraction
│   ├── auto_runner.sh           # Automated execution
│   ├── setup_server.sh          # Server setup
│   └── quick_test.py            # Quick verification
├── data/
│   └── raw/
│       ├── adult.data           # Adult training data
│       ├── adult.test           # Adult test data
│       ├── default_of_credit_card_clients.xls  # Credit data
│       └── lsac.csv             # LSAC data
├── results/                     # Experiment outputs
├── REPORTS/                     # Daily reports
├── docs/
│   ├── FAIRNESS_PGD_DESIGN.md   # Attack design doc
│   └── UTKFACE_PIPELINE.md     # Image pipeline design
├── tests/
│   └── test_fairness_pgd.py     # Unit tests
├── experiments/
│   └── run_fairness_pgd.py      # Main training script
├── FRIEND_GUIDE.md              # This file - for AI reading
├── README.md
└── LICENSE

KEY CLASSES AND THEIR LOCATIONS:

1. FairnessTargetedPGD (src/corruption/adversarial.py, line ~203)
   Methods:
   - __init__(alpha, target_metric, pgd_steps, coordinated, random_state)
   - compute_dp_gradient(y, a) → gradient array
   - compute_if_gradient(y, a) → gradient array (placeholder)
   - compute_fairness_gradient(y, a) → combined gradient
   - _select_targets(grad, n_corrupt, a) → target indices
   - _attack_labels_fairness(y, a) → corrupted labels
   - corrupt(X, y, a, model, device) → X_c, y_c, a_c, corrupt_mask

2. AdversarialCorruptor (src/corruption/adversarial.py, line ~1)
   Existing heuristic attack for comparison
   Methods:
   - _attack_features_fgsm(X, y, corrupt_idx)
   - _attack_labels_coordinated(y, a, corrupt_idx)
   - _attack_attributes(a, corrupt_idx)
   - corrupt(X, y, a) → X_c, y_c, a_c, corrupt_mask

3. RandomCorruptor (src/corruption/adversarial.py, line ~204)
   Random baseline for comparison

4. MLPClassifier (src/models/classifier.py)
   Simple feedforward neural network
   Methods:
   - __init__(input_dim, hidden_dims, dropout)
   - forward(x) → logits
   - predict_proba(x, temperature) → probabilities
   - predict(x) → binary 0/1

5. Dataset Loaders (src/data/datasets.py)
   - load_adult(data_dir) → X, y, a, name
   - load_credit(data_dir) → X, y, a, name
   - load_lsac(data_dir) → X, y, a, name
   - load_utkface(data_dir, feature_cache) → X, y, a, name
   - get_dataset(name, data_dir, test_size, val_size, random_state)
     → X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, name

IMPORTANT IMPLEMENTATION NOTES:

- StandardScaler fit ONLY on training data (no data leakage)
- RandomState used for reproducibility (seed=42)
- Stratified splits for train/test/val
- Protected attribute 'a' is never used as model input
- All data standardized to mean=0, std=1

================================================================================
SECTION 5: LOGIN CREDENTIALS AND SERVER ACCESS
================================================================================

JUPYTERHUB - Primary Interface
-------------------------------
URL: http://flair2.iitgn.ac.in:8000/hub/login
Type: Jupyter Notebook interface running on GPU server
Use: Login here to access terminal and run experiments

Credentials:
- Username: srujan.sai
- Password: ss#081

How to use:
1. Open browser (Chrome/Firefox) on any device
2. Navigate to URL above
3. Enter username and password
4. Click "Start" or "Login"
5. From dashboard: New → Terminal to open command line

SSH Access - Alternative
------------------------
Host: 10.0.62.234
Port: 22
Protocol: SSH with X11 forwarding optional

Credentials:
- Username: srujan.sai
- Password: ss#081

From Linux/Mac:
ssh -X srujan.sai@10.0.62.234

From Windows:
Use PuTTY with Xming for graphical forwarding

JupyterHub vs SSH:
- JupyterHub: Easiest, browser-based, good for most tasks
- SSH: Direct terminal access, better for large file transfers

SERVER DIRECTORY STRUCTURE
--------------------------
Base path: /data/srujan.sai/

- /data/srujan.sai/                    # Home directory
- /data/srujan.sai/DRO-FairML/         # Project code (git clone here)
- /data/srujan.sai/DRO-FairML/data/raw/ # Tabular datasets
- /data/srujan.sai/UTKFace/            # UTKFace images (when downloaded)
- /data/srujan.sai/utkface_features.npz # Extracted CNN features
- /data/srujan.sai/venv/               # Python virtual environment

GPU SERVER SPECIFICATIONS
-------------------------
Hardware:
- CPU: 2x Intel Xeon processors
- GPU: 2x Nvidia L40S
- GPU Memory: 48GB each (96GB total)
- RAM: 128GB DDR4
- Storage: SSD (fast read/write)

Software:
- OS: Ubuntu 22.04 LTS
- CUDA: Available via PyTorch
- Python: 3.10+
- PyTorch: With CUDA support

GPU Access Verification:
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

Expected output:
CUDA available: True
GPU count: 2

IMPORTANT: ALL actual computation happens on this GPU server. Friend's 
laptop/PC is just a display terminal - it does ZERO computation.

================================================================================
SECTION 6: DAILY WORK PROTOCOL - EXECUTION STEPS
================================================================================

WHEN FRIEND SAYS: "do your work" or "run experiments" or similar

AI AGENT MUST EXECUTE ALL OF THE FOLLOWING STEPS IN ORDER:

STEP 1: ACCESS JUPYTERHUB
- Open browser to: http://flair2.iitgn.ac.in:8000/hub/login
- Login with: srujan.sai / ss#081
- Click: New → Terminal (opens command line on GPU server)

STEP 2: NAVIGATE TO PROJECT DIRECTORY
cd /data/srujan.sai/DRO-FairML
pwd  # Verify we're in the right directory

STEP 3: GET LATEST CODE FROM GITHUB
git pull origin main

If git reports errors:
- "Couldn't find remote ref": git fetch origin && git checkout main && git pull
- "Please commit your changes first": git stash && git pull && git stash pop
- "Permission denied": Check if token needed, try: git pull https://<token>@github.com/Srujan0798/DRO-FairML.git

STEP 4: ENSURE PYTHON PACKAGES ARE INSTALLED
pip install numpy torch scikit-learn pandas scikit-image --quiet

Verify installation:
python3 -c "import numpy; import torch; import sklearn; print('All packages OK')"

If import fails, try:
pip install --upgrade pip
pip install numpy torch scikit-learn pandas scikit-image

STEP 5: SET PYTHONPATH
export PYTHONPATH=/data/srujan.sai/DRO-FairML

This ensures Python can find our src/ modules.

STEP 6: CREATE OUTPUT DIRECTORIES
mkdir -p /data/srujan.sai/DRO-FairML/REPORTS
mkdir -p /data/srujan.sai/DRO-FairML/results

STEP 7: RUN MAIN EXPERIMENT
python3 scripts/test_fairness_pgd.py

Expected runtime: 5-10 minutes on CPU (faster on GPU)

This script will:
- Load Adult, Credit, LSAC datasets
- For each dataset:
  - Train clean model (no corruption) → record DP
  - Train with Random corruption → record DP
  - Train with Heuristic adversarial → record DP
  - Train with FairnessTargetedPGD → record DP
- Print comparison table
- Save results

STEP 8: SAVE OUTPUT TO FILES
# Save full output
python3 scripts/test_fairness_pgd.py > /data/srujan.sai/DRO-FairML/results/output_$(date +%Y%m%d_%H%M%S).txt 2>&1

# Save summary to daily report
echo "================================" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
echo "Date: $(date)" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
echo "================================" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
tail -50 /data/srujan.sai/DRO-FairML/results/output_*.txt >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt 2>/dev/null || echo "Output capture failed" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt

STEP 9: COMMIT AND PUSH TO GITHUB
cd /data/srujan.sai/DRO-FairML
git add -A
git commit -m "Auto-run $(date): Fairness PGD experiments"
git push origin main

If push fails:
- Network timeout: Wait 30 seconds, retry (max 3 attempts)
- "Permission denied": May need authentication token
- "Repository not found": Check remote URL: git remote -v

STEP 10: WRITE EXECUTION LOG
echo "----------------------------------------" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt
echo "Execution at $(date)" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt
echo "Experiments completed successfully" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt
echo "Results saved to REPORTS/daily.txt" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt
echo "Code pushed to GitHub" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt
echo "" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt

STEP 11: VERIFY COMPLETION
ls -la /data/srujan.sai/DRO-FairML/results/
ls -la /data/srujan.sai/DRO-FairML/REPORTS/
cat /data/srujan.sai/DRO-FairML/REPORTS/daily.txt | tail -30

================================================================================
SECTION 7: EXPERIMENT OUTPUT SPECIFICATIONS
================================================================================

SCRIPT: scripts/test_fairness_pgd.py

WHAT IT DOES:
1. Loads three datasets in sequence: Adult, Credit, LSAC
2. For each dataset, runs 4 experiments:
   - Clean: No corruption, measure DP
   - Random: Random corruption (alpha=0.2), measure DP
   - Heuristic: AdversarialCorruptor (alpha=0.2), measure DP
   - Grad-PGD: FairnessTargetedPGD (alpha=0.2), measure DP
3. Compares results across methods

TRAINING CONFIGURATION:
- Model: MLPClassifier([128, 64], dropout=0.1)
- Optimizer: Adam(lr=0.01)
- Epochs: 20 (reduced from 100 for speed)
- Loss: Binary cross-entropy
- Early stopping: None (fixed epochs)

EXPECTED OUTPUT FORMAT:
============================================================
Dataset: adult, alpha=0.2
============================================================
Train: 29393, Val: 6784, Test: 9045
Protected: a=0: 9558, a=1: 19835
Label: y=0: 22108, y=1: 7285

[1/5] Clean baseline...
Clean: DP=X.XXXX, Acc=X.XXXX (Xs.Xs)

[2/5] Random corruption...
Random: DP=X.XXXX, Acc=X.XXXX (n=XXXXX)

[3/5] Heuristic adversarial...
Heuristic: DP=X.XXXX, Acc=X.XXXX (n=XXXXX)

[4/5] Fairness-Targeted PGD (gradient)...
Grad-PGD: DP=X.XXXX, Acc=X.XXXX (n=XXXXX)

--- Summary for adult ---
Clean:        DP=X.XXXX, Acc=X.XXXX
Random:       DP=X.XXXX, Acc=X.XXXX
Heuristic:    DP=X.XXXX, Acc=X.XXXX
Grad-PGD:     DP=X.XXXX, Acc=X.XXXX
DP Increase:  Clean→Grad: +X.XXXX

============================================================
Dataset: credit, alpha=0.2
============================================================
[... similar output ...]

============================================================
Dataset: lsac, alpha=0.2
============================================================
[... similar output ...]

============================================================
FINAL SUMMARY
============================================================

ADULT:
  Clean:      DP=X.XXXX
  Grad-PGD:   DP=X.XXXX
  Attack effect: +X.XXXX

CREDIT:
  Clean:      DP=X.XXXX
  Grad-PGD:   DP=X.XXXX
  Attack effect: +X.XXXX

LSAC:
  Clean:      DP=X.XXXX
  Grad-PGD:   DP=X.XXXX
  Attack effect: +X.XXXX

DONE: All tests complete
============================================================

INTERPRETATION:
- Clean DP = baseline unfairness in trained model
- Grad-PGD DP = unfairness after gradient-based attack
- Attack effect = Grad-PGD DP - Clean DP (should be POSITIVE)
- Positive = attack successfully increased unfairness (expected)
- Negative = something is wrong with attack implementation

SUCCESS CRITERIA:
- Grad-PGD DP > Clean DP for all datasets (attack works)
- Specifically:
  - Adult: +0.05 to +0.10 increase expected
  - Credit: +0.005 to +0.020 increase expected
  - LSAC: +0.02 to +0.05 increase expected

================================================================================
SECTION 8: ERROR HANDLING AND TROUBLESHOOTING
================================================================================

IF PACKAGE NOT FOUND ERROR:
----------------------------
Error: ModuleNotFoundError: No module named 'numpy'

Fix:
pip install numpy torch scikit-learn pandas scikit-image

If still fails:
pip install --upgrade pip
pip install numpy torch scikit-learn pandas scikit-image

Verify:
python3 -c "import numpy; print(numpy.__version__)"

IF PYTHONPATH ERROR:
--------------------
Error: ModuleNotFoundError: No module named 'src'

Fix:
export PYTHONPATH=/data/srujan.sai/DRO-FairML
echo $PYTHONPATH  # Verify it's set

Make permanent:
echo "export PYTHONPATH=/data/srujan.sai/DRO-FairML" >> ~/.bashrc

IF GIT PUSH FAILS:
------------------
Error: "Failed to push to remote" or "Permission denied"

Fix steps:
1. Check remote URL: git remote -v
   Should show: https://github.com/Srujan0798/DRO-FairML.git

2. Check network: ping github.com

3. Retry push:
   git push origin main
   (Wait 30 seconds if timeout)

4. If auth fails:
   git push https://github.com/Srujan0798/DRO-FairML.git
   (May need personal access token)

5. If still fails:
   Save results locally, try again next day

IF DATA FILES MISSING:
----------------------
Error: FileNotFoundError: adult.data not found

Fix:
ls -la /data/srujan.sai/DRO-FairML/data/raw/

If empty:
- Adult: Will auto-download on first run
- Credit: Will auto-download on first run
- LSAC: May need manual download

IF GPU NOT DETECTED:
--------------------
Error: CUDA not available or GPU count = 0

Check:
nvidia-smi  # Should show GPU info
python3 -c "import torch; print(torch.cuda.is_available())"

If GPU unavailable:
- Experiments will run on CPU (slower but works)
- Test script works on CPU
- Just proceed with CPU execution

IF SCRIPT HANGS OR TIMEOUT:
---------------------------
Error: Script running forever

Fix:
1. Press Ctrl+C to interrupt
2. Check for infinite loop (rare)
3. Re-run with timeout:
   timeout 600 python3 scripts/test_fairness_pgd.py

IF CUDA OUT OF MEMORY:
---------------------
Error: CUDA out of memory

Fix:
- Reduce batch size (edit scripts if needed)
- Clear GPU cache: torch.cuda.empty_cache()
- Use CPU: export CUDA_VISIBLE_DEVICES="" (last resort)

IF RESULTS LOOK WRONG:
----------------------
Check:
1. Random seed = 42 (should be in code)
2. Data splits are same each run
3. No data leakage (scaler fit only on train)
4. Corruption applied correctly
5. DP computation is correct

Contact Srujan if major discrepancy from expected values.

================================================================================
SECTION 9: AUTO RUNNER SCRIPT
================================================================================

SCRIPT: /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

This script automates Steps 1-11 from Section 6.

USAGE:
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

WHAT IT DOES:
1. cd to /data/srujan.sai/DRO-FairML
2. git pull origin main
3. Creates REPORTS/ and results/ directories
4. Runs test_fairness_pgd.py
5. Saves output to results/output_TIMESTAMP.txt
6. Appends summary to REPORTS/daily.txt
7. Commits changes with timestamp
8. Pushes to GitHub

PREFERRED WORKFLOW:
Instead of running commands manually, prefer:
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

This ensures consistent execution and proper result saving.

================================================================================
SECTION 10: EXPECTED EXPERIMENTAL RESULTS
================================================================================

Based on local testing (MacBook Air, no GPU):

ADULT DATASET:
- Clean DP: 0.14 (baseline)
- After Grad-PGD: 0.22 (+0.08 increase)
- Interpretation: Attack significantly increased unfairness

CREDIT DATASET:
- Clean DP: 0.01 (nearly fair)
- After Grad-PGD: 0.02 (+0.01 increase)
- Interpretation: Attack had small but measurable effect

LSAC DATASET:
- Clean DP: 0.02 (fair)
- After Grad-PGD: 0.05 (+0.03 increase)
- Interpretation: Attack notably increased unfairness

COMPARISON WITH OTHER METHODS:

| Method     | Adult DP | Credit DP | LSAC DP |
|------------|----------|-----------|---------|
| Clean      | 0.14     | 0.01      | 0.02    |
| Random     | 0.12     | 0.01      | 0.01    |
| Heuristic  | 0.06     | 0.01      | 0.01    |
| Grad-PGD   | 0.22     | 0.02      | 0.05    |

Note: Heuristic actually REDUCED DP on Adult (bad implementation).
Grad-PGD correctly INCREASES DP (our improvement).

WHY GRAD-PGD IS BEST:
- Only method that consistently INCREASES DP
- Analytical gradient ensures optimal attack
- Coordinated mode targets minority group (70% of budget)

================================================================================
SECTION 11: UTKFace PIPELINE (FUTURE WORK)
================================================================================

CURRENT STATUS: NOT YET RUN - GPU required, takes ~45 minutes

SCRIPT: scripts/extract_utkface_features.py

WHAT IT DOES:
1. Loads UTKFace images from /data/srujan.sai/UTKFace/
2. Passes each image through ResNet18 (pretrained on ImageNet)
3. Extracts 512-dimensional feature vector per image
4. Saves features to /data/srujan.sai/utkface_features.npz

WHY THIS IS NEEDED:
- Raw images are too large for direct training
- CNN features capture semantic information
- Pre-trained CNN (no fine-tuning) works as feature extractor

INPUT:
- UTKFace images in /data/srujan.sai/UTKFace/
- Format: {age}_{gender}_{race}_{date}.jpg.chip.jpg

OUTPUT:
- /data/srujan.sai/utkface_features.npz
  - X: (N, 512) feature matrix
  - age: (N,) array of ages
  - gender: (N,) array of gender labels (0/1)
  - race: (N,) array of race labels (0-4)

HOW TO RUN (when images are downloaded):
python3 scripts/extract_utkface_features.py \
    --data-dir /data/srujan.sai/UTKFace \
    --output /data/srujan.sai/utkface_features.npz \
    --batch-size 64

EXPECTED TIME: ~45 minutes on L40S GPU
EXPECTED OUTPUT: "Extracted features: X=(200000, 512)"

AFTER FEATURE EXTRACTION:
1. Load features: data = np.load('utkface_features.npz')
2. X = data['X']  # (200K, 512)
3. y = data['gender']  # or data['race']
4. a = data['race']  # or data['gender']
5. Train FairnessTargetedPGD experiments on image features

CURRENT PRIORITY:
- Run tabular experiments first (Adult, Credit, LSAC)
- UTKFace is secondary but planned

================================================================================
SECTION 12: GIT WORKFLOW AND VERSION CONTROL
================================================================================

REPOSITORY: https://github.com/Srujan0798/DRO-FairML
BRANCH: main (default, only branch we use)

DAILY GIT COMMANDS:

1. Get latest code:
   cd /data/srujan.sai/DRO-FairML
   git pull origin main

2. Check what changed:
   git status
   git log --oneline -5

3. After experiments, commit results:
   git add -A
   git commit -m "Friend: experiments on $(date +%Y%m%d)"
   git push origin main

4. If push fails:
   git push origin main
   (Retry, may need to wait for network)

5. Check commit history:
   git log --oneline -10

REMOTE CONFIGURATION:
git remote -v
# Should show:
# origin  https://github.com/Srujan0798/DRO-FairML.git (fetch)
# origin  https://github.com/Srujan0798/DRO-FairML.git (push)

BRANCHES:
- main: Primary branch, all work done here
- No other branches currently in use

================================================================================
SECTION 13: DAILY REPORT FORMAT
================================================================================

Write reports to: /data/srujan.sai/DRO-FairML/REPORTS/daily.txt

Append new entries, do NOT overwrite.

FORMAT:
================================
DATE: YYYY-MM-DD HH:MM:SS
EXPERIMENT: FairnessTargetedPGD on Adult, Credit, LSAC

RESULTS:
Adult: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX
Credit: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX
LSAC: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX

STATUS: SUCCESS / PARTIAL / ERRORS
NOTES: Any observations, issues, or things to investigate

COMMIT: [yes/no] - Whether code was pushed to GitHub

================================

Example entry:
================================
DATE: 2026-05-27 14:30:00
EXPERIMENT: FairnessTargetedPGD on Adult, Credit, LSAC

RESULTS:
Adult: Clean DP=0.1422, Corrupted DP=0.2206, Effect=+0.0784
Credit: Clean DP=0.0130, Corrupted DP=0.0234, Effect=+0.0104
LSAC: Clean DP=0.0176, Corrupted DP=0.0457, Effect=+0.0282

STATUS: SUCCESS
NOTES: Attack worked as expected on all datasets. DP increased significantly.
  Adult showed strongest effect (+0.08), LSAC showed notable effect (+0.03).
  All experiments completed in ~8 minutes on CPU.

COMMIT: yes

================================

================================================================================
SECTION 14: PROJECT ROLES AND COMMUNICATION
================================================================================

ROLES:

1. Srujan (Project Lead)
   - Develops and improves code
   - Reviews pushed results
   - Makes strategic decisions
   - Communicates with professor
   - Stays on local Mac, doesn't run server experiments

2. Friend (Server Operator)
   - Runs experiments on GPU server via JupyterHub
   - Types ONE LINE prompt to AI
   - AI executes all work autonomously
   - Reports results to Srujan
   - Does NOT develop code or make decisions

3. AI Agent (Autonomous Executor)
   - Reads FRIEND_GUIDE.md for full context
   - Executes all tasks on GPU server
   - Fixes errors autonomously
   - Commits and pushes to GitHub
   - Only asks for help if truly cannot fix

WORKFLOW:

1. Srujan develops code locally on Mac
2. Srujan pushes to GitHub
3. Friend gives AI the one-line prompt
4. AI pulls from GitHub, runs experiments, saves results
5. AI pushes results to GitHub
6. Srujan reviews pushed code and results
7. Srujan makes improvements, pushes new code
8. Repeat

COMMUNICATION:

Friend → Srujan (WhatsApp/Discord):
- "Experiments done, DP results: Adult 0.22, Credit 0.02, LSAC 0.05"
- "Error occurred: GPU not found" (rare)
- "No results for 2 hours, something might be wrong"

Srujan → Friend (WhatsApp/Discord):
- "New code pushed, run experiments"
- "Check the new FRIEND_GUIDE.md for updates"
- "UTKFace images arrived, can start feature extraction"

================================================================================
SECTION 15: COMPLETE COMMAND REFERENCE
================================================================================

# ACCESS SERVER
# Open browser: http://flair2.iitgn.ac.in:8000/hub/login
# Login: srujan.sai / ss#081

# NAVIGATE
cd /data/srujan.sai/DRO-FairML
pwd

# GET LATEST CODE
git pull origin main

# SET PYTHONPATH
export PYTHONPATH=/data/srujan.sai/DRO-FairML

# INSTALL PACKAGES (if needed)
pip install numpy torch scikit-learn pandas scikit-image

# CREATE OUTPUT DIRECTORIES
mkdir -p /data/srujan.sai/DRO-FairML/REPORTS
mkdir -p /data/srujan.sai/DRO-FairML/results

# RUN EXPERIMENTS (manual)
python3 scripts/test_fairness_pgd.py

# RUN EXPERIMENTS (with output saved)
python3 scripts/test_fairness_pgd.py > results/output_$(date +%Y%m%d_%H%M%S).txt 2>&1

# RUN AUTO RUNNER (automated - does everything)
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

# SAVE TO REPORTS
echo "Date: $(date)" >> REPORTS/daily.txt
tail -30 results/output_*.txt >> REPORTS/daily.txt

# COMMIT AND PUSH
git add -A
git commit -m "Friend: experiments on $(date +%Y%m%d)"
git push origin main

# CHECK RESULTS
cat REPORTS/daily.txt
ls -la results/
cat results/output_*.txt

# CHECK GPU
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# VERIFY PACKAGES
python3 -c "import numpy; import torch; import sklearn; print('OK')"

# CHECK GIT STATUS
git status
git log --oneline -5

================================================================================
SECTION 16: QUICK REFERENCE CARD
================================================================================

FRIEND'S ONE LINE TO AI:
-------------------------
Do the DRO-FAIR research work. Repo: https://github.com/Srujan0798/DRO-FairML. 
Login: http://flair2.iitgn.ac.in:8000/hub/login (user: srujan.sai, pass: ss#081). 
Server path: /data/srujan.sai/DRO-FairML. Run experiments, save results in REPORTS/, 
commit and push to GitHub when done. If anything fails, fix it yourself. 
Only tell me if you truly cannot fix it.

AI DOES EVERYTHING ELSE:

1. Login to JupyterHub (http://flair2.iitgn.ac.in:8000/hub/login)
2. cd /data/srujan.sai/DRO-FairML
3. git pull origin main
4. pip install packages (if needed)
5. export PYTHONPATH=/data/srujan.sai/DRO-FairML
6. python3 scripts/test_fairness_pgd.py
7. Save results to REPORTS/
8. git add -A && git commit && git push
9. Write summary report

FRIEND'S LAPTOP DOES: NOTHING (just displays results)
GPU SERVER DOES: ALL computation

================================================================================
SECTION 17: IMPORTANT NOTES FOR AI AGENTS
================================================================================

CRITICAL REMINDERS FOR AI EXECUTING THIS WORK:

1. Friend's laptop is just a terminal - all computation is on GPU server
2. Never ask friend to type commands beyond the one line prompt
3. Fix errors yourself - only escalate if truly cannot resolve
4. Always save results before pushing to GitHub
5. Append to REPORTS/daily.txt, never overwrite
6. Verify GPU is available but don't require it (CPU works too)
7. If script fails, diagnose and retry (check imports, paths, data)
8. Commit with timestamp: "Friend: experiments on YYYYMMDD"
9. Verify push succeeded, retry if failed
10. Write clear summary of what was done for Srujan to review

WHAT TO DO IF STUCK:
- Package import fails → pip install
- Git push fails → retry after 30s, max 3 times
- Script hangs → Ctrl+C, check errors, re-run
- Data missing → check path, auto-download if possible
- GPU not found → use CPU, continue anyway

WHAT TO DO IF TRULY CANNOT FIX:
- Explain clearly what failed
- Explain what you tried
- Ask friend to inform Srujan
- Save any partial results locally

DO NOT:
- Ask friend to edit code
- Ask friend to debug
- Ask friend to run commands manually (beyond initial prompt)
- Give up without trying multiple fixes
- Leave results unsaved

================================================================================
SECTION 18: PROJECT HISTORY AND DEVELOPMENT
================================================================================

COURSE PROJECT COMPLETED:
- DRO-FAIR framework implemented
- Tested on Adult, Credit, LSAC datasets
- Fairness metrics: DP, IF, EO
- DRO training:ERM with fairness constraints

RESEARCH EXTENSION (Current Work):
- New attack: FairnessTargetedPGD (gradient-based)
- Better than heuristic attacks (verified on local Mac)
- Need to run at scale on GPU server
- Need to test on UTKFace image data (future)

KEY MILESTONES:
- [DONE] FairnessTargetedPGD class implemented
- [DONE] Test script written and verified locally
- [DONE] UTKFace pipeline designed
- [IN PROGRESS] Run experiments on FLAIR2 server
- [PENDING] UTKFace feature extraction
- [PENDING] Image-based fairness experiments
- [PENDING] DRO-FAIR training with defense

LOCAL TEST RESULTS (MacBook Air, no GPU):
Adult:  Clean DP=0.14, Grad-PGD DP=0.22 (+0.08)
Credit: Clean DP=0.01, Grad-PGD DP=0.02 (+0.01)
LSAC:   Clean DP=0.02, Grad-PGD DP=0.05 (+0.03)

EXPECTED ON SERVER: Similar results (computation is CPU-based anyway)

================================================================================
END OF FRIEND_GUIDE.md
================================================================================
THIS FILE IS READ BY AI TO UNDERSTAND FULL PROJECT CONTEXT
FRIEND TYPES ONE LINE, AI DOES EVERYTHING ELSE
================================================================================