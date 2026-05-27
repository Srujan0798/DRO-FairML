# DRO-FAIR RESEARCH - FRIEND'S LAPTOP DOES NOTHING!

## ⚠️ IMPORTANT: friend's laptop is just a WINDOW to the GPU server

Friend's laptop is ONLY used to login to JupyterHub. That's it!

ALL actual work runs on FLAIR2 GPU SERVER. Friend's laptop will NOT:
- Heat up
- Use GPU
- Use CPU
- Make fan noise
- Work hard

Everything happens on the GPU server - friend's laptop just shows the results.

---

## ONE LINE PROMPT FOR FRIEND:

```
Do the DRO-FAIR research work. Repo: https://github.com/Srujan0798/DRO-FairML. Login: http://flair2.iitgn.ac.in:8000/hub/login (user: srujan.sai, pass: ss#081). Server path: /data/srujan.sai/DRO-FairML. Run experiments, save results in REPORTS/, commit and push to GitHub when done. If anything fails, fix it yourself. Only tell me if you truly cannot fix it.
```

---

## WHAT HAPPENS (Technical truth):

1. Friend's laptop opens a web browser → goes to JupyterHub
2. JupyterHub is running on FLAIR2 GPU SERVER (not friend's laptop)
3. Friend opens Terminal in JupyterHub
4. AI connects to the GPU server's terminal
5. AI runs commands ON THE GPU SERVER (not friend's laptop)
6. GPU server does all computation using its 2x Nvidia L40S GPUs (48GB each)
7. Results shown in JupyterHub → which is displayed on friend's laptop
8. Friend's laptop only shows the results - does zero computation

## GPU SERVER SPECS (where ALL work happens):
- CPU: 2x Intel processors
- GPU: 2x Nvidia L40S, 48GB memory EACH
- RAM: 128GB
- OS: Ubuntu 22.04 LTS
- Location: FLAIR2 server room at IIT Gandhinagar

## Friend's laptop specs needed:
- Web browser (Chrome/Firefox)
- Internet connection
- That's it!

---

## WHAT FRIEND DOES (Takes 30 seconds):

1. Open Chrome/Firefox on laptop
2. Go to: http://flair2.iitgn.ac.in:8000/hub/login
3. Login: user `srujan.sai` pass `ss#081`
4. Click: New → Terminal
5. Paste the ONE LINE prompt above
6. Press Enter
7. Walk away - everything runs on GPU server

---

## HOW IT WORKS:

```
FRIEND'S LAPTOP (just browser) 
        ↓
    JupyterHub login
        ↓
    Terminal opened
        ↓
    AI runs commands HERE (on GPU server)
        ↓
    GPU does all computation
        ↓
    Results shown in browser
        ↓
    Friend's laptop shows results - did nothing!
```

---

## YOU: Just give the one line prompt to friend and walk away

AI will handle everything on the GPU server. Your friend's laptop will stay cool and quiet.