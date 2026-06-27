# 🚀 How to Upload This Project to GitHub

Follow these steps exactly — you'll be done in 10 minutes!

---

## STEP 1 — Create a GitHub Account (if you don't have one)
1. Go to https://github.com
2. Click **Sign up** → enter your email, password, username
3. Verify your email

---

## STEP 2 — Create a New Repository
1. After logging in, click the **"+"** icon (top right) → **New repository**
2. Fill in:
   - **Repository name:** `stock-price-predictor`
   - **Description:** `ML project to predict stock prices using Linear Regression, Random Forest, and LSTM`
   - Set to **Public**
   - ✅ Check **"Add a README file"** — NO (we already have one)
3. Click **"Create repository"**
4. You'll see a page with a URL like: `https://github.com/your-username/stock-price-predictor`

---

## STEP 3 — Install Git (if not already installed)
- **Windows:** Download from https://git-scm.com/download/win → install with defaults
- **Mac:** Open Terminal and run: `git --version` (it will auto-install if needed)
- **Linux:** `sudo apt install git`

---

## STEP 4 — Upload the Project Files

Open **Terminal** (Mac/Linux) or **Git Bash** (Windows) and run these commands one by one:

```bash
# Navigate to the project folder
cd path/to/stock-price-predictor

# Initialize git
git init

# Add all files
git add .

# Commit with a message
git commit -m "Initial commit: Stock Price Predictor project"

# Connect to your GitHub repo (replace YOUR-USERNAME with your actual username)
git remote add origin https://github.com/YOUR-USERNAME/stock-price-predictor.git

# Push to GitHub
git branch -M main
git push -u origin main
```

When asked for username/password:
- Username: your GitHub username
- Password: use a **Personal Access Token** (not your password)

---

## STEP 5 — Create a Personal Access Token (GitHub Password Alternative)
1. Go to GitHub → click your profile photo → **Settings**
2. Scroll down → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
3. Click **Generate new token (classic)**
4. Give it a name, set expiry to **90 days**, check **"repo"** scope
5. Click **Generate token** → **COPY IT NOW** (you won't see it again!)
6. Use this token as the password when Git asks

---

## STEP 6 — Verify Upload
1. Go to `https://github.com/YOUR-USERNAME/stock-price-predictor`
2. You should see all your files!
3. The README will display nicely on the main page

---

## ✅ Final Checklist
- [ ] README.md shows on the repo homepage
- [ ] `stock_predictor.py` is uploaded
- [ ] `requirements.txt` is uploaded
- [ ] `data/` and `models/` folders are visible

**Share this link with your manager:** `https://github.com/YOUR-USERNAME/stock-price-predictor`
