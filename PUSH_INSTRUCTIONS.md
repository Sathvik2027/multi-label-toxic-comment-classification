# Push to YOUR existing GitHub repository

**Important:** This is a prepared starter copy; it does NOT modify or push anything to GitHub automatically. Review all files, ownership and privacy before upload.

## If your repository already exists (recommended)

1. Clone your own repo onto your computer: `git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git`.
2. Copy the **contents** of this starter folder into that clone (not its parent folder). Do not overwrite existing work without checking diffs.
3. Run: `python scripts/check_public_repo.py`.
4. Run: `git status --short` and inspect exactly what you are about to upload.
5. Add controlled paths (avoid blindly uploading all downloads):
   `git add README.md .gitignore requirements.txt docs/ data/README.md src/ notebooks/ results/ scripts/ PUSH_INSTRUCTIONS.md`
6. Check: `git diff --cached --name-only` and `git diff --cached --stat`.
7. Commit: `git commit -m "Document NLP-04 baselines, transformer runs and research audits"`.
8. Push: `git push -u origin HEAD`. If collaborating, prefer a separate feature branch + pull request, not pushing directly to shared main.

## If no repo exists

Create an empty GitHub repo (do not pre-create a README), then in this folder run:
```
git init
git branch -M main
git add README.md .gitignore requirements.txt docs/ data/README.md src/ notebooks/ results/ scripts/ PUSH_INSTRUCTIONS.md
git diff --cached --name-only
git commit -m "Initial NLP-04 research milestone"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```
Do not run the `remote add` line if origin already exists. Never force-push to resolve a conflict.

## Continue after Phase 2
Commit new **executed** experiment code and safe summary CSVs with a separate commit, e.g. `git commit -m "Add controlled sampling and fusion ablations"`.

## Review before pushing
- Scan notebook **sources** for credentials, private emails, and hard-coded sensitive text; notebook **outputs are already stripped** in this copy.
- The original `src` modules and notebook code were not rewritten and still depend on your original Drive project setup.
- Avoid GitHub upload of private or paid documents, team members' proprietary work, and any model checkpoints without explicit decisions about licensing/storage.
- The results folder contains historical aggregate scores; no Phase-2 result is claimed yet.
