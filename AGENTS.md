# Project rules — DP-FL (Non-IID Dirichlet split, Opacus DP-SGD, Flower, RDP accounting)

## Restart rule
This repo does not get deleted. If a stage runs slow or breaks, debug in place — a slow run
is a solvable problem (see the small-subset workflow below), not a reason to start over.

## Small-subset-first (mandatory for every stage)
Every stage must be built and verified on a tiny subset first: ~1,000 CIFAR-10 images, 2–3
simulated clients, 2 rounds, small batch size. Do not run on full CIFAR-10 or the full client
count until the subset version runs correctly end-to-end. This is non-negotiable for Stage 4
(Opacus) specifically — per-sample gradient computation is slow by nature; a GPU (Colab free
tier) plus small batches plus subset-first is the fix, not a rewrite.

## Autonomous git workflow — do this without asking
After every meaningfully complete unit of work (a working function, a passing checkpoint, a
fixed bug — NOT every file save), automatically:
1. `git add <the specific files that changed>` — never blind `git add -A` unless you've
   confirmed there's nothing untracked you don't want (check `.gitignore` covers venv/, data/,
   __pycache__/, *.pt, wandb/ etc. first).
2. Write the commit message to a temp file, then `git commit -F <tempfile>` — never
   `git commit -m "..."`. This avoids quote-related terminal approval issues.
3. `git push`.
Do not ask for confirmation before doing this. Do not describe the commands you're about to run
and wait — just run them, then tell me what you committed in one line.

## Commit message style — write like the person doing the work, not a changelog
- Lowercase start, no period at the end, imperative mood: `add dirichlet split visualization`,
  not `Added Dirichlet Split Visualization.`
- No conventional-commit prefixes (`feat:`, `fix:`, `chore:`). Just say what happened:
  `fix flower client not aggregating on 3rd round`, `bump batch size down, opacus was OOMing`,
  `wip on rdp accountant, epsilon looks off`, `typo in requirements.txt`.
- Small, frequent, honest commits — including "in progress" and "fix previous commit" ones.
  Real research repos are not one giant commit at the end; they're a messy, readable history.
  Don't batch a whole stage into one commit.
- It's fine to have a commit that just fixes something you broke two commits ago. Don't rewrite
  history to hide it.

## Learn track
Skipped for now — move fast through BUILD, no checkpoint questions, no waiting for
confirmation before proceeding. Still log a one-line note of what was built at each stage
for later reference.

## Reproducibility
Seed torch/numpy/random from Stage 0 onward. Every run's hyperparameters (σ, C, α, δ) must be
loggable and traceable — if I can't explain how a resulting ε was calculated, treat the run as
invalid per the original project spec, and fix the logging before moving on, not after.
