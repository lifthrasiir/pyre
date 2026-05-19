---
name: merge
description: Plan and execute a safe merge of the current working branch into main by accumulating only green-test changes onto a separate `<branch>-merge` branch. Invoked via `/merge`. Use this skill whenever the user asks to "merge", "land", "prepare for merge", "cherry-pick safely", or any equivalent request to move work from a long-running branch toward main. The skill enforces the house policy that `pyre/check.py` must be all green, and it prefers line-level imports over whole-commit cherry-picks (bare `git cherry-pick` is only safe for trivially conflict-free, self-contained commits). Default to using this skill any time the user hints at moving work from a long-running branch toward main, even without explicit keywords.
---

# Safe Merge via `-merge` Branch

## Why this skill exists

The working branch has accumulated many commits; some of them break `pyre/check.py`. House policy is:

**`pyre/check.py` must be all green — no exceptions — before anything lands on main.**

Cherry-picking by commit almost always breaks the build because commits depend on each other and on intermediate states that never made it through review. So we never move whole commits. We move **code**, one small slice at a time, and run the tests after each slice.

Keep the original working branch untouched — it remains the source of truth for "what we eventually want". The `-merge` branch is a clean, always-green staging area that starts at main and only grows with verified slices.

## Invocation

When the user types `/merge`:

1. Parse any follow-up text as scoping hints (e.g. `/merge only the heapcache changes`, `/merge focus on blackhole.rs`). Default is "work through everything safely".
2. Check current state and plan.
3. Execute the plan with the user.

## Step 1 — Establish the `-merge` branch

Determine:
- `WORK` = the current branch (e.g. `stdlib`)
- `MERGE` = `<WORK>-merge` (e.g. `stdlib-merge`)
- `BASE` = `main`

Run:

```bash
git branch --show-current            # confirm WORK
git rev-parse --verify main          # confirm BASE exists
git rev-parse --verify <MERGE> 2>/dev/null   # does MERGE already exist?
```

If `<MERGE>` does **not** exist:
- Create it at `main`: `git branch <MERGE> main`
- Do **not** check it out yet unless the user wants to start immediately. Confirm with the user before switching.

If `<MERGE>` already exists:
- Verify it is a descendant of (or equal to) `main`. If not, surface the divergence and ask the user whether to reset it to `main` or continue from its current tip.
- Check whether `pyre/check.py` passes on `<MERGE>` before adding anything.

State the plan concisely before moving code: which branch is source, which is destination, and what the user wants to land first.

## Step 2 — Survey what needs to move

Work from the current state of both branches:

```bash
git diff <MERGE>..<WORK> --stat      # what's left to bring over
git diff main..<MERGE> --stat        # what's already landed safely
```

`git diff main <MERGE>` is how we track progress — it grows with each safe slice. Show the user the remaining diff bucketed by file/area so they can point at what to try first.

## Step 3 — Prefer line-by-line slices; cherry-pick only when trivially clean

Default to **line-by-line** imports. The reason is that commits on `WORK` tend to:

- Depend on earlier broken states, scaffolding, or partial refactors that never land.
- Mix correctness fixes (safe) with speculative behavior changes (unsafe) in a single commit.
- Split renames and follow-up cleanups across commits — taking only one leaves dangling references.

That said, `git cherry-pick` is acceptable for a commit that meets **all** of:

- Applies cleanly (no conflict, no `--strategy` tricks).
- Is self-contained — touches one concern, with no implicit dependency on other un-landed commits.
- Has been inspected hunk-by-hunk, not just by subject line.

If any of those fail, fall back to line-by-line. When in doubt, line-by-line is always safe; cherry-pick is an optimization.

Safe slicing procedure (line-by-line path):

1. Pick a small, self-contained change from `git diff <MERGE>..<WORK>` for one file. Prefer:
   - A single added/renamed helper function with no new call sites yet.
   - A bugfix hunk that does not depend on new types or new fields.
   - A pure refactor (e.g. extracted local, renamed local variable) verifiable by inspection.
   - Comment, doc, or dead-code removal.
2. Write the change **by hand** on `<MERGE>` — read the source from `WORK`, but apply it with `Edit`/`Write` onto `<MERGE>`. Do not use `git checkout <WORK> -- <file>` for partial changes; that grabs the whole file and usually breaks things.
3. For a clean whole-file import where inspection confirms the file is self-contained (e.g. an isolated new module with no existing callers on main), `git checkout <WORK> -- <path>` is acceptable, but verify with the test run immediately after.
4. Before committing, re-read the diff against main (`git diff main -- <file>`) and confirm only the intended lines moved.

## Step 4 — Test every slice

Run `python ./pyre/check.py` after every slice, no matter how small. This is non-negotiable — the whole point of the `-merge` branch is that it is always green.

```bash
python ./pyre/check.py
```

Outcomes:

- **All green** → commit immediately, then pick the next slice.
- **Red** → do NOT commit. Either:
  - Shrink the slice further (most common fix — revert part of the change and retry).
  - Pull in the missing dependency (a helper, a type, a rename) from `WORK` first, then re-test.
  - If the change is irreducibly unsafe on its own, mark it as "deferred" and move on to a different slice. Tell the user which slice was deferred and why.

Never "commit and fix later". `<MERGE>` must stay green at every commit.

## Step 5 — Commit on success

When `pyre/check.py` is green, commit with a factual message (CLAUDE.md convention — no speculation about goals, no `Co-Authored-By`, English only):

```bash
git add <files>
git commit -m "<concise factual summary>"
```

Then loop back to Step 3 with the next slice.

Periodically re-run:

```bash
git diff <MERGE>..<WORK> --stat
```

to see what is left. The goal is to drive this diff toward empty, one green commit at a time.

## Step 6 — Wrap-up

When the user decides to stop (either the diff is empty or the remaining work is not safely landable):

- Report what landed on `<MERGE>` (`git log main..<MERGE> --oneline`).
- Report what is deferred, per-file, with the reason it could not land.
- Confirm the original `WORK` branch is untouched (`git log <WORK>` unchanged vs. when we started).
- Do **not** push, merge, or open a PR unless the user explicitly asks — CLAUDE.md rule.

## Hard rules (recap)

- Prefer line-by-line slices. `git cherry-pick` is allowed only for trivially conflict-free, self-contained commits that have been inspected hunk-by-hunk.
- Never commit on `<MERGE>` without a green `pyre/check.py` immediately before.
- Never modify `WORK` from this skill — it is read-only here.
- Never push `<MERGE>` or open a PR without explicit user instruction.
- Always commit after each green test — do not batch multiple slices into one commit, because a later red test cannot be bisected if slices are batched.
