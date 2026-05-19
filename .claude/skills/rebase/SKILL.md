---
name: rebase
description: Rebase the current working branch onto local `main` (NOT `origin/main`), resolving every conflict by consulting the authoritative RPython/PyPy source under `` rather than arbitrarily picking a side. Invoked via `/rebase`. Use whenever the user asks to "rebase", "rebase onto main", "restack on main", "replay on top of main", "pull main under my branch", or any equivalent request to rewrite the current branch on top of main while resolving conflicts. Before rebasing, this skill commits any unstaged work and squashes WIP noise into concise commits. During rebase, conflicts are resolved per-hunk — commit-level granularity is explicitly too coarse; each conflicting region is judged individually against upstream RPython/PyPy and merged to the more RPython-orthodox shape. Default to using this skill any time the user mentions rebasing, pulling main under the branch, or conflict resolution in the context of a long-running branch. Complements but does not replace `/merge` (which builds a green `-merge` branch) and `/parity` (which defines "RPython-orthodox").
---

# Rebase onto local main, conflict-by-conflict, RPython-first

## Why this skill exists

The working branch has drifted from `main` over time. Rebasing pulls `main`'s progress under our work so that subsequent merging, reviewing, and landing operate against an up-to-date base. Two things make this rebase non-routine:

1. **The base is `main`, not `origin/main`.** `main` in this worktree is authoritative and may be ahead of `origin/main` (the user syncs manually). Using `origin/main` silently rebases onto a stale base.
2. **Conflicts are not 50/50 choices.** A conflict usually means both sides edited the same region with different intents. The right answer is almost always "whichever side matches RPython/PyPy upstream more faithfully" — sometimes that's ours, sometimes it's main's, and sometimes it's a blend taking individual lines from both. Picking a side wholesale loses information.

Commit boundaries do not reliably align with these semantic units. A single commit can contain five unrelated hunks; one hunk in that commit may belong to us and four to main. Therefore conflict resolution must operate on the hunk/region level, not on `git checkout --ours` / `--theirs` at file level.

## Invocation

When the user types `/rebase`:

1. Parse any follow-up text as scoping hints (e.g. `/rebase finish what we started`, `/rebase skip the tests for now`). Default is "rebase the current branch onto local main, resolve every conflict".
2. Do the prep work in Step 1 before touching `git rebase`.
3. Execute the rebase per Steps 2–5.

## Step 1 — Prepare the working tree

Goal: enter the rebase with a clean, conceptually-coherent commit history. Messy history makes conflict resolution harder because unrelated hunks pile up inside the same rebase step.

1. Check state:

   ```bash
   git status --short
   git log main..HEAD --oneline
   git rev-parse --verify main
   ```

2. **Unstaged / untracked changes** — if there is real uncommitted work, commit it. Use the `commit` convention from CLAUDE.md: factual English message, no `Co-Authored-By`, no speculation about goals. If the changes are clearly WIP/scratch that should not land, confirm with the user before discarding or stashing.

3. **Squash WIP noise** — if `git log main..HEAD` shows obvious fixups, revert-my-last-change, "typo", "trying X" commits, consolidate them with `git rebase -i main` (**before** the real rebase) or `git reset --soft` + fresh commits. Rationale: during the real rebase, each small WIP commit will re-apply and may re-introduce conflicts that were already resolved one commit later. Squashing these out collapses the noise.

   Squashing is *not* for consolidating meaningful commits into one giant commit. Keep commits that represent distinct concepts separate — it makes per-hunk conflict reasoning easier, because each step's diff stays narrow.

4. Confirm the squash plan with the user before rewriting history, especially if the branch is shared.

5. Before invoking `git rebase`, ensure `main` exists locally and is at the tip the user intends. Do **not** run `git fetch origin main` and rebase on `origin/main` — the user may have a local `main` that is ahead of origin, and overwriting it is a silent loss.

## Step 2 — Start the rebase

```bash
git rebase main
```

That's it — local `main`, no `origin/`. If the rebase completes cleanly with no conflicts, skip to Step 5.

If the rebase refuses to start because of a dirty working tree, go back to Step 1.

## Step 3 — Resolve each conflict per hunk, not per file or commit

When `git rebase` stops on a conflict, a single invocation may have produced multiple conflicted files, and each file may have multiple `<<<<<<<` / `=======` / `>>>>>>>` regions. **Each region is judged independently.** Do not resolve them all the same way; do not run `git checkout --ours` / `--theirs`; do not accept IDE "Take Current" / "Take Incoming" bulk actions.

For each conflict region:

1. **Identify the semantic question.** What did the current side (HEAD / main in a rebase — git flips ours/theirs during rebase; check `git rebase --show-current-patch` if unsure) change, and what did the incoming side (our branch's commit) change? State both intents in one sentence each before choosing.

2. **Locate the RPython/PyPy counterpart.** Use the mechanical rule documented in the `/parity` skill — strip the crate root, swap `.rs` for `.py`, prepend the upstream root at ``. Open the upstream file. Read the region that corresponds to the conflicted region.

   If the mechanical rule fails to produce a valid path, read the `/parity` skill's "When the mechanical transform fails" section — the failure itself may indicate that one side of the conflict has drifted away from upstream structure, which is a strong signal about who to side with.

3. **Judge RPython-orthodoxy per region.** Lean toward the side whose:
   - Structure matches the RPython file line-by-line (identifier names, argument order, control flow).
   - Dependencies (helper functions, types, fields referenced) exist in upstream.
   - Comments match or cite upstream.

   Against the side whose:
   - Names do not appear anywhere in `rpython/` or `pypy/` under the project root.
   - Added side tables, caches, or feature flags without upstream basis (see `/parity`'s "Signals of NEW-DEVIATION").
   - Comments say `// workaround`, `// pyre-only`, `// adaptation` without a cited upstream line.

4. **Blend when both sides are partially orthodox.** It is common that main fixed a real bug in the conflicted region and our branch renamed a variable in the same region. In that case the resolution is main's fix + our rename, not one or the other. Write the blended region by hand in the editor — do not rely on any conflict marker to auto-resolve a blend.

5. **If neither side is orthodox**, fix the region to match upstream even if that means writing code different from both sides. A rebase is a legitimate moment to fix a pre-existing deviation in a hunk you had to touch anyway. Don't expand scope beyond the conflicting region, though — an unrelated clean hunk is out of scope for this skill.

6. **Cite the upstream line(s) consulted** in a brief note the user will see when you summarize. This is how the user audits your conflict resolution after the fact without re-running the rebase.

7. Remove the `<<<<<<<` / `=======` / `>>>>>>>` markers and save.

After every region in a file is resolved, `git add` the file. Only after every conflicted file in the current rebase step is added, continue with Step 4.

## Step 4 — Verify, then continue the rebase

Before `git rebase --continue`:

1. Re-read the diff for this step to catch accidental deletions:

   ```bash
   git diff --cached
   ```

2. For each resolved file, sanity-check that the logic compiles and matches upstream structure. If the file is Rust and the change is non-trivial, `cargo check -p <crate>` on the affected crate is cheap and catches obvious breakage before the next conflict compounds the damage. Full `cargo test` is overkill between steps — save that for Step 5.

3. Continue:

   ```bash
   git rebase --continue
   ```

4. If more conflicts appear, return to Step 3. Each conflict region gets the same RPython-first treatment.

5. If at any step the rebase resolution is genuinely ambiguous and upstream gives no guidance, **stop and ask the user** rather than guess. Partial rebases can be stashed and resumed; a wrong silent resolution propagates through the rest of the rebase.

If things go badly wrong:

```bash
git rebase --abort     # returns to pre-rebase state; no work lost
```

`--abort` is safe — it's the designed undo. Prefer aborting and restarting over `git rebase --skip` of a commit you don't understand.

## Step 5 — Final sanity check

After the rebase completes:

1. `git log main..HEAD --oneline` — confirm the commit list is what was expected, and that the history is linear on top of `main`.
2. `python ./pyre/check.py` — end-to-end verification. A clean compile does not prove the resolutions were correct; runtime tests do.
3. `cargo test --all` on touched crates if the rebase was large.
4. Report to the user:
   - Number of conflicts resolved.
   - For each non-trivial resolution: file:line on our side, file:line upstream consulted, one-sentence justification.
   - Whether `pyre/check.py` is green.

Do **not** push, merge, or force-push unless the user explicitly asks. CLAUDE.md rule.

## Hard rules (recap)

- Base is `main`, never `origin/main`.
- Every conflict region is judged individually. No bulk `--ours` / `--theirs`, no IDE "take all current", no trust-one-side-per-file shortcuts.
- Every non-trivial resolution cites the upstream RPython/PyPy file:line it was judged against.
- Commit any unstaged work (or explicitly stash with user consent) before starting the rebase.
- Squash WIP noise only; do not collapse meaningful commits into one.
- Never `git push --force` without explicit user instruction, even after a successful rebase.
- When in doubt between two plausible resolutions, stop and ask the user — wrong silent resolutions compound.

## Interaction with other skills

- **`/parity`** — defines "RPython-orthodox" in detail (mechanical upstream path lookup, NEW-DEVIATION signals, structural vs functional parity). Read its body when a conflict requires judging orthodoxy and the summary above is insufficient.
- **`/merge`** — separate workflow. `/merge` accumulates green slices on a `<branch>-merge` branch starting from `main`; `/rebase` rewrites the current branch on top of `main` in place. They are not interchangeable. Use `/rebase` when the goal is "my branch should be linear on top of current main"; use `/merge` when the goal is "I need a staging branch where `pyre/check.py` is always green".
- **`/commit`** — the commit convention (factual English, no `Co-Authored-By`, no "improves X" speculation) applies to any commits this skill creates during Step 1 prep.
