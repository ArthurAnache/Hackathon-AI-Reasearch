# Team Git Collaboration Rules

These instructions apply to every coding agent and every teammate working in this repository. Follow them whenever
you inspect, edit, commit, pull, merge, or push code.

The default branch of this repository is `main` (checked with `git symbolic-ref refs/remotes/origin/HEAD`).
Commit messages follow the format in `CLAUDE.md` (Conventional Commits + version tag), which takes precedence over the
generic examples below.

## 1. Shared collaboration model

- The default branch (`main`) is the shared integration branch.
- Do not work directly on the default branch. Create a separate branch for each task.
- Use descriptive branch names, for example:
  - `feature/navigation`
  - `feature/vision-pipeline`
  - `fix/config-validation`
  - `docs/setup-guide`
- Publish work on the task branch and integrate it through a GitHub pull request.
- Do not assume separate files guarantee conflict-free work. Shared interfaces, imports, dependencies, configuration,
  and data formats can still overlap.
- Each teammate must use their own authorized GitHub account and credentials.

## 2. Inspect before acting

Before changing Git state or files, inspect:

```bash
git status --short --branch
git branch --show-current
git remote -v
git log -5 --oneline
```

- Identify the repository's actual default branch; do not assume it is named `main`.
- Check for uncommitted changes before switching branches, pulling, rebasing, resetting, or cleaning.
- Never discard, overwrite, reset, or silently stage another person's uncommitted work.
- If unrelated uncommitted changes exist, leave them untouched. If the next operation could disrupt them, stop and ask
  the user how to proceed.
- Do not expose or commit `.env` files, credentials, API keys, access tokens, private keys, or other secrets.

## 3. Start a task on its own branch

If the working tree is clean and the user is on the default branch:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feature/short-task-name
```

Replace `main` with the repository's actual default branch.

If the user is already on a task branch, do not create another branch without a reason. Check whether the branch is
shared with other teammates before rebasing it.

## 4. Keep changes focused

- Modify only files needed for the assigned task.
- Do not reformat, rename, or refactor unrelated files.
- Do not edit files another teammate is actively changing without coordinating.
- Before changing shared interfaces, APIs, schemas, dependency manifests, build settings, or central entry points,
  identify the impact and coordinate with affected teammates.
- Check `.gitignore` before staging new files.
- Do not add generated output, local environment files, machine-specific settings, or secrets unless the project
  explicitly requires them.

## 5. Fetch and integrate remote work safely

Fetch remote updates without changing the working tree:

```bash
git fetch origin
git status --short --branch
```

To update an individual, unpublished task branch with the latest default-branch changes:

```bash
git fetch origin
git switch feature/short-task-name
git rebase origin/main
```

Replace `main` with the actual default branch.

- Ensure the user's work is committed or safely preserved before rebasing.
- Rebase only an individual branch that is not being actively shared by multiple people, unless the team explicitly
  agrees.
- Do not run `git pull` blindly. Fetch first, then choose the intended integration operation.
- Never use `git reset --hard`, `git clean -fd`, or force-push as routine synchronization or conflict-resolution steps.

## 6. Resolve conflicts without losing work

If a merge or rebase reports conflicts:

1. Stop and inspect `git status`.
2. Read each conflicted file and understand both versions.
3. Preserve both sides' intended behavior where possible.
4. Never automatically choose ours or theirs.
5. If the correct resolution is unclear, ask the user or the teammate who owns the affected code. Do not guess.
6. After resolving, run relevant tests, lint, and build checks.
7. Stage only resolved files and continue:

   ```bash
   git add path/to/resolved-file
   git rebase --continue
   ```

For a merge, complete the merge only after all conflicts are correctly resolved. To abandon an operation, use
`git rebase --abort` or `git merge --abort` as appropriate, after explaining the consequences if user work may be
affected.

## 7. Review before committing

Before committing:

```bash
git status --short
git diff
git diff --cached
```

- Review the complete diff.
- Confirm that every staged change belongs to the current task.
- Stage explicit paths instead of using `git add .` or `git add -A` by default:

  ```bash
  git add path/to/file1 path/to/file2
  ```

- Use a concise, descriptive commit message in the format defined in `CLAUDE.md`:

  ```bash
  git commit -m "feat(ws2): add obstacle-aware navigation planner (V1.3)"
  ```

- Do not amend or rewrite a commit that has already been pushed unless the user explicitly requests it and the team
  agrees.

## 8. Push only the task branch

Before pushing:

```bash
git branch --show-current
git status --short
git log -5 --oneline
```

- Confirm the current branch is the intended task branch, not the default branch.
- Fetch and integrate the latest default-branch changes when appropriate.
- Run relevant checks and inspect the branch diff:

  ```bash
  git diff origin/main...HEAD
  ```

- Push the task branch:

  ```bash
  git push -u origin feature/short-task-name
  ```

- Never push directly to the default branch unless the user explicitly requests it and repository policy allows it.
- Never use `git push --force` or `git push --force-with-lease` unless the user explicitly approves rewriting the
  remote branch after the consequences are explained.
- Do not claim a push succeeded until Git confirms success.
- If authentication, permissions, network errors, or branch protection prevent the push, report the actual error. Do
  not bypass protections.

## 9. Pull requests and integration

- Open a pull request from the task branch into the default branch.
- Include a summary, key files changed, tests run, and known limitations.
- Request review when changes affect shared interfaces, dependencies, configuration, or another teammate's area.
- Do not merge a pull request unless the user explicitly asks and required reviews/checks have passed.
- After a PR is merged, update the local default branch:

  ```bash
  git switch main
  git fetch origin
  git pull --ff-only origin main
  ```

- Create a fresh task branch from the updated default branch for the next task.

## 10. Agent behavior and reporting

When asked to implement and publish a change:

- Inspect the Git state before editing.
- Report pre-existing uncommitted changes that could be affected.
- Make the smallest focused change.
- Run relevant tests and report exactly what passed or failed.
- Summarize changed files and important behavior.
- Review the diff before staging, committing, or pushing.
- Do not commit or push unless the user's request authorizes that action. If unclear, ask first.
- Do not silently resolve conflicts by discarding either side.
- Do not claim code is on GitHub until a successful push or pull-request operation confirms it.
- If an instruction conflicts with repository protections or these safety rules, stop and explain the conflict.

## Standard task workflow

```bash
# Start from the latest default branch
git switch main
git fetch origin
git pull --ff-only origin main

# Create an isolated task branch
git switch -c feature/short-task-name

# Make changes, inspect them, and run project tests
git status --short
git diff

# Stage only files belonging to this task
git add path/to/changed-file
git commit -m "<type>(<scope>): <description> (V<major>.<minor>)"

# Incorporate the latest shared changes
git fetch origin
git rebase origin/main

# Publish the task branch
git push -u origin feature/short-task-name

# Open a GitHub pull request for review
```

Repository-specific note: replace `main` in the examples if the default branch has another name. Follow the
repository's existing test, lint, build, branch-protection, and pull-request requirements. These instructions guide
Git behavior; they do not grant GitHub access or credentials.
