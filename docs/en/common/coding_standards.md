# Coding Standards

<!-- md-trans-meta sourceCommit=0844e66715835ea9dfcfa8ffff88fa5e03a46291 translatedAt=2026-08-21T02:48:10.798Z pushedAt=2026-08-21T02:52:09.164Z -->

All submitted code must pass the local pre-commit checks. This document describes how to use pre-commit and the rules of each check.

## (Required) Quick Start

Perform the following steps to ensure that checks run automatically on every `git commit`.

### 1. Installing pre-commit

```bash
pip install "pre-commit>=4.0.0"
```

Python 3.10 is recommended.

### 2. Install Git Hooks

Run the following command once in the repository root directory:

```bash
pre-commit install
```

After that, each `git commit` automatically triggers the checks.

### 3. Pre-commit Check

First run `git add` to stage the files to be committed, and then run the following command:

```bash
pre-commit run
```

This command checks only the **staged** files, which is consistent with the scope automatically triggered by `git commit`. If the check fails, fix the issues according to the prompts, run `git add` again, and then rerun the command.

To check only specified files that have not been staged, run the following command:

```bash
pre-commit run --files <file path>
```

### 4. Normal Commit

```bash
git commit -m "your message"
```

If the check passes, the commit succeeds; if it fails, the commit is blocked.

## Check Items (Reference)

The following table lists the content checked by pre-commit. If a check fails, fix the issue according to the prompt.

### Common File Check

| Check Item (Hook Name) | Purpose |
|----------------|------|
| trailing-whitespace | Removes trailing whitespace. |
| end-of-file-fixer | Ensures a newline at the end of the file. |
| check-yaml | Validates YAML syntax. |
| check-json | Validates JSON syntax. |
| check-added-large-files | Prevents committing oversized files. |
| check-merge-conflict | Detects unresolved merge conflict markers. |
| detect-private-key | Detects private keys or sensitive credential leaks. |
| codespell | Checks spelling (for comments, documentation, etc.). |

### Python Source Code Check

| Check Item | Purpose |
|--------|------|
| ruff-check / ruff-format | Checks code style check and automatic formatting (line width 120). |
| pylint | Checks logic defects such as undefined variables and incorrect imports. |
| bandit | Checks security vulnerability scanning (such as hardcoded passwords and dangerous functions). |
| typos | Checks spelling (covers all source files, including Python code). |

## Whitelist Configuration

When a check tool falsely reports a word as a spelling error, you can add it to the whitelist.

>[!NOTE]
>
>Whitelist changes must be submitted with the PR, and the reason must be explained in the PR description.

- typos whitelist: edit `pre-commit/typos.toml`

- codespell whitelist: in `.pre-commit-config.yaml`, locate `-L` of the codespell hook and append the falsely reported word

## FAQs

### Commit Still Fails After Hooks Automatically Modify Files

Run `git add` again on the modified files, and then run `git commit`.

### YAML Check False Positive When Modifying mkdocs.yml

This occurs because pymdownx uses non-standard tags such as `!!python/name`. You can temporarily skip the YAML check (do not use `--no-verify` to skip all checks):

```bash
SKIP=check-yaml git commit -m "your message"
```

### False Positive Spelling Errors

Handle them according to the "Whitelist Configuration" section by adding the words to the corresponding whitelist.

### Slow First Run

pre-commit downloads the environment for each hook (such as Ruff and Pylint), and subsequent runs use the cache.

### When to Use a Full Check

After installing hooks for the first time, or when you need to troubleshoot legacy issues, you can run `pre-commit run --all-files` to scan the entire repository. A full check is not required for routine commits.
