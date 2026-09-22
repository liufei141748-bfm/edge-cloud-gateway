# Experimental V1 release privacy audit

Audit date: 2026-09-22

Scope: files intended for the first public Git commit. This audit does not authorize repository creation, remote creation, or making a repository public.

## Publication boundary

The project directory contains local development assets that are intentionally retained on the workstation and excluded by `.gitignore`:

- `.env` and `.env.*` except `.env.example`;
- `config.local.toml` and `config.local.toml.*`;
- `.venv/`, `build/`, `dist/`, `*.egg-info/`, caches, coverage output, and editor metadata;
- `data/`, SQLite files, logs, and `outputs/`;
- `evaluation-private/` and generated danger A/B reports;
- internal development records listed explicitly in `.gitignore`, including the append-only project worklog and machine-specific verification artifacts.

These files must not be force-added. The ignored local database and evaluation outputs may contain request material and must be treated as private even when no credential is present.

## Public fixtures retained

The synthetic danger cases and scripted responses under `src/edge_cloud_gateway/data/` remain publishable research fixtures. They are fabricated test material, contain no real customer requests, and are required for the offline evaluation suite. Example request JSON and unit-test sentinel strings are also synthetic.

## Scan categories

The release scan checks the candidate public file set for:

- private absolute paths such as `/Users/...`;
- credential-like values (`sk-...`, bearer tokens, passwords, and assigned API keys);
- `.env`, local TOML, database, log, cache, build, and private evaluation artifacts;
- personal Obsidian paths or private chat/export material;
- provider hard-coding outside clearly named examples, compatibility tests, and historical evaluation descriptions.

Expected non-secret matches are limited to environment-variable **names**, placeholder values, fake unit-test sentinels, and the explicitly requested provider examples. No real secret value is required or stored by the public configuration.

## Result

- The simulated first-commit index contains 57 public candidate files. Ignored local assets were not included.
- No real API key, token, password, personal Obsidian path, or private request/log payload is intended for the public candidate set.
- Root configuration uses only `LOCAL_API_KEY` and `CLOUD_API_KEY` variable names; values remain external to the repository.
- Default runtime metrics use a strict metadata whitelist and do not persist prompt or context text. Raw/Working snapshot storage is now explicit opt-in through `[observability] save_context_snapshots=true`; the public default is false. Danger evaluation enables snapshots only inside its isolated evaluation store/report flow.
- Machine-specific paths were removed from the public README. Historical machine-specific verification documents are excluded from the public boundary rather than rewritten or deleted.
- Credential-pattern matches in the candidate set were limited to explicit fake test sentinels such as `Bearer secret-header`, `Bearer local-test-key`, and placeholder client keys; their tests verify redaction/header behavior and they are not usable credentials.
- A wheel built from a clean archive of the candidate file set contains the package modules and anonymized danger fixtures, and contains no `.DS_Store`.
- The repository is not yet initialized or public. A final human must inspect the exact staged file list and diff before the first commit.

## Required human check before publication

1. Initialize Git only in this project directory.
2. Run `git status --short --ignored` and confirm every local/private asset is ignored.
3. Review `git diff --cached --stat` and `git diff --cached` after staging.
4. Confirm the chosen repository name, owner, visibility, and remote URL.
5. Do not use `git add -f` for any ignored file.
