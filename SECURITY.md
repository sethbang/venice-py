# Security Policy

venice-py is an unofficial, community-maintained Python SDK for the Venice.ai API. It is not affiliated with or endorsed by Venice.ai.

## Reporting a vulnerability

Please do not report security problems in public issues, discussions or pull requests.

- **Preferred:** [open a private report](https://github.com/sethbang/venice-py/security/advisories/new) through GitHub's private vulnerability reporting.
- **Fallback:** email [howdy@sbang.dev](mailto:howdy@sbang.dev) if you cannot use GitHub.

A useful report includes the venice-py version, the Python version, the affected API (class, method or CLI command), steps to reproduce, and what an attacker gains.

**Never include a real API key, wallet private key, seed phrase or signed payment header** in a report, log or reproduction. Redact `Authorization`, `X-Sign-In-With-X` and `X-402-Payment` headers. If a key has already been exposed, revoke it in your [Venice API settings](https://venice.ai/settings/api) first.

## What to expect

venice-py has a single maintainer, so these targets are best effort:

| Step | Target |
| --- | --- |
| Acknowledge the report | within 3 business days |
| Fix a high- or critical-severity issue | within 14 days of confirmation |
| Lower severities | in a regular release |

The maintainer will keep you updated, coordinate a disclosure date with you, and credit you in the GitHub Security Advisory and the CHANGELOG unless you would rather stay anonymous.

## Supported versions

| Version | Status |
| --- | --- |
| Latest minor release (2.x) | Bug and security fixes |
| Previous minor release | Security fixes only |
| Older 2.x releases | Not supported; upgrade to the latest minor |
| 1.x (`venice-ai<2`) | Not supported |

Fixes ship as new patch releases on PyPI under the `venice-py` name. The `venice-ai` distribution is a metadata-only alias that installs `venice-py`, so upgrading `venice-py` covers it.

## Scope

In scope: code in this repository, including the `venice_ai` package, the `venice-py` CLI, the bundled Claude Code skills, the examples, and the release and CI workflows. Typical examples are credential leakage through logs, errors, or retries, unsafe handling of x402 payment signing, TEE attestation checks that can be bypassed, and dependency issues that affect the SDK.

Out of scope: vulnerabilities in Venice's API, website, models or account system. Report those to Venice directly, for example via [support@venice.ai](mailto:support@venice.ai). If you are unsure which side a problem is on, report it here privately and the maintainer will help route it.
