# Getting Help

venice-py is an **unofficial**, community-maintained SDK for the Venice.ai API. It is not affiliated with or endorsed by Venice.ai, and it has a single maintainer.

## Where to ask

| You want to… | Go to |
| --- | --- |
| Learn how to do something | The [documentation](https://venice-docs.sbang.dev/) and [examples](https://github.com/sethbang/venice-py/tree/main/examples) |
| Ask a usage question | [GitHub Discussions, Q&A](https://github.com/sethbang/venice-py/discussions/categories/q-a) |
| Report a bug in the SDK | [Bug report](https://github.com/sethbang/venice-py/issues/new?template=bug_report.yml) |
| Report the SDK disagreeing with Venice's API or docs | [API mismatch](https://github.com/sethbang/venice-py/issues/new?template=api_mismatch.yml) |
| Suggest a feature | [Feature request](https://github.com/sethbang/venice-py/issues/new?template=feature_request.yml) |
| Report a vulnerability | Privately, as described in [SECURITY.md](SECURITY.md). Do not use a public issue |

Please search existing issues and discussions first. When you post, include your venice-py version (`python -c "import venice_ai; print(venice_ai.__version__)"`), your Python version and a minimal reproduction. **Never post your API key.** Redact `Authorization` and wallet-signing headers from any logs.

## What's in scope

**In scope:** the `venice_ai` package, the `venice-py` CLI, the bundled Claude Code skills, the examples and the documentation site.

**Out of scope:** Venice itself. That covers your Venice account, API keys, billing, credits, rate-limit increases, model availability, content policy, and API outages or errors that also happen without the SDK. Contact Venice for these at [support@venice.ai](mailto:support@venice.ai) or on the [Venice Discord](https://discord.gg/askvenice). If you are not sure which side a problem is on, open an issue and the maintainer will help route it.

## Supported versions

- **SDK:** the latest minor release receives fixes, and the previous minor release receives security fixes. 1.x (`venice-ai<2`) is not supported. See [SECURITY.md](SECURITY.md#supported-versions).
- **Python:** 3.11 to 3.14. Each Python version is dropped at its CPython end of life, so 3.11 is dropped in October 2027.

## Response times

Best effort: a first response to issues and discussions within 5 business days. Security reports follow the faster targets in [SECURITY.md](SECURITY.md). Pull requests that follow [CONTRIBUTING.md](CONTRIBUTING.md) get reviewed fastest.
