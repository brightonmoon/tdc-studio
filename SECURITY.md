# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ Active development |

## Reporting a Vulnerability

TDC-Studio is a research/MLOps tool and does not process sensitive user data directly.
However, if you discover a security vulnerability, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainer directly or use GitHub's private
[Security Advisory](https://github.com/brightonmoon/tdc-studio/security/advisories/new) feature.

### What to include in your report

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You can expect an acknowledgement within **7 days** and a resolution timeline within **30 days**.

## Scope

The following are **in scope**:
- Code execution vulnerabilities in the CLI or FastAPI serving layer
- Dependency vulnerabilities with known CVEs (please check [Dependabot alerts](https://github.com/brightonmoon/tdc-studio/security/dependabot) first)

The following are **out of scope**:
- Vulnerabilities in third-party datasets provided via TDC
- Issues requiring physical access to a machine running the software
