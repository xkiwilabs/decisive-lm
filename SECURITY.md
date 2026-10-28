# Security

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately through GitHub's "Report a
vulnerability" form on this repository, or by email to michael.j.richardson@mq.edu.au. We aim to reply within a week.

## Supported versions

Fixes are made on the latest release and on `main`.

## Things to know

- **Secrets stay out of configs.** Keep server addresses, keys and tokens in the untracked `configs/local.yaml` or in
  environment variables, never in a committed file. The release export (`scripts/export_release.py`) fails if it
  finds anything shaped like a credential.
- **Bundles are PyTorch files.** Bundles are loaded with `torch.load(..., weights_only=True)`. Load bundles only from
  sources you trust all the same.
- **A model's answers are not a security boundary.** Don't rely on the model alone for decisions where an attacker
  controls the input, such as moderation or guardrails. Inputs can be written to mislead it. Review the least
  confident answers, as the guide describes.
