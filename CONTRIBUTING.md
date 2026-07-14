# Contributing and solo pull-request workflow

LearnBridge is currently maintained by one contributor. Every substantive change still reaches `main` through a GitHub pull request.

## Branch workflow

1. Update local `main`: `git switch main && git pull --ff-only`.
2. Create a focused branch: `git switch -c feature/short-description` or `fix/short-description`.
3. Make small, descriptive commits using imperative messages.
4. Run `make all` and inspect `git diff main...HEAD`.
5. Push the branch and open a PR into `main` using the repository template.
6. Wait for GitHub Actions to pass. Perform the solo self-review checklist below and record it in a PR comment.
7. Resolve every review note, then squash-merge the PR. Delete the branch after merging.

Do not commit directly to `main` except for the initial repository bootstrap. Configure GitHub branch protection to require a pull request and passing CI before merging.

## Solo self-review

Because no second team member exists, the author conducts a documented second-pass review in the GitHub diff view. This does not pretend to be an independent approval; the PR description must state `Solo project — self-review performed`.

- Read every changed line outside the editor.
- Confirm the change has a single clear purpose and sufficient context.
- Review tests first, then design, functionality, complexity, naming, comments, consistency, documentation, accessibility, and security.
- Confirm generated metrics are computed rather than fabricated.
- Confirm explanations use only catalog metadata.
- Confirm no secrets, large base-model binaries, or private data are included.
- Run `make all` and record the result.
- Address or explicitly resolve every review comment before merging.

If the course strictly requires another person’s approval, request review from the instructor, teaching assistant, or an assigned peer; GitHub does not allow an author to approve their own PR.

## Review style

Reviews should be respectful and constructive. Ask questions, explain the concern, and suggest a concrete improvement. Distinguish blocking issues from optional suggestions and provide references where useful.

These practices adapt the [Practical Pull Request Review Guide](https://github.com/mawrkus/pull-request-review-guide) and [The Art of Giving and Receiving Code Reviews](https://www.alexandra-hill.com/2018/06/25/the-art-of-giving-and-receiving-code-reviews/).
