# Continue this project with Claude Code

The root `CLAUDE.md` provides instructions that Claude Code reads when working in this checkout. It is a handoff, not an access grant. No Claude credentials, GitHub tokens or external agent were installed or connected by this change.

1. Install Claude Code using its [official setup instructions](https://code.claude.com/docs/en/setup), then sign in through its supported account flow. Availability and billing depend on your Claude account.
2. Clone the project with your existing GitHub authentication. A public clone allows reading; pushing requires an account with repository write permission. If required, use `gh auth login` in your terminal; never paste tokens into chat or the repository.
3. Start Claude Code from the project directory:

```bash
git clone https://github.com/VamshiKrishnaBandari07/Spatial-AI-system-for-Humanoid-Robotics.git
cd Spatial-AI-system-for-Humanoid-Robotics
git switch -c claude/colab-validation
claude
```

If you already have a checkout, use it and inspect existing changes first instead of cloning over it. Keep Claude's normal permission prompts; do not use unrestricted permission-bypass flags. Repository instructions load automatically, but account authentication and tool permissions still require your action.

## Starting task

Paste this into Claude Code:

```text
Read CLAUDE.md and docs/COLAB.md. Continue the existing SpatialMemory project.
Inspect the working tree and latest commits before editing. Verify the Colab
setup and launch workflow, including launch after notebook variables are cleared.
Run the existing regression tests and diagnose actual errors without changing
the model architecture or weakening identity, registration or uncertainty gates.
Keep private recordings, restricted datasets and checkpoints outside Git.
Do not claim a real phone benchmark or working public deployment from local tests.
Make fixes on the current branch and report changes, test results and blockers.
Ask before publishing changes unless I explicitly authorize a push.
```

Claude Code running on your computer cannot automatically see private datasets or `/workspace` files from the original cloud session. Supply approved local input paths when needed. GitHub access also does not grant control of your Google Colab session; run the notebook in your account or provide a separately authorized runtime integration.
