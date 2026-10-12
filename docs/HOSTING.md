# Hosted workspace

GitHub Pages serves `site/`: a responsive video-preview and report-review interface
with a launcher for the existing Colab analysis UI. It cannot execute Python,
store uploaded recordings, protect API keys, or run neural inference. The browser
previews and JSON reports remain local; there are no analytics or remote scripts.

## Enable the public site

1. Open repository **Settings → Pages → Build and deployment**.
2. Select **GitHub Actions** as the source.
3. Open **Actions → Publish web workspace → Run workflow** on `main`.
4. Wait for successful deployment and use the URL shown by the deployment.

Expected address after deployment:
https://VamshiKrishnaBandari07.github.io/Spatial-AI-system-for-Humanoid-Robotics/

This URL is not evidence that deployment has succeeded. Environment API access
was denied during setup, so Pages enablement and live deployment could not be
verified from the development workspace.

## Run video analysis

Use **Launch analysis in Colab** from the site. Execute notebook setup and startup
cells, choose a UI password, open its temporary link, and upload the two original
recordings. The selected browser previews are not transferred to Colab. Save the
results archive before the runtime ends. See [Colab setup](COLAB.md).

The separate [visual-language comparison adapter](VISUAL_COMPARISON.md) remains a
command-line stage requiring a server-side API credential; never put credentials
in Pages HTML, JavaScript, reports or repository files. Its live inference has not
been validated. The static report viewer can read its output once available.

## Hosting limits

The Colab runtime is temporary and resource availability is not guaranteed.
Always-on uploads and inference require a separately hosted, authenticated backend
with sufficient RAM/GPU resources and storage. The existing local FastAPI service
is not a hardened multi-user public deployment. GitHub Actions runs only static
site deployment, not inference. Current detection accuracy remains unresolved.
