# Deployment Guide

How to get the daily risk brief running on GitHub Actions.

---

## Prerequisites

- Repository pushed to GitHub (public or private)
- Python API keys for the services you want to use

---

## 1. Required secrets

Go to **GitHub → your repo → Settings → Secrets and variables → Actions → New repository secret**.

| Secret | Required | Where to get it |
|--------|----------|-----------------|
| `ANTHROPIC_API_KEY` | Yes | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `POLYGON_API_KEY` | One of these two | [polygon.io](https://polygon.io) → Dashboard → API Keys (free tier available) |
| `NEWS_API_KEY` | One of these two | [newsapi.org](https://newsapi.org) → Get API Key (free tier: 100 req/day) |

The agent picks whichever news key is present — Polygon is preferred.

---

## 2. Optional: Slack notifications

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App → From scratch**
2. Name it `Risk Brief Bot`, pick your workspace
3. **Incoming Webhooks → Activate → Add New Webhook to Workspace**
4. Choose the channel (e.g. `#portfolio-alerts`)
5. Copy the webhook URL

Add as a repo secret:

| Secret | Value |
|--------|-------|
| `SLACK_WEBHOOK_URL` | `https://hooks.slack.com/services/T.../B.../...` |

The workflow posts a preview of the markdown brief and links to the full artifact.
If this secret is not set, the Slack step is silently skipped.

---

## 3. Optional: Email notifications

The workflow uses [`dawidd6/action-send-mail`](https://github.com/dawidd6/action-send-mail).
It sends the HTML report as the email body — renders correctly in Gmail, Apple Mail, and Outlook.

**Gmail setup (recommended):**

1. Enable 2-Step Verification on the sending account
2. Go to **Google Account → Security → App Passwords**
3. Generate an app password for "Mail"

Add these secrets:

| Secret | Example value |
|--------|---------------|
| `MAIL_SERVER` | `smtp.gmail.com` |
| `MAIL_USERNAME` | `yourname@gmail.com` |
| `MAIL_PASSWORD` | Your 16-character app password |
| `MAIL_TO` | `recipient@example.com` (or comma-separated list) |

**Other SMTP providers** (Outlook, SendGrid, etc.) work the same way — just change `MAIL_SERVER`.

If `MAIL_TO` is not set, the email step is silently skipped.

---

## 4. Schedule

The workflow runs at **12:00 UTC on weekdays**:

| Season | Local time |
|--------|-----------|
| EST (Nov–Mar, UTC−5) | 7:00 AM ET |
| EDT (Mar–Nov, UTC−4) | 8:00 AM ET |

NYSE opens at 9:30 AM ET, so the brief is always ready before the open.

To change the time, edit the `cron` line in `.github/workflows/daily_brief.yml`:

```yaml
- cron: "0 12 * * 1-5"   # change 12 to your preferred UTC hour
```

Use [crontab.guru](https://crontab.guru) to verify cron expressions.

---

## 5. Manual trigger

You can run the workflow at any time from the Actions tab:

1. Go to **Actions → Daily Risk Brief → Run workflow**
2. Optionally enter a custom portfolio file path (default: `data/portfolio.csv`)
3. Click **Run workflow**

This is useful for testing before the scheduled run kicks in.

---

## 6. Viewing reports

Every run uploads the `.md` and `.html` files as an artifact:

1. Go to **Actions → Daily Risk Brief → (click a run) → Artifacts**
2. Download `risk-brief-<run-id>.zip`
3. Open `risk_brief_YYYY-MM-DD.html` in a browser for the full formatted report

Artifacts are retained for **30 days**.

---

## 7. Updating your portfolio

Edit `data/portfolio.csv` directly in the repo:

```csv
ticker,shares,cost_basis
AAPL,50,150.00
MSFT,30,280.00
GOOGL,10,120.00
```

Push the change and the next run will use the updated holdings automatically.

For a private portfolio, consider keeping holdings in a GitHub secret and writing them to a temp file at runtime instead of committing them.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: Portfolio file not found` | Portfolio path wrong | Check the path in the workflow input or `main.py` default |
| `No news provider is configured` | Both news secrets missing | Add `POLYGON_API_KEY` or `NEWS_API_KEY` |
| `PriceFetchError: No data returned` | Bad ticker in portfolio | Check `data/portfolio.csv` for typos |
| `RuntimeError: Agent did not call submit_risk_memo` | Anthropic API issue or token limit | Check `ANTHROPIC_API_KEY`; raise `max_tokens` in `memo.py` |
| Slack step skipped silently | `SLACK_WEBHOOK_URL` not set | Add the secret |
| Email step skipped silently | `MAIL_TO` not set | Add the secret |
| `Stream idle timeout` error | Transient Anthropic issue | Re-run the workflow from the Actions tab |
