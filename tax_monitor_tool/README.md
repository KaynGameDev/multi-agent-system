# Tax Monitor Tool

This folder exposes the tax monitor as a standalone Python tool package so it can be called without the agent graph.

## Run once

```bash
python3 -m tax_monitor_tool
```

## Run continuously

```bash
python3 -m tax_monitor_tool --daemon
```

The tool reads the same tax-monitor environment variables used by the main app, including:

- `TAX_MONITOR_ENABLED=true`
- `TAX_MONITOR_URL`
- `TAX_MONITOR_USERNAME`
- `TAX_MONITOR_PASSWORD`
- `TAX_MONITOR_CAPTURE_GROUP`
- `TAX_MONITOR_SLACK_CHANNEL`

OTP collection through Slack still depends on the surrounding app or another Slack event consumer delivering 6-digit replies into the verification broker.
