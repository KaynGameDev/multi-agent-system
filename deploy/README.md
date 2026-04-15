# Deploy Jade Web Chat Behind Cloudflare Tunnel

This folder contains production-oriented templates for publishing the Jade web interface at `chat.jade-games.com` through Cloudflare Tunnel.

Recommended shape:

1. Run Jade on a Linux host or VM.
2. Bind Jade only to `127.0.0.1:8000`.
3. Publish that local origin through `cloudflared`.
4. Protect the public hostname with Cloudflare Access.
5. Keep Jade's built-in web auth enabled as defense in depth until the rollout is stable.

## Files

- `env/chat.jade-games.com.env.example`
  Production environment template for the Jade app.
- `systemd/jade-agent-web.service`
  Example `systemd` unit for running Jade on boot.
- `cloudflared/chat-jade-games.com.yml`
  Example named-tunnel configuration for `chat.jade-games.com`.

## Recommended rollout

1. Copy `env/chat.jade-games.com.env.example` to a secure location on the server.
2. Replace placeholders with real values and secrets.
3. Install Python dependencies and make sure `agent_env/bin/python` works.
4. Install the `systemd` unit and start Jade.
5. Install `cloudflared`, create a named tunnel, and update `cloudflared/chat-jade-games.com.yml`.
6. Route `chat.jade-games.com` to the tunnel.
7. Add a Cloudflare Access self-hosted application for `chat.jade-games.com`.
8. Validate login, chat, and document conversion from the public hostname.

## Cloudflare commands

Run these on the server after installing `cloudflared`:

```bash
cloudflared tunnel login
cloudflared tunnel create jade-chat
cloudflared tunnel route dns jade-chat chat.jade-games.com
```

Then copy `cloudflared/chat-jade-games.com.yml` to `~/.cloudflared/config.yml`, fill in the real tunnel UUID and credentials path, and install the service:

```bash
sudo cloudflared --config /home/<linux-user>/.cloudflared/config.yml service install
sudo systemctl enable --now cloudflared
```

## Cloudflare Access

Before you publish the hostname to your whole team, create a Cloudflare Access self-hosted application for `chat.jade-games.com`.

Suggested policy:

- Allow only your company email domain.
- Require your existing IdP login if available.
- Keep Jade's own `WEB_AUTH_ENABLED=true` during the first rollout.

## Validation checklist

- `curl -I http://127.0.0.1:8000/` returns `200` on the host.
- `systemctl status jade-agent-web` is healthy.
- `systemctl status cloudflared` is healthy.
- Visiting `https://chat.jade-games.com` first hits Cloudflare Access, then Jade login.
- Chat history loads and new chats persist after a Jade restart.
- Google Sheets and Docs features still work with the server's credentials.

## Secrets and safety

- Do not commit populated `.env` files.
- Do not store Cloudflare tunnel credentials in the repo.
- Keep `WEB_HOST=127.0.0.1`, not `0.0.0.0`.
- Keep `WEB_ALLOWED_HOSTS=chat.jade-games.com,127.0.0.1,localhost`.
