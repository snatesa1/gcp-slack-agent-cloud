# 🛠️ Slack App Setup Guide

Follow these steps to create and configure your Slack app for the **GCP Trading Agent**.

## 🚀 Pro Tip: Use App Manifest (Fastest)
Instead of manual configuration, you can copy-paste this JSON into the **App Manifest** tab in Slack (under **Settings**) to configure everything instantly. 

> [!IMPORTANT]
> Replace `YOUR_CLOUD_RUN_URL` with your actual deployment URL (e.g., `https://trading-agent-xxx.run.app`).

```json
{
    "display_information": {
        "name": "Trading Agent"
    },
    "features": {
        "app_home": {
            "home_tab_enabled": true,
            "messages_tab_enabled": true,
            "messages_tab_read_only_enabled": false
        },
        "bot_user": {
            "display_name": "Trading Agent",
            "always_online": true
        },
        "slash_commands": [
            {
                "command": "/analyze",
                "url": "https://YOUR_CLOUD_RUN_URL/slack/events",
                "description": "Analyze a stock ticker",
                "usage_hint": "[TICKER]|[DAYS] (e.g., AAPL|30)",
                "should_escape": false
            }
        ]
    },
    "oauth_config": {
        "scopes": {
            "bot": [
                "chat:write",
                "commands",
                "im:history"
            ]
        }
    },
    "settings": {
        "event_subscriptions": {
            "request_url": "https://YOUR_CLOUD_RUN_URL/slack/events",
            "bot_events": [
                "message.im"
            ]
        },
        "interactivity": {
            "is_enabled": true,
            "request_url": "https://YOUR_CLOUD_RUN_URL/slack/events"
        }
    }
}
```

---

## 🛑 Manual Configuration (Step-by-Step)

If you prefer to configure manually, follow these sections:

### 1. Create the App
1. Go to the [Slack API: Your Apps](https://api.slack.com/apps) dashboard.
2. Click **Create New App** -> **From scratch**.
3. **App Name**: `Trading Agent`.
4. **Workspace**: Select your target workspace.

### 2. Configure Scopes
1. Navigate to **OAuth & Permissions**.
2. Under **Bot Token Scopes**, add:
   - `chat:write`
   - `commands`
   - `im:history`

### 3. Enable App Home & Messages
1. Navigate to **App Home**.
2. Under **Show Tabs**, check **Allow users to send Slash commands and messages from the messages tab**.
   - *This ensures you don't see the "sending messages turned off" error.*

### 4. Install & Get Tokens
1. Install the app to your workspace from the **OAuth & Permissions** page.
2. Copy the **Bot User OAuth Token** (`xoxb-...`).

### 5. Get Signing Secret
1. Navigate to **Basic Information**.
2. Scroll to **App Credentials** and copy the **Signing Secret**.
   - *This is used for security to verify Slack requests.*

### 6. Enable Events (Message support)
1. Navigate to **Event Subscriptions**.
2. Toggle **On**.
3. **Request URL**: `https://YOUR_CLOUD_RUN_URL/slack/events`.
4. Subscribe to `message.im` under **Bot Events**.

### 7. Enable Interactivity
1. Navigate to **Interactivity & Shortcuts**.
2. Toggle **On**.
3. **Request URL**: `https://YOUR_CLOUD_RUN_URL/slack/events`.

---

## ⚙️ Project Configuration

Once you have the values, update your deployment:

1. Add to your local `.env`:
   ```bash
   SLACK_BOT_TOKEN=xoxb-your-token
   SLACK_SIGNING_SECRET=your-signing-secret
   ```
2. Upload to GCP Secret Manager:
   ```bash
   ./upload_secrets.sh
   ```
3. Redeploy:
   ```bash
   ./deploy_app.sh
   ```

🚀 **Ready!** You can now type `/analyze MSFT` or just DM the bot `TSLA` to get a full report!
