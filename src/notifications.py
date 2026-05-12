import requests
import json

class NotificationHandler:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url

    def notify(self, message):
        print(f" [Notification] {message}")
        if self.webhook_url:
            try:
                payload = {"text": message}
                requests.post(self.webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
            except Exception as e:
                print(f" [WARNING] Failed to send notification: {e}")
