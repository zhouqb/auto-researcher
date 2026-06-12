import os

# Never fire real desktop notifications or webhooks from the test suite.
os.environ.setdefault("DESKTOP_NOTIFICATIONS", "false")

# Never export traces to a real Langfuse from the test suite: empty env vars
# take precedence over any keys in the developer's ~/.env.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
