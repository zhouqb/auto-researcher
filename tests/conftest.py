import os

# Never fire real desktop notifications or webhooks from the test suite.
os.environ.setdefault("DESKTOP_NOTIFICATIONS", "false")
