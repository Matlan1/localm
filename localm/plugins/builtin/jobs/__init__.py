# SPDX-License-Identifier: AGPL-3.0-or-later
"""Jobs plugin: scheduled recurring tasks (in-app agent jobs).

A job runs a chat or coder prompt on an interval or 5-field cron schedule; the
scheduler wakes periodically, finds enabled + due jobs, runs each via the
runner, and records the result.

Modules:
  store.py      - Job dataclass + JobStore (persist defs + results, path-confined)
  scheduler.py  - JobScheduler (asyncio loop) + a self-contained cron matcher
  runner.py     - run_job(): chat (inference engine) / coder (agent) execution
  plug.py       - register(host): mounts /api/jobs routes + starts the scheduler
  cli.py        - `localm job` add/list/run/remove/enable/disable
"""

from __future__ import annotations
