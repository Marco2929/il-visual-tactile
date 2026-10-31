"""Recording-plan labelling, kept out of the recorder.

record.py owns cameras, bags and the ZED. It does not own the experiment
design, so none of this lives there: the recorder calls into `attach()` and
otherwise knows nothing about Folienlagen, blocks or positions. A different
plan is a different module, not a patch to the recorder.

    from record_utils.plan import attach
    plan = attach(node)          # None when PLAN_ENABLED=0

Layout:
    schedule.py   the plan itself: vocabulary, loop geometry, slot -> labels
    store.py      labels.json read/write + progress reconstructed from disk
    session.py    cursor, attempts, operator I/O; the recorder-facing part
"""

from .session import attach, PlanSession
from . import schedule, store

__all__ = ["attach", "PlanSession", "schedule", "store"]