"""
TEMPORARY COMPATIBILITY SHIM — do not build on this module.

Three modules on ``main`` import from ``classification.config1``:

    classification/evaluation/config.py
    classification/evaluation/io.py
    classification/infrastructure/io.py

That module has never existed in the repository; the real one is
``classification/config.py``. As a result ``evaluate`` and the classification
I/O helpers cannot be imported on ``main`` at all.

Two of the names those modules import are also gone:

    PROVIDERS_TO_RUN    -> renamed to STRATEGIES_TO_RUN
    validate_providers  -> renamed to validate_strategies

Both renames happened in commit dba2d7a ("Strategy based classification
pipeline instead of provider based"), whose message explicitly notes that the
evaluation code still has to be updated.

This shim re-exports everything from ``classification.config`` and aliases the
two renamed names, so that the pre-filter work package can run ``evaluate``
against its own output without editing files owned by other team members.

REMOVE THIS FILE once the evaluation modules import from
``classification.config`` directly. It is deliberately a new file rather than
an edit to the three broken modules, so that deleting it is the only cleanup
step required.

See ``classification/prefilter/README.md`` -> "Findings for the team".
"""

from classification.config import *  # noqa: F401,F403
from classification.config import (  # noqa: F401  (explicit re-export)
    CLASSIFICATION_DIR,
    CLASSIFICATION_LIMIT,
    DATA_DIR,
    DEFAULT_INPUT_FILE,
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PREDICTION_STAGE,
    MODEL_REGISTRY,
    RESULTS_DIR,
    STRATEGIES,
    STRATEGIES_TO_RUN,
    STRATEGY_REGISTRY,
    get_model_config,
    get_strategy_config,
    validate_strategies,
)

# ── Renamed in dba2d7a, still imported under the old names ──────────

PROVIDERS_TO_RUN = STRATEGIES_TO_RUN

validate_providers = validate_strategies
