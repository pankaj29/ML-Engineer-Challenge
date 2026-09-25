# %% [markdown]
# # API smoke test, one group per cell
#
# The same checks `scripts/smoke_test_api.py` runs in one go, split so you can
# run a group at a time and look at the result before moving on.
#
# **Open this file in VS Code and click "Run Cell"** above any `# %%` block, or
# Ctrl+Enter. PyCharm's scientific mode and Jupytext read the same markers.
# Run the setup cell first; after that the cells are independent and you can
# rerun any one of them on its own.
#
# The checks live in `smoke_test_api.py` rather than here, so the script you
# run in CI and the cells you run by hand cannot drift apart.
#
# Needs the stack up: `docker compose up -d`.

# %% setup: run this first
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "scripts").is_dir():  # started from scripts/ rather than the repo root
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_api import (
    check_auth,
    check_batch,
    check_classification,
    check_detection,
    check_health,
    check_input_validation,
    check_models,
    check_similarity,
    open_session,
    results,
    summarise,
)

# Gateway on :80, which is what a caller talks to. For the API's own port with
# no nginx in front: open_session("direct"). For anything else:
# open_session(base_url="https://your-host", key="YOUR_KEY").
s = open_session()

# %% health and metrics
# Public, no key needed. /health reports each dependency separately, so this is
# the cell that tells you which service is the problem.
check_health(s)

# %% authentication
# A valid key gets a JWT, a bad key gets 401, and an unauthenticated call gets
# 401. Refusing correctly is part of the contract, not a failure.
token = check_auth(s)
token[:32]

# %% classification
# Base64 and multipart, plus top_k. Prints the model, runtime and top label, so
# it is also how you confirm which runtime is actually serving.
check_classification(s)

# %% detection
check_detection(s)

# %% similarity
# Embed, index, search, stats. The interesting assertion is that an image
# indexed a moment ago is found again: under a per-process index behind more
# than one replica, it would not be.
check_similarity(s)

# %% batch (slow, around 10 seconds)
# Submits a 3-image job, polls until it reaches a terminal state, then submits
# a second one and cancels it. The poll is why this cell is the slow one.
check_batch(s)

# %% models
check_models(s)

# %% input validation
# Malformed base64, a non-image payload, an SSRF attempt at the cloud metadata
# address, and an empty body. All four must be refused.
check_input_validation(s)

# %% summary of everything run so far
# `results` accumulates across cells, so this covers whichever groups you ran,
# in whatever order.
summarise()

# %% start over without restarting the kernel
results.clear()

# %% switch to the API's own port, bypassing nginx
# Anything that passes here but fails on the gateway is nginx: its body limit,
# its timeouts or its proxy headers.
s.client.close()
results.clear()
s = open_session("direct")
