"""Where the URDF->USD converter is allowed to write.

IsaacLab hardcodes its scratch directory. ``AssetConverterBase.__init__`` does::

    if cfg.usd_dir is None:
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._usd_dir = f"/tmp/IsaacLab/usd_{time_tag}_{random.randrange(10000)}"
    ...
    os.makedirs(self.usd_dir, exist_ok=True)

-- a literal ``/tmp``, reached by no environment variable, TMPDIR included. On a
shared machine ``/tmp/IsaacLab`` belongs to whoever ran IsaacLab on it first,
and because the per-run leaf does not exist yet the ``makedirs`` is what fails::

    PermissionError: [Errno 13] Permission denied: '/tmp/IsaacLab/usd_20261112_173054_6311'

Passing ``usd_dir`` explicitly is the only way out, so every robot config here
sets it from this helper.

Deliberately still unique per process per run, like upstream, rather than a
stable path the converter could reuse. ``usd_dir`` is documented as enabling
lazy conversion, and the hash that guards it covers the config and the URDF's
own bytes -- but ``_config_to_hash`` warns that "it only checks the main asset
file", so a changed *mesh* under an unchanged URDF would silently load a stale
robot. Re-converting each run costs a few seconds; debugging a model that no
longer matches its STLs costs rather more. The caching is available if someone
later wants it, with that caveat understood.

The PID replaces upstream's ``random.randrange(10000)``: two ranks starting
inside the same second collide with probability 1e-4 there, and never here.
"""

from __future__ import annotations

from datetime import datetime
import os
import tempfile

__all__ = ["usd_cache_dir"]


def usd_cache_dir(robot: str) -> str:
    """Return a per-run directory for ``robot``'s generated USD.

    Rooted at :func:`tempfile.gettempdir`, so it follows ``TMPDIR`` -- which is
    how a batch job moves it somewhere the job's own user owns. With ``TMPDIR``
    unset this is ``/tmp/IsaacLab/...``, matching upstream's location.
    """
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        tempfile.gettempdir(), "IsaacLab", f"usd_{robot}_{tag}_{os.getpid()}"
    )
