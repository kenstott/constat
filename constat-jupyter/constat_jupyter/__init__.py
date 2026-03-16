import polars
polars.Config.set_tbl_hide_column_data_types(True)
polars.Config.set_tbl_rows(100)

from .client import ConstatClient, Session
from .models import SolveResult, Artifact, StepInfo, ConstatError

__all__ = ["ConstatClient", "Session", "SolveResult", "Artifact", "StepInfo", "ConstatError"]
__version__ = "0.1.1"


def load_ipython_extension(ipython):
    from .magic import ConstatMagic, _constat_cell_transform

    magic = ConstatMagic(ipython)
    ipython.register_magics(magic)

    # Register input transformer so %%constat cells are rewritten to
    # ``await _constat_run(...)`` — Jupyter's native async execution
    # blocks properly until the query completes.
    mgr = ipython.input_transformer_manager
    if _constat_cell_transform not in mgr.cleanup_transforms:
        mgr.cleanup_transforms.insert(0, _constat_cell_transform)

    # Inject a stub so %%constat before %constat connect gives a clear error
    async def _stub(*_a, **_kw):
        from IPython.display import display, HTML
        display(HTML(
            '<div style="color:red;padding:8px;border:1px solid red;border-radius:4px">'
            '<b>Error:</b> Not connected. Run <code>%constat connect</code> first.</div>'
        ))
        return None
    ipython.user_ns.setdefault("_constat_run", _stub)


def unload_ipython_extension(ipython):
    from .magic import _constat_cell_transform

    mgr = ipython.input_transformer_manager
    if _constat_cell_transform in mgr.cleanup_transforms:
        mgr.cleanup_transforms.remove(_constat_cell_transform)
    for name in ("_constat_run", "_constat_client", "_constat_session", "_constat_result"):
        ipython.user_ns.pop(name, None)
