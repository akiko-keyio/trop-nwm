"""Rich UI utilities for progress tracking and logging."""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import time
from typing import Callable, TypeVar

import joblib
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.text import Text

# Global console for consistent output
console = Console()

# Configure Rich Logger
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, console=console)],
)
logger = logging.getLogger("trop_nwm")
logger.setLevel(logging.INFO)

# Suppress noisy third-party debug logs
for lib in ["eccodes", "cfgrib", "matplotlib", "PIL", "joblib"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


F = TypeVar("F", bound=Callable)


# ============================================================================
# Progress Mode Configuration
# ============================================================================
class ProgressMode:
    """Configuration for progress display mode.
    
    Modes:
        - 'rich': Use rich Progress with spinners and bars (default, prettier but may conflict)
        - 'simple': Use logger.info for progress (compatible with other frameworks)
    
    Set via environment variable or code:
        export TROP_NWM_PROGRESS_MODE=simple
        # or in Python:
        from trop_nwm.log_utils import set_progress_mode
        set_progress_mode('simple')
    """
    _mode = os.getenv("TROP_NWM_PROGRESS_MODE", "rich").lower()

    @classmethod
    def get(cls) -> str:
        """Get current progress mode."""
        return cls._mode

    @classmethod
    def set(cls, mode: str) -> None:
        """Set progress mode ('rich' or 'simple')."""
        if mode not in ("rich", "simple"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'rich' or 'simple'")
        cls._mode = mode

    @classmethod
    def is_rich(cls) -> bool:
        """Check if using rich progress mode."""
        return cls._mode == "rich"


def set_progress_mode(mode: str) -> None:
    """Set progress display mode.
    
    Args:
        mode: 'rich' for fancy progress bars, 'simple' for logger.info
    """
    ProgressMode.set(mode)


class SimpleTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed as [x.x s]."""

    def render(self, task) -> Text:
        elapsed = task.elapsed
        if elapsed is None:
            return Text("-", style="bold gold1")
        return Text(f"[{elapsed:.1f} s]", style="bold gold1")


def track_step(description: str) -> Callable[[F], F]:
    """Decorator to display progress during method execution.
    
    Uses rich Progress in 'rich' mode, only completion message in 'simple' mode.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            desc_markup = description
            ordinal = ""
            rest = description
            match = re.match(r"^(\d+/\d+)(.*)", description)
            if match:
                ordinal, rest = match.groups()
                desc_markup = f"[bold gold1]{ordinal}[/]{rest}"

            if ProgressMode.is_rich():
                # Rich mode: use Progress with spinner
                progress = Progress(
                    SpinnerColumn(style="bold gold1"),
                    TextColumn("[progress.description]{task.description}"),
                    SimpleTimeElapsedColumn(),
                    transient=True,
                )
                progress.add_task(desc_markup, total=None)
                with progress:
                    result = func(*args, **kwargs)
            else:
                # Simple mode: no spinner, just execute
                result = func(*args, **kwargs)
            
            # Both modes: show completion message
            elapsed = time.perf_counter() - t0
            if ordinal:
                console.print(
                    f"[bold blue]• {ordinal}[/]{rest} [bold blue][{elapsed:.1f} s][/]"
                )
            else:
                console.print(
                    f"[bold blue]•[/] {description} [bold blue][{elapsed:.1f} s][/]"
                )

            return result

        return wrapper  # type: ignore

    return decorator


@contextlib.contextmanager
def joblib_rich_progress(description: str, total: int):
    """Context manager for joblib parallel processing with progress display.
    
    Uses rich Progress bar in 'rich' mode, only completion message in 'simple' mode.
    """
    t0 = time.perf_counter()
    desc_markup = description
    ordinal = ""
    rest = description
    match = re.match(r"^(\d+/\d+)(.*)", description)
    if match:
        ordinal, rest = match.groups()
        desc_markup = f"[bold gold1]{ordinal}[/]{rest}"

    if ProgressMode.is_rich():
        # Rich mode: use Progress bar
        progress = Progress(
            SpinnerColumn(style="bold gold1"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            transient=True,
        )
        task_id = progress.add_task(desc_markup, total=total)

        # Patch joblib
        print_progress = joblib.parallel.Parallel.print_progress

        def update_progress(self):
            progress.update(task_id, completed=self.n_completed_tasks)
            return print_progress(self)

        joblib.parallel.Parallel.print_progress = update_progress
        try:
            with progress:
                yield progress
        finally:
            joblib.parallel.Parallel.print_progress = print_progress
    else:
        # Simple mode: no progress bar, just execute
        yield None
    
    # Both modes: show completion message
    elapsed = time.perf_counter() - t0
    if ordinal:
        console.print(
            f"[bold blue]• {ordinal}[/]{rest} [bold blue][{elapsed:.1f} s][/]"
        )
    else:
        console.print(
            f"[bold blue]•[/] {description} [bold blue][{elapsed:.1f} s][/]"
        )
