"""Rich UI utilities for progress tracking and logging."""

from __future__ import annotations

import contextlib
import functools
import logging
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


class SimpleTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed as [x.x s]."""

    def render(self, task) -> Text:
        elapsed = task.elapsed
        if elapsed is None:
            return Text("-", style="bold gold1")
        return Text(f"[{elapsed:.1f} s]", style="bold gold1")


def track_step(description: str) -> Callable[[F], F]:
    """Decorator to display a spinner during method execution and a checkmark upon completion."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            desc_markup = description
            ordinal = ""
            rest = description
            match = re.match(r"^(\d+/12)(.*)", description)
            if match:
                ordinal, rest = match.groups()
                desc_markup = f"[bold gold1]{ordinal}[/]{rest}"

            progress = Progress(
                SpinnerColumn(style="bold gold1"),
                TextColumn("[progress.description]{task.description}"),
                SimpleTimeElapsedColumn(),
                transient=True,
            )
            progress.add_task(desc_markup, total=None)
            with progress:
                result = func(*args, **kwargs)
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
    """Context manager to patch joblib's parallel processing with a Rich progress bar."""
    t0 = time.perf_counter()
    desc_markup = description
    ordinal = ""
    rest = description
    match = re.match(r"^(\d+/12)(.*)", description)
    if match:
        ordinal, rest = match.groups()
        desc_markup = f"[bold gold1]{ordinal}[/]{rest}"

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
        elapsed = time.perf_counter() - t0
        if ordinal:
            console.print(
                f"[bold blue]• {ordinal}[/]{rest} [bold blue][{elapsed:.1f} s][/]"
            )
        else:
            console.print(
                f"[bold blue]•[/] {description} [bold blue][{elapsed:.1f} s][/]"
            )
