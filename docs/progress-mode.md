# Progress Mode Configuration

## Overview

`trop-nwm` uses [rich](https://github.com/Textualize/rich) for beautiful progress display. However, when used alongside other frameworks that also use rich, the progress bars may conflict and cause flickering.

To solve this, we provide two progress modes:

## Modes

### 1. `rich` mode (default)
- Beautiful spinners and progress bars
- Transient display (disappears after completion)
- **May conflict** with other rich-using frameworks

### 2. `simple` mode
- Uses `logger.info()` for progress updates
- Still benefits from rich's pretty logging (colors, formatting)
- **Compatible** with other frameworks

## Usage

### Method 1: Environment Variable (Recommended)

Set before running your script:

```bash
# On Linux/Mac
export TROP_NWM_PROGRESS_MODE=simple

# On Windows PowerShell
$env:TROP_NWM_PROGRESS_MODE="simple"

# On Windows CMD
set TROP_NWM_PROGRESS_MODE=simple
```

Then run your code normally:

```python
from trop_nwm import ZTDNWMGenerator

gen = ZTDNWMGenerator(...)  # Will use simple mode
ztd = gen.compute()
```

### Method 2: In Python Code

Set at the beginning of your script:

```python
import trop_nwm

# Switch to simple mode
trop_nwm.set_progress_mode('simple')

# Now use trop-nwm as usual
gen = trop_nwm.ZTDNWMGenerator(...)
ztd = gen.compute()
```

Or switch back to rich mode:

```python
trop_nwm.set_progress_mode('rich')
```

## Example Output

### Rich Mode
```
⠋ 1/10: Load and Format dataset [2.3 s]
• 1/10: Load and Format dataset [2.3 s]
⠹ 2/10: Interpolate horizontally [15.2 s]
```

### Simple Mode
```
[INFO] Starting: 1/10: Load and Format dataset
[INFO] ✓ Completed: 1/10: Load and Format dataset [2.3 s]
[INFO] Starting: 2/10: Interpolate horizontally
[INFO] Progress: 1250/5000 (25%)
[INFO] Progress: 2500/5000 (50%)
[INFO] ✓ Completed: 2/10: Interpolate horizontally [15.2 s]
```

## When to Use Simple Mode

Use simple mode when:
- Using `trop-nwm` in a larger framework that also uses rich
- Running in environments where terminal control codes cause issues
- You prefer simpler, log-based output

## When to Use Rich Mode

Use rich mode (default) when:
- Running standalone scripts
- You want the prettiest output
- No conflicts with other frameworks
