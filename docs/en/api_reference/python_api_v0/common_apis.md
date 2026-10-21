# Public Interface

## Function

During the use of the msModelSlim tool, the process log information and log level can be configured through the `set_logger_level` API. The logs include those printed on the screen.

This API is optional. If not configured, the default log level is used, which is INFO.

## Prototype

```python
set_logger_level(level="info")
```

## Parameters

| Log Level | Meaning |
| ------ | ------ |
| notset | No log level is set. Logs of all levels are printed by default. |
| debug | Logs of debug, info, warn/warning, error, fatal, and critical levels are printed. |
| info | Logs of info, warn/warning, error, fatal, and critical levels are printed.|
| warn | Logs of warn/warning, error, fatal, and critical levels are printed. |
| warning | Logs of warn/warning, error, fatal, and critical levels are printed. |
| error | Logs of error, fatal, and critical levels are printed. |
| fatal | Logs of fatal and critical levels are printed. |
| critical | Logs of fatal and critical levels are printed.|

The log level is case-insensitive, meaning Info, info, and INFO are all valid values.

## Sample

```python
from msmodelslim import set_logger_level  
set_logger_level("info")  
```
