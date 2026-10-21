# Command-Line API Document Checklist

> Used to check command-line interface documents that are created or reworked based on the *[Command-Line API Document Template](05_cli_api_contract_template.md)*. For the general check items, see the *[Common Checklist](00_common_checklist.md)*; for the check mechanism (severity levels, conclusion rules, check execution, and the waiver mechanism), see Section 4.2 of the *[msModelSlim Documentation Standards](./README.md)*. The following are the check items specific to the command-line API document type, separated into tables by severity level.

## 1. Check Items

### 1.1 ERROR (Mandatory)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| AE-01 | Consistent command name | The command name in the document title matches the actual executable command. | Fix the command name in the title |
| AE-02 | Command format | "Command Format" gives a complete syntax summary, listing required parameters directly, marking optional parameters with square brackets and values to be replaced with angle brackets, and is not presented as a command that can be copied directly. | Fix the syntax summary to ensure it matches `--help` |
| AE-03 | Parameter list consistent with the implementation | The "Parameter List" covers all parameters exposed to users in `--help` and in the implementation code, with no omissions or extras; the name, alias, type, passing form, default value, and required or optional status of each parameter are consistent with the implementation; Boolean parameters clearly distinguish the no-value switch usage from the explicit-value usage. | Add the missing parameters, delete the parameters used only internally, and fix the inconsistent fields |

### 1.2 WARN (Recommended)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| AW-01 | Function description | "Function Description" accurately describes the capability, applicable scenarios, and boundaries of the command. | Add or fix the function description |
| AW-02 | Parameter relationships | "Parameter Relationships" describes rules such as mutual exclusion, dependencies, and priorities; when there are no parameter relationships, it explicitly states "None". | Add the parameter relationships, or state "None" |
| AW-03 | Referenced configurations | Check only when the "Referenced Configurations" section exists: the links point to the field descriptions of the configuration documents or to standalone configuration documents, rather than only to the source code. | Fix the link targets |
| AW-04 | Environment variables | Check only when the "Environment Variables" section exists: the default values are consistent with the code; global environment variables are not described again. | Fix the default values or delete the duplicated descriptions |
| AW-05 | Usage examples | Check only when the "Usage Examples" section exists: the first example is a minimal runnable scenario; paths, model names, and similar values in the examples are replaced with uppercase shell variables. | Streamline the example parameters and replace the concrete values with variables |
| AW-06 | Exit codes and exception handling | Check only when the "Exit Codes and Exception Handling" section exists: when stable exit codes are not defined, only the meaning of `0` and non-`0` is described, without listing undefined exit codes. | Reduce it to a description of `0` and non-`0` |
| AW-07 | Security notes | Check only when the "Security Notes" section exists: the risks, trigger conditions, and mitigations are described. | Add the risks, trigger conditions, or mitigations |

## 2. Usage Notes

- **Validation scope**: command-line interface documents written based on the *[Command-Line API Document Template](05_cli_api_contract_template.md)*, stored in `docs/zh/api_reference/cli/`.
