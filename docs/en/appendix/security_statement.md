# msModelSlim Security Statement

## System Security Hardening

You are advised to enable the **Address Space Layout Randomization** (ASLR) (level 2) in the system. Run the following command to enable it:

    echo 2 > /proc/sys/kernel/randomize_va_space

## Running User Recommendations

 1. Use the principle of least privilege. For example, prevent other users from writing data by disabling permissions such as `666` and `777`.

 2. Before installing or using the tools, ensure the user's `umask` is set to `0027` or a more restrictive value. Otherwise, issues may occur, such as source code compilation failures, configuration file read failures, or excessive permissions for generated directories and files.

 3. All tools in this repository are designed to run with minimal permissions. For security reasons, do not use `root` or other privileged accounts. Always install and execute tools as a regular user.
 
 4. If a tool depends on CANN, use the CANN package installed by the same non-privileged user. After running the `source` command, do not modify environment variables in `set_env.sh`.

## File Permission Control

 1. Ensure that the directory permissions are `750`.

 2. When providing input files to the tool as command inputs, it is recommended that the file owner matches the process owner of the tool and that file permissions restrict write access for `group` and `others`. By default, tool files written to the drive are not writable by others. You can manually control the permissions for the generated files as needed.

 3. Proper permission control is essential during installation and use. For details, see the following table.

### File Permission Reference

| Type                              | Maximum Linux Permission|
| ---------------------------------- | ------------------- |
| Home directory                        | 750 (rwxr-x---)   |
| Program files (including scripts and libraries)    | 550 (r-xr-x---)   |
| Program file directory                      | 550 (r-xr-x---)   |
| Configuration files                          | 640 (rw-r-----)   |
| Configuration file directory                      | 750 (rwxr-x---)   |
| Log files (recorded or archived)    | 440 (r--r-----)   |
| Log files (being recorded)                | 640 (rw-r-----)   |
| Log file directory                      | 750 (rwxr-x---)   |
| Debug files                         | 640 (rw-r-----)   |
| Debug file directory                     | 750 (rwxr-x---)   |
| Temporary file directory                      | 750 (rwxr-x---)   |
| Maintenance and upgrade file directory                  | 770 (rwxrwx---)   |
| Service data files                      | 640 (rw-r-----)   |
| Service data file directory                  | 750 (rwxr-x---)   |
| Key components, private keys, certificates, and ciphertext file directory| 700 (rwx------)   |
| Key components, private keys, certificates, and ciphertext files    | 600 (rw-------)   |
| APIs and scripts for encryption and decryption            | 500 (r-x------)   |

## Vulnerability Security Statement

The MindStudio community highly values the security of community versions. Vulnerability management specialists are specifically designated to handle vulnerability-related matters. To build a more secure AI full-process toolchain, we look forward to your participation.

### Vulnerability Handling Process

For each security vulnerability, the MindStudio community assigns personnel to follow up and handle it. The end-to-end vulnerability handling process is shown in the following figure.

   ![Vulnerability Handling Process](images/vulnerability_handling_process.png)

The following sections explain the vulnerability reporting, assessment, and disclosure processes.

### Vulnerability Reporting

You can contact the MindStudio community team by submitting an issue. We will arrange for a security vulnerability specialist to contact you promptly.
Note that to ensure security, do not include specific information about security privacy in the issue.

#### Response to Reports

1. The MindStudio community will confirm, analyze, and report security vulnerability issues within three working days, while initiating the security handling process.
2. The MindStudio security team will assign confirmed security vulnerability issues to dedicated personnel and follow up on them.
3. During the process of classifying, confirming, and fixing security vulnerabilities, as well as releasing patches, we will provide timely updates on the report.

### Vulnerability assessment

The industry widely uses the CVSS standard to assess vulnerability severity. When using CVSS v3.1 for vulnerability assessment, MindStudio sets specific attack scenarios and performs assessments based on the actual impact within those scenarios. Vulnerability severity assessment involves assessing the difficulty of exploitation as well as the impact on confidentiality, integrity, and availability after exploitation, resulting in a numerical score.

#### Vulnerability Assessment Metrics

MindStudio assesses vulnerability severity levels by using the following vector metrics:

- Attack vector (AV): indicates the "remoteness" of an attack and how a vulnerability can be exploited.
- Attack complexity (AC): describes the difficulty of executing an attack and the factors required for a successful attack.
- User interaction (UI): determines whether the attack requires user participation.
- Privileges required (PR): records the level of user authentication required for a successful attack.
- Scope (S): determines whether an attack can affect components with different permission levels.
- Confidentiality (C): measures the impact resulting from information disclosure to unauthorized parties.
- Integrity (I): measures the impact resulting from information tampering.
- Availability (A): measures the impact on users' access to data or services when needed.

#### Assessment Principles

- Assess the severity level of a vulnerability, not the risk.
- The assessment must be based on an attack scenario where a successful attack can compromise the confidentiality, integrity, and availability of the system.
- When a security vulnerability has multiple attack scenarios, the attack scenario with the highest CVSS score (that is, with the greatest impact) shall prevail in the assessment.
- If a vulnerability exists in an embedded or invoked library, perform the assessment after determining the attack scenario based on how the library is used in the product.
- If a security defect cannot be triggered or does not affect confidentiality, integrity, or availability (CIA), the CVSS score is 0.

#### Assessment Procedure

To assess the severity level of a vulnerability, perform the following steps:

1. Set a possible attack scenario and score based on this attack scenario.
2. Identify the vulnerable component and affected components.
3. Select values for base metrics.

   - Select values for the exploitability metrics (attack vector, attack complexity, privileges required, user interaction, and scope) based on the vulnerable component.

   - Ensure impact metrics (confidentiality, integrity, and availability) reflect the impact on either the vulnerable component or the affected components, whichever is more severe.

#### Severity Rating

| **Severity Rating** | **CVSS Score** | **Vulnerability Fix Time**|
| ------------------------------- | --------------------- | ---------------- |
| Critical               | 9.0~10.0              | 7 days             |
| High                     | 7.0~8.9               | 14 days            |
| Medium                   | 4.0~6.9               | 30 days            |
| Low                      | 0.1~3.9               | 30 days            |

### Vulnerability Disclosure

After a security vulnerability is fixed, the MindStudio community will release a security advisory (SA) and security notice (SN). The SA includes technical details of the vulnerability, type, reporter, CVE ID, affected versions, and fixed versions.
To ensure security for MindStudio users, the MindStudio community will not publicly disclose, discuss, or confirm security issues until after investigation and fixing are complete and an SA has been released.

### Appendix

#### MindStudio SA

Currently maintained versions have no security vulnerabilities.

#### MindStudio SN

No vulnerability notices are currently available for third-party open-source components.

## Data Security Statement

Loading and saving data during the use of the tools may involve data risks.

## Build Security Statement

To reduce security risks, you can perform permission control on files within the source code directory. During the build process, you can modify build scripts as needed to avoid security risks and ensure the security of the build results.

## Runtime Security Statement

If an exception occurs during operation, the tool will exit the process and print error messages. This is expected behavior. You are advised to locate the specific cause of the error based on the error prompts, such as by viewing log files or result files generated during the collection and parsing process.

## Public Network Address Statement

For details, see [Public URLs](public_urls.xlsx).

## Public API Statement

msModelSlim is developed in C++ and Python. All public APIs are documented.

For scripting languages such as Python where source code is released, use the public APIs specified in the documentation. Do not directly call source code APIs that are not explicitly disclosed.

## Usage of Safe Functions

While unsafe functions are not forcibly disabled, you are advised to use their safe variants that explicitly take a buffer length parameter, for example, `memset_s` and `memcpy_s`.

## Communication Matrix

| No.| Code Repository        | Function      | Source Device | Source IP Address        | Source Port| Destination Device| Destination IP Address       | Destination Port<br>(Listening)| Protocol| Port Description          | Port Configuration    | Listening Port Configurable (Yes/No)         | Authentication Mode| Encryption Mode| Plane   | Version    | Special Scenarios| Remarks|
|:---|:------------|:---------|:-----|:------------| :----- |:-----|:------------|:--------------| :--- |:---------------|:---------|:-------------------| :------- | :------- |:--------|:-------|:-----| :--- |
| 1  | msModelSlim | Inter-card communication for quantization| Single-node multi-device| 127.0.0.1 |        | Single-node multi-device| 127.0.0.1 | Dynamic search. The default value is 29500-29599. The value ranges from 1024 to 65535.   | TCP  | Dynamically searches for an available port for inter-card communication. Search starts at 29500 by default, and the tool tries up to 100 ports. You can specify the port through the `MASTER_PORT` environment variable. The port must be in the range from 1024 to 65535.| N/A     | Yes (using the `MASTER_PORT` environment variable)               |  |   | Service plane    | master | None   |      |
| 2  | msModelSlim | Multi-process context sharing| Local on a single node| N/A (Unix domain socket)|        | Local on a single node| N/A (Unix domain socket)| N/A (using a file system socket path)| Unix domain socket | Shares context data between processes through the unix domain socket. The socket file is created by default in the system temporary directory (`/tmp/peercred_manager_<pid>.sock`), and the file permissions are `0600`. It uses `SO_PEERCRED` for kernel-level UID authentication and allows connections only from processes with the same UID.| N/A| The socket file path can be specified in code.| Kernel-level UID authentication with `SO_PEERCRED`| N/A (local communication)| Service plane| master | Linux only|      |
