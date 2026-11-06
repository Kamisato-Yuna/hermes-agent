---
name: smb-share-access
description: Read and write Windows SMB shares through smbfs, with no mount and no root.
version: 2.0.0
author: 流萤 (Firefly) for 汐儿
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [smb, windows, ad, file-share, smbclient, credentials]
    category: devops
    requires_toolsets: [terminal]
---

# Windows 共享访问（smbfs）

## 何时使用本技能

满足任一条件时使用本技能：

- 用户给出以 `\\` 开头的路径，例如 `\\ftdytool.hytch.com\FTDY\hq_tool\...`。
- 用户给出“共享”“网络盘”“UNC 路径”“Windows 路径”等说法。
- 用户要读、写、列目录、上传、下载内网共享中的文件。

不要挂载共享。不要创建 `/mnt` 路径。不要 root。使用 `smbfs`。

## 第一步：确定身份

共享使用用户自己的 AD 账号。不要用别人的账号，也不要共用固定账号。

AD 账号 = `digital\<FChat 用户名>`。两者已经在环境中保持一致。

FChat 用户名不是用户 id。按下面顺序取值：

1. 消息上下文里的发送者用户名（`username`，不是 26 位 id）。
2. 只有 id 时，用 `smb-auth whoami --user-id <id>` 反查。

```bash
smb-auth whoami --user-id 4a8hs5ppepdc3k8utp9q1iftfe
# digital\wudd
```

## 第二步：检查凭据

```bash
smb-auth check --user-id <id> --host <HOST>
```

按退出码处理：

| 退出码 | 输出 | 处理 |
|---|---|---|
| 0 | `OK <path>` | 直接访问，不要再问密码 |
| 3 | `NEED_PASSWORD digital\xxx` | 向用户索要 AD 密码（第三步） |
| 4 | `INVALID_CREDENTIALS digital\xxx` | 密码已失效，向用户索要新密码（第三步） |
| 5 | 无法反查用户名 | 请用户直接给出 FChat 用户名 |
| 6 | 客户端缺失或主机不可达 | 报告网络问题，不要索要密码 |

## 第三步：向用户索要 AD 密码

只在退出码为 3 或 4 时索要。提问要短，只说清三件事：

- 要连哪个主机或共享。
- 需要 `digital\<用户名>` 的域密码。
- 密码只用于连接共享，会存到容器内的凭据文件。

示例：

> 访问 `\\ftdytool.hytch.com\FTDY` 需要 `digital\wudd` 的 AD 密码，请发给我。我会存到容器内的凭据文件，之后默认复用。

拿到密码后立即校验并保存：

```bash
printf '%s\n' '<密码>' | smb-auth set --user-id <id> --host <HOST>
```

- 成功时输出 `STORED <path>`。
- 被拒绝时输出 `credentials rejected`，不要保存，重新索要。
- 不要在最终回复里复述密码。不要写入记忆。不要在之后的消息里重发这条命令。

## 第四步：访问

用 `check` 返回的凭据路径访问。把路径设为环境变量，避免每次写长参数。

```bash
export SMB_CREDENTIALS=$(smb-auth path --user-id <id>)

smbfs ls   "//HOST/SHARE/目录"
smbfs cat  "//HOST/SHARE/路径/文件"
smbfs get  "//HOST/SHARE/路径/文件" /opt/data/cache/文件
smbfs put  /opt/data/cache/文件 "//HOST/SHARE/路径/文件"
smbfs mkdir //HOST/SHARE/目录
smbfs rm   "//HOST/SHARE/路径/文件"
```

## 凭据复用规则

- 凭据按用户保存：`/opt/data/.smb-creds/<用户名>.ini`，权限 0600。
- 一旦保存，后续访问默认直接复用，不再问密码。
- 只有两种情况重新索要：`check` 返回 4（失效），或用户明确要求更换账号。
- 用户要求更换时：`smb-auth forget --user-id <id>`，再走第三步。
- 用户要切换到别的域账号时，用 `--username` 指定，不要覆盖别人的文件。

## 路径转换规则

1. 反斜杠改为斜杠。
2. 保留开头的两个斜杠。
3. 中文保持 UTF-8，不要 URL 编码。

```
\\ftdytool.hytch.com\FTDY\hq_tool\markdown\梁伟添 离职交接.md
→ //ftdytool.hytch.com/FTDY/hq_tool/markdown/梁伟添 离职交接.md
```

路径含空格时，整条路径加引号。

## 示例：读一个中文名文件

```bash
export SMB_CREDENTIALS=$(smb-auth path --user-id <id>)
smbfs cat "//ftdytool.hytch.com/FTDY/hq_tool/markdown/梁伟添 离职交接.md"
```

## 限制

- 没有挂载，因此没有本地路径。程序不能直接 `open()` 共享文件，必须先 `get`。
- 只支持整文件读写，不支持随机读写。
- 每次调用新建一次 SMB 会话。大量小文件会慢，先批量 `get` 再本地处理。
- 大文件先下载到 `/opt/data/cache/`，不要用 `/tmp`。

## 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `NT_STATUS_LOGON_FAILURE` | 账号或域不对 | 重新 `set` 密码，确认域为 `digital` |
| `NT_STATUS_ACCESS_DENIED` | 没有权限，或路径不存在 | 用上级目录 `smbfs ls` 确认 |
| `NT_STATUS_BAD_NETWORK_NAME` | 共享名拼错 | 确认 `//HOST/SHARE` 中的 SHARE |
| `NT_STATUS_HOST_UNREACHABLE` | 网络不通 | 从容器测 `HOST:445` 是否可达 |
| 中文文件名乱码 | locale 不是 UTF-8 | 确认容器 locale 为 UTF-8 |
| `no credentials` | 没设 `SMB_CREDENTIALS` | 先跑 `smb-auth path` |

## 部署

本技能随镜像分发。镜像内包含：

- `smbfs` 与 `smb-auth`，位于 `/usr/local/bin/`。
- `smbclient`，由 apt 安装。
- 本文件，位于 `/opt/hermes/skills/devops/smb-share-access/`。

容器不需要额外权限，不需要 `CAP_SYS_ADMIN`，不需要宿主预挂载。

身份反查需要 `MATTERMOST_URL` 与 `MATTERMOST_TOKEN`。它们在容器的 `/opt/data/.env` 中。

## 安全

- 不要在回复或日志里回显密码。
- 不要把凭据文件内容贴到对话里。
- 只访问用户明确给出的共享。不要扫描内网主机，不要枚举无关共享。
- 每个用户只用自己的账号。不要代用、借用或猜测别人的凭据。
