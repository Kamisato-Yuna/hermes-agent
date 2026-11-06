---
name: smb-share-access
description: Read and write Windows SMB shares through smbfs, with no mount and no root.
version: 1.0.0
author: 流萤 (Firefly) for 汐儿
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [smb, windows, ad, file-share, smbclient]
    category: devops
    requires_toolsets: [terminal]
---

# Windows 共享访问（smbfs）

## 何时使用本技能

用户在对话里给出一个 Windows 路径，或以 `\\` 开头的内网共享路径时，使用本技能。
用户要读、写、列目录、上传、下载共享中的文件时，使用本技能。

不要挂载共享。不要创建 `/mnt` 路径。不要要 root。使用 `smbfs`。

## 路径转换规则

1. 反斜杠改为斜杠。
2. 保留开头的两个斜杠。
3. 中文保持 UTF-8，不要做 URL 编码。

```
\\10.99.1.157\部门工作\技术中心\00中心工作\G公共资料
→ //10.99.1.157/部门工作/技术中心/00中心工作/G公共资料
```

`smbfs` 也接受反斜杠，但推荐用斜杠，以避免 shell 转义问题。

路径包含空格时，必须用引号包住整个路径。

## 凭据

凭据文件：`/opt/data/.smb-credentials`（权限 0600）。
用 INI 分节保存多个账号：

```ini
[default]
username=digital\xuanran
password=********
domain=digital

[another]
username=digital\someone
password=********
domain=digital
```

- 不写 `--user` 时使用 `[default]`。
- 指定账号时加 `--user another`。
- 也可以用环境变量 `SMB_USER` / `SMB_PASSWORD` / `SMB_DOMAIN`。
- 用环境变量 `SMB_CREDENTIALS` 可以把凭据文件放到别的位置。

## 命令

```bash
smbfs ls    //HOST/SHARE/目录            # 列目录
smbfs ls    //HOST/SHARE/目录 --long     # 带大小与时间
smbfs cat   //HOST/SHARE/路径/文件       # 打印文件到 stdout
smbfs get   //HOST/SHARE/路径/文件 [本地路径]   # 下载
smbfs put   本地路径 //HOST/SHARE/路径/文件     # 上传
smbfs mkdir //HOST/SHARE/目录
smbfs rmdir //HOST/SHARE/目录
smbfs rm    //HOST/SHARE/路径/文件
```

## 示例

```bash
smbfs ls "//10.99.1.157/部门工作/技术中心/00中心工作/G公共资料/B_周月报/9月/费用"

smbfs get "//10.99.1.157/部门工作/技术中心/00中心工作/G公共资料/B_周月报/9月/费用/费用明细.xlsx" \
          /opt/data/cache/费用明细.xlsx

smbfs put /opt/data/cache/报告.docx "//10.99.1.157/部门工作/技术中心/00中心工作/G公共资料/B_周月报/9月/费用/报告.docx"
```

## 工作流程

1. 把用户给的路径转成 `//HOST/SHARE/...`。
2. 用 `smbfs ls` 确认路径存在。
3. 用 `smbfs get` 把文件下载到 `/opt/data/cache/`。
4. 在本地处理文件。
5. 需要回写时，用 `smbfs put` 上传。
6. 在回复里报告结果，不要回显密码。

## 限制

- 没有挂载，所以没有本地路径。程序不能用 `open()` 直接读共享文件。必须先 `get`。
- 只支持整文件读写。不支持随机读写与大文件分块修改。
- 每次调用都新建一次 SMB 会话。大量小文件操作会慢。可以先 `get` 整个目录的关键文件，再本地处理。
- 大文件先下载到 `/opt/data/cache/`，不要下载到 `/tmp`（可能空间不足）。

## 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `NT_STATUS_LOGON_FAILURE` | 账号或域不对 | 检查 `.smb-credentials` 的 `username`/`domain` |
| `NT_STATUS_ACCESS_DENIED` | 没有权限，或路径不存在 | 用上级目录 `smbfs ls` 确认 |
| `NT_STATUS_BAD_NETWORK_NAME` | 共享名拼错 | 确认 `//HOST/SHARE` 中的 SHARE |
| `NT_STATUS_HOST_UNREACHABLE` | 网络不通 | 先测 `10.99.x.x:445` 是否可达 |
| 中文文件名乱码 | 协议协商问题 | `smbfs` 默认 `-m SMB3`，确认系统 locale 为 UTF-8 |
| `no credentials` | 找不到凭据 | 加 `--user`，或补 `[default]` 节 |

## 新机器部署

1. 安装客户端：`apt-get install -y smbclient`（Debian/Ubuntu）。
2. 写凭据文件：`install -m 600 /dev/null /opt/data/.smb-credentials`，再按上面的格式填入。
3. 验证：`smbfs ls //HOST/SHARE`。

本技能随镜像分发，不需要给容器加任何权限，也不需要 `CAP_SYS_ADMIN`。

## 安全

- 不要在回复或日志中回显密码。
- 不要把 `.smb-credentials` 的内容贴到对话里。
- 只访问用户明确给出的共享。不要扫描内网主机或尝试枚举其他共享。
