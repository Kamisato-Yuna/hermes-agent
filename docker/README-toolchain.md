# 两层镜像与工具链共享

本文说明 profile 镜像的构建与分发。目标是让多个 profile 容器共用一个工具链层。

## 1. 镜像分层

本仓库使用两个镜像。

| 镜像 | 内容 | 典型大小 |
| --- | --- | --- |
| `hermes-toolchain:<os>-<commit>` | 基础层、apt 依赖、`/opt/hermes/tools` 全部二进制 | 3.2 GB |
| `hermes-gateway:<version>-<commit>` | toolchain 全部层 + `.venv` + 源码 | 4.5 GB |

profile 镜像从 toolchain 镜像继承。两个镜像的同名层完全一致。因此宿主磁盘只保存一份工具链。

第二个 profile 镜像只增加约 1.3 GB。第三个及以后相同。

## 2. 构建

只用干净的克隆构建。不要用开发工作树。开发工作树含 `.codegraph/` 索引缓存和调试文件。`.codegraph/` 约 1.2 GB。

```bash
git clone https://github.com/Kamisato-Yuna/hermes-agent.git hermes-clean
cd hermes-clean
git fetch --tags origin
```

不要用 `--depth 1` 克隆。浅克隆的历史被嫁接，`git describe` 算出的距离错误。版本号也会变成 `vunknown`。

### 2.1 生成构建身份

镜像的版本号来自源码树里的 `install-stamp.json`。该文件被 `.gitignore` 排除，干净克隆里没有。

构建前用仓库自己的代码生成它。

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from hermes_cli.source_stamp import write_source_stamp
d = write_source_stamp(Path('.'))
print(d['displayVersion'], d['commit'][:8])
"
```

输出应类似 `0.22.7+207.g169354c 169354c4`。如果输出 `unknown`，说明 tag 或历史不全。

### 2.2 构建 toolchain 镜像

先建 toolchain 镜像。用 `--target runtime_base` 只构建工具链阶段。

```bash
export P=http://127.0.0.1:18080        # 出网代理，按需
C=$(git rev-parse --short=8 HEAD)

docker build --network host --platform linux/amd64 \
  -f Dockerfile.ubuntu2604 --target runtime_base \
  -t hermes-toolchain:u2604-$C \
  --build-arg HTTP_PROXY=$P --build-arg HTTPS_PROXY=$P \
  --build-arg http_proxy=$P --build-arg https_proxy=$P \
  .
```

再建 profile 镜像。此步骤复用上一步的全部层。

```bash
docker build --network host --platform linux/amd64 \
  -f Dockerfile.ubuntu2604 \
  -t hermes-gateway:firefly-0.22.7-$C \
  --build-arg HTTP_PROXY=$P --build-arg HTTPS_PROXY=$P \
  --build-arg http_proxy=$P --build-arg https_proxy=$P \
  .
```

规则：两个 `-t` 必须来自同一个 commit。否则层不共享。

## 3. 分发

### 3.1 直接传镜像文件

把两个镜像放进同一个 `save` 流。共享层只写一次。

```bash
docker save hermes-toolchain:u2604-$C hermes-gateway:firefly-0.22.7-$C \
  | gzip -1 | ssh <target> "gunzip | docker load"
```

接收方按以下顺序加载。先 toolchain，再 profile。

### 3.2 用 registry

先推 toolchain。后推 profile。接收方先拉 toolchain，再拉 profile。

```bash
docker tag hermes-toolchain:u2604-$C  <registry>/hermes-toolchain:u2604-$C
docker tag hermes-gateway:firefly-0.22.7-$C <registry>/hermes-gateway:firefly-0.22.7-$C
docker push <registry>/hermes-toolchain:u2604-$C
docker push <registry>/hermes-gateway:firefly-0.22.7-$C
```

## 4. 部署

compose 只引用 profile 镜像。不需要额外挂载。工具链已在镜像层内。

```yaml
services:
  hermes-manager:
    image: hermes-gateway:firefly-0.22.7-<commit>
```

### 4.1 旧部署的 plugins 挂载

旧部署把宿主的 `~/.hermes/hermes-agent/plugins` 挂到 `/opt/hermes/plugins`。该目录是上一版 Hermes 的副本。升级镜像后，旧副本会覆盖镜像内的新插件。

后果是插件与核心 API 不匹配。日志出现 `AttributeError: module 'tools.web_tools' has no attribute ...`。

升级后必须同步该目录。或删除该挂载，只用镜像内的插件。

```bash
# 备份旧副本，从新镜像同步
mv ~/.hermes/hermes-agent/plugins ~/.hermes/hermes-agent/plugins.bak-$(date +%Y%m%d)
docker create --name tmp <profile-image>
docker cp tmp:/opt/hermes/plugins ~/.hermes/hermes-agent/plugins
docker rm tmp
docker restart <container>
```

## 5. 验证

先查工具链可用。

```bash
docker run --rm --entrypoint sh <profile-image> -c \
  'hermes --version; ls /usr/local/bin/smbfs; du -sh /opt/hermes'
```

再查层共享。两个镜像的层前缀必须一致。

```bash
docker image inspect -f '{{range .RootFS.Layers}}{{println .}}{{end}}' <toolchain> > /tmp/tc
docker image inspect -f '{{range .RootFS.Layers}}{{println .}}{{end}}' <profile> > /tmp/pf
head -n 17 /tmp/tc | diff - <(head -n 17 /tmp/pf) && echo shared
```

## 6. 约束

- 不要在镜像内改 `/opt/hermes/tools`。改动不进镜像，且破坏层共享。
- 升级 Hermes 代码时，如果 `pyproject.toml` 或工具清单不变，只需重建 profile 镜像。
- 如果工具清单变更，重建 toolchain 镜像。然后重建 profile 镜像。
- `.dockerignore` 必须保留 `.codegraph/` 条目。该目录是本地索引缓存，约 1.2 GB。
