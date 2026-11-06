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
git clone --depth 1 https://github.com/Kamisato-Yuna/hermes-agent.git hermes-clean
cd hermes-clean
```

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
