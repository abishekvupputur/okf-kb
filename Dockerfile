FROM python:3.13-slim
# git: every bundle is a repository (history, restore). tar: bundle export.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
# pip is only needed to build; removing it (and its bundled wheel) drops its vendored, often-outdated
# certifi/urllib3 copies from the runtime image.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y pip && rm -rf /usr/local/lib/python3*/ensurepip
# The embedding model (app/embed.py) is downloaded here, at build time, so the running kb never
# fetches anything; HF_HUB_OFFLINE makes sure of that. One ONNX thread (also set in embed.py).
ENV KB_MODELS=/srv/models HF_HUB_OFFLINE=0 OMP_NUM_THREADS=1
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('Qdrant/stella-en-400M-v5-doc-onnx', cache_dir='/srv/models')" \
 && chmod -R a+rX /srv/models
ENV HF_HUB_OFFLINE=1
COPY app ./app
RUN useradd --system --uid 10002 kb && mkdir -p /data && chown kb /data
# The base image's admin and packaging tools (apt, util-linux, perl, bash, ncurses, …) carry most of
# Trivy's findings, usually with no fix released, and the app never runs them. Only the purge runs
# after this, so they can go (dpkg's database stays consistent, Trivy still sees what is left).
# git keeps working: the local commands kb uses are C; perl and curl only serve git's remote and
# interactive helpers, which kb never calls.
RUN dpkg --purge --force-remove-essential --force-remove-protected --force-depends \
      apt libapt-pkg7.0 adduser debconf perl-base perl perl-modules-5.40 libperl5.40 liberror-perl \
      libcurl3t64-gnutls \
      util-linux mount bsdutils login liblastlog2-2 libsmartcols1 libmount1 libblkid1 libuuid1 \
      libsystemd0 libudev1 bash libreadline8t64 readline-common libncursesw6 libtinfo6 ncurses-bin ncurses-base
USER kb
EXPOSE 8000
CMD ["python", "-m", "app.server"]
