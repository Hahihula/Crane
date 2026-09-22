#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sudo ./scripts/register-crane-service.sh SERVICE_NAME CRANE_SERVE_ARGS...

Examples:
  sudo ./scripts/register-crane-service.sh crane-llm \
    --model-path checkpoints/model.gguf --port 8080

  sudo env SERVICE_GPU=1 ./scripts/register-crane-service.sh crane-asr \
    --model-path checkpoints/asr-model --model-type ASR_TYPE --port 8081

  sudo env SERVICE_GPU=2 ./scripts/register-crane-service.sh crane-tts \
    --model-path checkpoints/tts-model --model-type TTS_TYPE --port 8082

Optional environment variables:
  SERVICE_USER       User that runs the service (default: invoking sudo user)
  SERVICE_GPU        CUDA_VISIBLE_DEVICES value (default: unset)
  CRANE_SERVE_BIN    crane-serve executable (default: target/release/crane-serve)
  SERVICE_WORKDIR    Working directory (default: repository root)
EOF
}

if [[ ${EUID} -ne 0 ]]; then
    echo "error: run this script with sudo" >&2
    exit 1
fi

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

service_name=$1
shift

if [[ ! ${service_name} =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]]; then
    echo "error: invalid service name: ${service_name}" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
work_dir=${SERVICE_WORKDIR:-${repo_dir}}
serve_bin=${CRANE_SERVE_BIN:-${repo_dir}/target/release/crane-serve}
service_user=${SERVICE_USER:-${SUDO_USER:-}}

if [[ -z ${service_user} || ${service_user} == root ]]; then
    echo "error: set SERVICE_USER to a non-root deployment user" >&2
    exit 2
fi

if [[ ! -x ${serve_bin} ]]; then
    echo "error: executable not found: ${serve_bin}" >&2
    exit 2
fi

if [[ ! -d ${work_dir} ]]; then
    echo "error: working directory not found: ${work_dir}" >&2
    exit 2
fi

# Quote one argument using systemd's double-quoted ExecStart syntax.
unit_quote() {
    local value=$1
    [[ ${value} != *$'\n'* && ${value} != *$'\r'* ]] || {
        echo "error: command arguments cannot contain newlines" >&2
        exit 2
    }
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//%/%%}
    printf '"%s"' "${value}"
}

exec_start=$(unit_quote "${serve_bin}")
for arg in "$@"; do
    exec_start+=" $(unit_quote "${arg}")"
done

unit_file="/etc/systemd/system/${service_name}.service"
tmp_file=$(mktemp)
trap 'rm -f -- "${tmp_file}"' EXIT

{
    echo '[Unit]'
    echo "Description=Crane server (${service_name})"
    echo 'After=network-online.target'
    echo 'Wants=network-online.target'
    echo
    echo '[Service]'
    echo 'Type=simple'
    echo "User=${service_user}"
    printf 'WorkingDirectory=%s\n' "${work_dir}"
    if [[ -n ${SERVICE_GPU:-} ]]; then
        printf 'Environment="CUDA_VISIBLE_DEVICES=%s"\n' "${SERVICE_GPU//%/%%}"
    fi
    printf 'ExecStart=%s\n' "${exec_start}"
    echo 'Restart=on-failure'
    echo 'RestartSec=5'
    echo 'TimeoutStopSec=30'
    echo
    echo '[Install]'
    echo 'WantedBy=multi-user.target'
} >"${tmp_file}"

install -m 0644 "${tmp_file}" "${unit_file}"
systemctl daemon-reload
systemctl enable --now "${service_name}.service"

echo "registered and started: ${service_name}.service"
echo "status: systemctl status ${service_name}"
echo "logs:   journalctl -u ${service_name} -f"
