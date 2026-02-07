python 3.12+ is not supported.
Install python 3.11 first. On Fedora 42:
dnf5 install python3.11

then as user:
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip uninstall -y numpy
pip install "numpy<2"
pip install tflite-runtime librosa soundfile numpy tqdm



Then maybe a bash function:

_dpd_runner() {
    local model_name="$1"
    shift
    local PROJ_DIR="/home/safari/src/github/DPDFNet"
    local args=()

    # Convert file paths to absolute paths so they work after 'cd'
    for arg in "$@"; do
        # If arg exists as file OR doesn't look like a flag (likely an output path)
        if [[ -e "$arg" ]] || [[ "$arg" != -* ]]; then
            args+=("$(realpath -m "$arg")")
        else
            args+=("$arg")
        fi
    done

    # Run inside a subshell so your terminal stays put
    (
        cd "$PROJ_DIR" && \
        ./venv/bin/python3.11 ./enhance.py \
            --model_name "$model_name" \
            "${args[@]}"
    )
}

dpd() {
    _dpd_runner "dpdfnet2_48khz_hr" "$@"
}

dpd8() {
    _dpd_runner "dpdfnet8" "$@"
}
