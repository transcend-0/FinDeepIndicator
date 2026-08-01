# Pipeline: (qa_llm_search.sh | qa_agent.sh) -> qa_llm_extract.sh -> qa_llm_compare.sh -> compare_numerical.sh
# Usage: bash script/pipeline.sh [search|agent] [--input_path PATH] [--output_dir DIR]
# Examples:
#    bash script/pipeline.sh agent --input_path QA/all.csv --output_dir result/all_agent
#    bash script/pipeline.sh search --input_path QA/all.csv --output_dir result/all_search


STAGE1="${1:-search}"; shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_path) export QA_INPUT_PATH="$2"; shift 2 ;;
        --output_dir) export QA_OUTPUT_DIR="$2";  shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

run_stage() {
    local name="$1"
    echo "=== [$(date '+%F %T')] Running ${name}.sh ==="
    bash -c "source script/${name}.sh; wait"
    echo "=== [$(date '+%F %T')] Finished ${name}.sh ==="
}

case "${STAGE1}" in
    search|qa_llm_search) export RESULT_DIR=qa_llm_search; run_stage qa_llm_search ;;
    agent|qa_agent)       export RESULT_DIR=qa_agent;      run_stage qa_agent ;;
    *) echo "Usage: $0 [search|agent] [--input_path PATH] [--output_dir DIR]" >&2; exit 1 ;;
esac

run_stage qa_llm_extract
run_stage qa_llm_compare
run_stage compare_numerical
