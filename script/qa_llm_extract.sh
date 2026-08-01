API_KEY=YOUR_OPENROUTER_API_KEY
BASE_URL=https://openrouter.ai/api/v1
MAX_CONCURRENCY=30

MODEL_NAME=deepseek/deepseek-v4-flash

EXTRACT_TYPE_LIST=(
    answer
    formula
    data
    data_indicator
)

if [ -n "${QA_OUTPUT_DIR}" ]; then
    BASE="${QA_OUTPUT_DIR%/}"
    SRC_DIR="${BASE}"
    DST_DIR="${BASE}/extract"
    LOG_DIR="log/${BASE#result/}/extract"
else
    RESULT_DIR="${RESULT_DIR:-qa_llm_search}"
    SRC_DIR="result/${RESULT_DIR}"
    DST_DIR="result/${RESULT_DIR}_extract"
    LOG_DIR="log/${RESULT_DIR}_extract"
fi

mkdir -p "${DST_DIR}" "${LOG_DIR}"

for INPUT_PATH in "${SRC_DIR}"/*.csv; do
    FILE_BASE=$(basename "${INPUT_PATH}" .csv)
    OUTPUT_PATH="${DST_DIR}/${FILE_BASE}.csv"

    (
        CURRENT_INPUT="${INPUT_PATH}"
        for EXTRACT_TYPE in "${EXTRACT_TYPE_LIST[@]}"; do
            python -u qa_llm_extract.py \
                --api_key "${API_KEY}" \
                --base_url "${BASE_URL}" \
                --input_path "${CURRENT_INPUT}" \
                --output_path "${OUTPUT_PATH}" \
                --max_concurrency "${MAX_CONCURRENCY}" \
                --model_name "${MODEL_NAME}" \
                --extract_type "${EXTRACT_TYPE}"
            CURRENT_INPUT="${OUTPUT_PATH}"
        done
    ) &> "${LOG_DIR}/${FILE_BASE}.log" &
done
