API_KEY=YOUR_OPENROUTER_API_KEY
BASE_URL=https://openrouter.ai/api/v1
MAX_CONCURRENCY=30
BATCH_SIZE=5

MODEL_NAME=deepseek/deepseek-v4-flash

if [ -n "${QA_OUTPUT_DIR}" ]; then
    BASE="${QA_OUTPUT_DIR%/}"
    SRC_DIR="${BASE}/extract"
    DST_DIR="${BASE}/compare"
    LOG_DIR="log/${BASE#result/}/compare"
else
    RESULT_DIR="${RESULT_DIR:-qa_llm_search}"
    SRC_DIR="result/${RESULT_DIR}_extract"
    DST_DIR="result/${RESULT_DIR}_compare"
    LOG_DIR="log/${RESULT_DIR}_compare"
fi

mkdir -p "${DST_DIR}" "${LOG_DIR}"

for INPUT_PATH in "${SRC_DIR}"/*.csv; do
    FILE_BASE=$(basename "${INPUT_PATH}" .csv)
    OUTPUT_PATH="${DST_DIR}/${FILE_BASE}.csv"

    python -u qa_llm_compare.py \
        --api_key "${API_KEY}" \
        --base_url "${BASE_URL}" \
        --input_path "${INPUT_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --max_concurrency "${MAX_CONCURRENCY}" \
        --batch_size "${BATCH_SIZE}" \
        --model_name "${MODEL_NAME}" \
        &> "${LOG_DIR}/${FILE_BASE}.log" &
done
