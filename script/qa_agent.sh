API_KEY=YOUR_OPENROUTER_API_KEY
BASE_URL=https://openrouter.ai/api/v1
MAX_CONCURRENCY=30
SERPER_API_KEY=YOUR_SERPER_API_KEY

INPUT_PATH="${QA_INPUT_PATH:-QA/all.csv}"
STAGE_DIR="${QA_OUTPUT_DIR:-result/qa_agent}"
STAGE_DIR="${STAGE_DIR%/}"
LOG_DIR="log/${STAGE_DIR#result/}"

mkdir -p "${LOG_DIR}" "${STAGE_DIR}"

MODEL_NAME_LIST=(
    google/gemini-3-flash-preview
    qwen/qwen3.6-max-preview
    anthropic/claude-sonnet-4.6
)

for i in "${!MODEL_NAME_LIST[@]}"; do
    MODEL_NAME="${MODEL_NAME_LIST[i]}"
    FILE_NAME="${MODEL_NAME##*/}"
    nohup python -u qa_agent.py \
        --api_key ${API_KEY} \
        --base_url ${BASE_URL} \
        --output_path "${STAGE_DIR}/${FILE_NAME}.csv" \
        --input_path "${INPUT_PATH}" \
        --max_concurrency ${MAX_CONCURRENCY} \
        --model_name ${MODEL_NAME} \
        --search_backend serper \
        --serper_api_key ${SERPER_API_KEY} \
        --return_conversation \
        &> "${LOG_DIR}/${FILE_NAME}.log" &
done