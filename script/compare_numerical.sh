TOLERANCES=(0.01 0.005 0.001)

if [ -n "${QA_OUTPUT_DIR}" ]; then
    BASE="${QA_OUTPUT_DIR%/}"
    SRC_DIR="${BASE}/compare"
    DST_DIR="${BASE}/numerical"
    LOG_DIR="log/${BASE#result/}/numerical"
else
    RESULT_DIR="${RESULT_DIR:-qa_llm_search}"
    SRC_DIR="result/${RESULT_DIR}_compare"
    DST_DIR="result/${RESULT_DIR}_numerical"
    LOG_DIR="log/${RESULT_DIR}_numerical"
fi

mkdir -p "${DST_DIR}" "${LOG_DIR}"

for INPUT_PATH in "${SRC_DIR}"/*.csv; do
    FILE_BASE=$(basename "${INPUT_PATH}" .csv)
    OUTPUT_PATH="${DST_DIR}/${FILE_BASE}.csv"

    (
        if [ -f "${OUTPUT_PATH}" ]; then
            CURRENT_INPUT="${OUTPUT_PATH}"
        else
            CURRENT_INPUT="${INPUT_PATH}"
        fi
        for TOLERANCE in "${TOLERANCES[@]}"; do
            for COMPARE_TYPE in numerical csv; do
                python -u compare_numerical.py \
                    --input_path "${CURRENT_INPUT}" \
                    --output_path "${OUTPUT_PATH}" \
                    --compare_type "${COMPARE_TYPE}" \
                    --tolerance "${TOLERANCE}"
                CURRENT_INPUT="${OUTPUT_PATH}"
            done
        done
    ) &> "${LOG_DIR}/${FILE_BASE}.log" &
done
