You are replanning the remaining steps of an analysis after the user provided input mid-execution.

## Original problem
{problem}

## Completed work so far
{scratchpad_context}

## Existing tables
{existing_tables_list}

## User input received during execution
{user_answer}

## Instructions
Based on the user's input and the work completed so far, plan ONLY the remaining steps needed to complete the analysis.
- Do NOT repeat work already done (see completed work above).
- Do NOT add any user_input steps — the user has already provided their input above. Use it directly.
- Use existing tables where possible.
- The user's input MUST directly inform the approach and logic of the remaining steps. Reference specific values, mappings, or choices the user provided — do not ignore them.
- If the user provided mappings or correlations, the generated code MUST load and apply those mappings from the saved table.
- Number steps starting from {next_step_number}.
