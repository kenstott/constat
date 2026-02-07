Provide the value for this fact.

FACT: {question}

Respond with ONLY valid JSON in this exact format:
{{"{json_key}": <value>}}

The value can be:
- A number (integer or decimal): {{"planets": 8}} or {{"rate": 3.14}}
- A string: {{"country": "United States"}}
- An ISO date: {{"founding_date": "1776-07-04"}}

Examples:
- "planets in solar system" → {{"planets": 8}}
- "average CEO compensation" → {{"avg_ceo_compensation": 15000000}}
- "capital of France" → {{"capital": "Paris"}}
- "US Independence Day" → {{"independence_day": "1776-07-04"}}

For statistics/averages, use typical values. For unknowns, estimate.

YOUR JSON RESPONSE: