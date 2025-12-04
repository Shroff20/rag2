import pandas as pd


def build_rag_prompt(query : str, df_results: pd.DataFrame):


    divider = "-------------------------------------------\n"

    context_text = ""

    for idx in df_results.index:
        context_text += divider
        context_text += "<filename>: " + df_results.loc[idx, "basename"] + "\n"
        context_text += df_results.loc[idx, "document"] + "\n"
        context_text += divider

    template = f"""
You are a technical assistant helping answer questions based strictly on the provided documents.

Guidelines:
- Answer the question based on the context below.
- If the context does not contain the answer, say "I cannot answer this based on the provided documents."
- Use direct quotations when possible, and include the full context.
- If you can answer, at the end of each section, append the citations like this: (source: <filename>).
- Use multiple citations if needed.

<context>
{context_text}
</context>
{divider}
Question: {query}
"""
    return template
