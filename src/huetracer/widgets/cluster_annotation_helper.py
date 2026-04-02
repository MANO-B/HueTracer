"""
Cluster annotation helper for single-cell transcriptomics (LLM-based)
"""
import re
import ast
import json
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, Markdown



# Manual cluster annotation helper (no API)
class ClusterAnnotationHelper:
    """
    Manual cluster annotation UI for single-cell transcriptomics.
    Usage:
        helper = ClusterAnnotationHelper(genes_df)
        helper.show()
        # annotation_dict is available as helper.annotation_dict
    """
    def __init__(self, genes_df):
        self.genes_df = genes_df
        self.annotation_dict = {}
        self.sample_desc_w = widgets.Textarea(
            value="",
            placeholder="例: colon adenocarcinoma (CRC) / paired normal, processing notes, QC notes など自由記載",
            description="Sample",
            layout=widgets.Layout(width="100%", height="110px")
        )
        self.n_top_w = widgets.IntSlider(
            value=min(20, len(genes_df)),
            min=5,
            max=max(5, min(100, len(genes_df))),
            step=1,
            description="Top N",
            continuous_update=False
        )
        self.make_prompt_btn = widgets.Button(description="Generate prompt", button_style="primary")
        self.parse_btn = widgets.Button(description="Save annotation_dict", button_style="success")
        self.prompt_out = widgets.Output()
        self.resp_in = widgets.Textarea(
            value="",
            placeholder="ChatGPTの返答（Python dictだけ）をここに貼り付けて**Save annotation_dict**をクリックしてください。\n例:{'C0':'B_Cell','C1':'T_Cell',...}\n(After you get the answer from ChatGPT, paste the reply (Python dict only) into the **Reply** box in this notebook, then click **Save annotation_dict**.)",
            description="Reply",
            layout=widgets.Layout(width="100%", height="180px")
        )
        self.status_out = widgets.Output()
        self.make_prompt_btn.on_click(self.on_make_prompt)
        self.parse_btn.on_click(self.on_parse)

    @staticmethod
    def _normalize_cluster_name(col):
        s = str(col).strip()
        if re.fullmatch(r"\d+", s):
            return f"C{s}"
        if re.fullmatch(r"C\d+", s):
            return s
        return s

    def genes_df_to_cluster_markers(self, n_top: int = 20):
        df = self.genes_df.head(n_top).copy()
        cluster_markers = {}
        for col in df.columns:
            c = self._normalize_cluster_name(col)
            genes = [str(x) for x in df[col].tolist() if pd.notna(x)]
            cluster_markers[c] = genes
        return cluster_markers

    @staticmethod
    def build_prompt(cluster_markers: dict, sample_desc: str):
        lines = []
        lines.append("You are an expert single-cell transcriptomics annotator.")
        lines.append("Infer likely cell types for each cluster from marker genes.")
        lines.append("")
        lines.append("Return format (IMPORTANT):")
        lines.append("- Return ONLY a Python dict literal. No code blocks. No explanations.")
        lines.append("- Use **single quotes** for all keys and values (e.g., `'C0'`, `'T_Cell'`).")
        lines.append("Keys must match the provided cluster IDs exactly and be quoted with single quotes: `'C0'`, `'C1'`, `'C2'`, ...")
        lines.append("- Values must also be single-quoted strings, using '_' separators (e.g., `'T_Cell'`, `'CAF_ECM_Rich'`).")
        lines.append("- If uncertain, still guess but add suffix `'_uncertain'`.")
        lines.append("")
        lines.append("Sample description (free text):")
        lines.append(sample_desc.strip() if sample_desc.strip() else "(not provided)")
        lines.append("")
        lines.append("Marker genes per cluster (top genes):")
        for c in sorted(cluster_markers.keys(), key=lambda x: int(re.sub(r"\D","",x)) if re.sub(r"\D","",x) else 10**9):
            lines.append(f"- {c}: {', '.join(cluster_markers[c])}")
        lines.append("")
        lines.append("Output example:")
        lines.append("{'C0':'B_Cell','C1':'T_Cell',...}")
        return "\n".join(lines)

    @staticmethod
    def extract_dict_from_text(text: str):
        t = text.strip()
        t = t.replace("‘", "'").replace("’", "'")
        t = t.replace("“", '"').replace("”", '"')
        try:
            obj = ast.literal_eval(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            chunk = m.group(0)
            for parser in ("py", "json"):
                try:
                    if parser == "py":
                        obj = ast.literal_eval(chunk)
                    else:
                        obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        raise ValueError("Could not parse a dict. Paste ONLY a dict like {'C0':'...', ...}.")

    def on_make_prompt(self, _):
        with self.prompt_out:
            self.prompt_out.clear_output()
            cluster_markers = self.genes_df_to_cluster_markers(n_top=self.n_top_w.value)
            prompt = self.build_prompt(cluster_markers, self.sample_desc_w.value)
            display(Markdown("### Copy this prompt into ChatGPT"))
            display(Markdown(f"```\n{prompt}\n```"))

    def on_parse(self, _):
        with self.status_out:
            self.status_out.clear_output()
            try:
                parsed = self.extract_dict_from_text(self.resp_in.value)
                normalized = {self._normalize_cluster_name(k): str(v) for k, v in parsed.items()}
                self.annotation_dict = normalized
                display(Markdown("✅ Saved to `annotation_dict`"))
                display(self.annotation_dict)
            except Exception as e:
                display(Markdown(f"❌ Error: {e}"))

    def show(self):
        ui = widgets.VBox([
            widgets.HTML("<h3>Manual cluster annotation (no API)</h3>"),
            self.sample_desc_w,
            self.n_top_w,
            widgets.HBox([self.make_prompt_btn, self.parse_btn]),
            self.prompt_out,
            self.resp_in,
            self.status_out
        ])
        display(ui)
