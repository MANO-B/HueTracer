import os
import gc
import anndata as ad
import ipywidgets as widgets
from IPython.display import display, clear_output
from huetracer.core.sc_dataset import (
    scan_sc_datasets, detect_dataset_type, read_one, qc_and_filter
)
from huetracer.plotting.sc_qc import plot_qc

class SCPlotWidget:
    """
    Interactive single-cell plot control panel widget for AnnData object.
    """
    def __init__(self, adata):
        import ipywidgets as widgets
        import pandas as pd
        from IPython.display import display, clear_output
        self.adata = adata
        self.all_genes = adata.var_names.astype(str).tolist()
        self.plot_type_w = widgets.Dropdown(
            options=[
                "UMAP: Leiden",
                "UMAP: Total counts",
                "UMAP: Selected genes",
                "Rank: Heatmap",
                "Rank: Dotplot",
                "Rank: Top genes table",
                "Rank: Violin (top genes)",
                "PCA scatter",
                "PCA variance ratio"
            ],
            value="UMAP: Leiden",
            description="Plot type:",
            layout=widgets.Layout(width="400px")
        )
        self.gene_search_w = widgets.Text(
            placeholder="Search gene...",
            description="Search:",
            layout=widgets.Layout(width="600px")
        )
        self.candidate_w = widgets.SelectMultiple(
            options=sorted(self.all_genes),
            layout=widgets.Layout(width="350px", height="220px")
        )
        self.selected_w = widgets.SelectMultiple(
            options=[],
            layout=widgets.Layout(width="350px", height="220px")
        )
        self.add_btn = widgets.Button(description="Add →")
        self.rm_btn = widgets.Button(description="← Remove")
        self.n_top_w = widgets.IntSlider(
            value=10, min=1, max=50,
            description="Top genes:"
        )
        self.draw_btn = widgets.Button(
            description="Draw",
            button_style="success",
            icon="chart-line"
        )
        self.out = widgets.Output()
        self.selected_genes = []
        self._setup_callbacks()
        self._build_ui()

    def _setup_callbacks(self):
        self.gene_search_w.observe(self.filter_genes, names="value")
        self.add_btn.on_click(self.add_gene)
        self.rm_btn.on_click(self.remove_gene)
        self.draw_btn.on_click(self.draw_plot)
        self.filter_genes()

    def _build_ui(self):
        self.ui = widgets.VBox([
            widgets.HTML("<h3>🧬 Single-cell Plot Control Panel</h3>"),
            self.plot_type_w,
            self.n_top_w,
            widgets.HTML("<b>Gene selection (for UMAP Selected genes)</b>"),
            self.gene_search_w,
            widgets.HBox([
                self.candidate_w,
                widgets.VBox([self.add_btn, self.rm_btn]),
                self.selected_w
            ]),
            self.draw_btn,
            self.out
        ])

    def display(self):
        from IPython.display import display
        display(self.ui)

    def filter_genes(self, _=None):
        q = self.gene_search_w.value.upper()
        if q == "":
            self.candidate_w.options = sorted(self.all_genes)
        else:
            self.candidate_w.options = [g for g in self.all_genes if q in g.upper()]

    def add_gene(self, _):
        for g in self.candidate_w.value:
            if g not in self.selected_genes:
                self.selected_genes.append(g)
        self.selected_w.options = self.selected_genes

    def remove_gene(self, _):
        for g in self.selected_w.value:
            if g in self.selected_genes:
                self.selected_genes.remove(g)
        self.selected_w.options = self.selected_genes

    def draw_plot(self, _):
        import scanpy as sc
        import pandas as pd
        from IPython.display import display, clear_output
        with self.out:
            clear_output()
            ptype = self.plot_type_w.value
            adata = self.adata
            if ptype == "UMAP: Leiden":
                sc.pl.umap(adata, color=["leiden"], use_raw=False)
                return
            if ptype == "UMAP: Total counts":
                sc.pl.umap(adata, color=["total_counts"], use_raw=False)
                return
            if ptype == "UMAP: Selected genes":
                if len(self.selected_genes) == 0:
                    print("Select genes first.")
                    return
                sc.pl.umap(adata, color=self.selected_genes, use_raw=False)
                return
            if ptype == "Rank: Heatmap":
                sc.pl.rank_genes_groups_heatmap(
                    adata,
                    use_raw=False,
                    swap_axes=True,
                    n_genes=self.n_top_w.value
                )
                return
            if ptype == "Rank: Dotplot":
                sc.pl.rank_genes_groups_dotplot(
                    adata,
                    groupby="leiden",
                    n_genes=self.n_top_w.value,
                    swap_axes=True
                )
                return
            if ptype == "Rank: Top genes table":
                genes_df = pd.DataFrame(
                    adata.uns["rank_genes_groups"]["names"]
                ).head(self.n_top_w.value)
                display(
                    genes_df.style.set_caption(
                        f"各クラスタで発現変動の大きい遺伝子（上位{self.n_top_w.value}）"
                    )
                )
                return
            if ptype == "Rank: Violin (top genes)":
                genes_df = pd.DataFrame(
                    adata.uns["rank_genes_groups"]["names"]
                )
                top_genes = genes_df.head(1).to_numpy().ravel()
                top_genes = list(dict.fromkeys(top_genes))
                for g in top_genes:
                    sc.pl.violin(
                        adata,
                        g,
                        groupby="leiden",
                        use_raw=False,
                        stripplot=False
                    )
                return
            if ptype == "PCA scatter":
                sc.pl.pca(adata, color="total_counts")
                return
            if ptype == "PCA variance ratio":
                sc.pl.pca_variance_ratio(adata, log=True)
                return

class SCFilterWidget:
    def __init__(self, base_dir, default_params=None):
        style = {"description_width": "160px"}
        w_full = widgets.Layout(width="900px")
        w_mid  = widgets.Layout(width="650px")
        w_small= widgets.Layout(width="260px")

        self.sc_base_dir_w = widgets.Text(
            value=base_dir,
            description="scRNA base dir:",
            style=style, layout=w_full
        )
        self.scan_btn = widgets.Button(description="Scan", button_style="info", icon="search")
        self.sc_list = widgets.SelectMultiple(
            options=[],
            description="Datasets:",
            style=style, layout=widgets.Layout(width="900px", height="180px")
        )
        self.read_mode_w = widgets.Dropdown(
            options=[("Auto", "auto"), ("10x mtx folder", "mtx"), ("10x h5", "h5"), ("h5ad", "h5ad")],
            value="auto",
            description="Read mode:",
            style=style, layout=w_mid
        )
        self.merge_join_w = widgets.Dropdown(
            options=[("intersection (safe)", "inner"), ("union (keep all genes)", "outer")],
            value="inner",
            description="Merge genes:",
            style=style, layout=w_mid
        )
        self.batch_key_w = widgets.Text(value="batch", description="batch_key:", style=style, layout=w_small)

        self.d_min_genes_w   = widgets.IntText(value=500,  description="min_genes per cell:", style=style, layout=w_small)
        self.d_min_cells_w   = widgets.IntText(value=50,   description="min_cells per gene:", style=style, layout=w_small)
        self.d_min_counts_w  = widgets.IntText(value=2000, description="min_counts per cell:", style=style, layout=w_small)
        self.d_max_counts_w  = widgets.IntText(value=20000,description="max_counts per cell:", style=style, layout=w_small)
        self.d_mt_prefix_w   = widgets.Text(value="MT-",   description="MT gene prefix:", style=style, layout=w_small)
        self.d_max_pct_mt_w  = widgets.FloatText(value=5.0,description="max % MT:", style=style, layout=w_small)

        self.override_box = widgets.HTML("<b>Per-sample override (select ONE dataset below)</b>")
        self.ov_target_dd = widgets.Dropdown(options=[], description="Target:", style=style, layout=w_full)
        self.use_override_w = widgets.Checkbox(value=True, description="Enable override for this target", indent=False)
        self.ov_min_genes_w  = widgets.IntText(value=500,  description="min_genes:", style=style, layout=w_small)
        self.ov_min_cells_w  = widgets.IntText(value=50,   description="min_cells:", style=style, layout=w_small)
        self.ov_min_counts_w = widgets.IntText(value=2000, description="min_counts:", style=style, layout=w_small)
        self.ov_max_counts_w = widgets.IntText(value=20000,description="max_counts:", style=style, layout=w_small)
        self.ov_mt_prefix_w  = widgets.Text(value="MT-",   description="MT prefix:", style=style, layout=w_small)
        self.ov_max_pct_mt_w = widgets.FloatText(value=5.0,description="max % MT:", style=style, layout=w_small)
        self.apply_override_btn = widgets.Button(description="Apply override to target", button_style="warning", icon="sliders")
        self.clear_override_btn = widgets.Button(description="Clear override for target", button_style="", icon="trash")
        self.run_btn = widgets.Button(description="Load → QC → Filter → Merge", button_style="success", icon="play")
        self.qc_btn  = widgets.Button(description="Show QC plots (merged)", button_style="primary", icon="bar-chart")
        self.out = widgets.Output()

        self.overrides = {}
        self._setup_callbacks()
        self._build_ui()

    def _default_params(self):
        return dict(
            min_genes=int(self.d_min_genes_w.value),
            min_cells=int(self.d_min_cells_w.value),
            min_counts=int(self.d_min_counts_w.value),
            max_counts=int(self.d_max_counts_w.value),
            mt_prefix=str(self.d_mt_prefix_w.value),
            max_pct_mt=float(self.d_max_pct_mt_w.value),
        )

    def _override_params_from_widgets(self):
        return dict(
            min_genes=int(self.ov_min_genes_w.value),
            min_cells=int(self.ov_min_cells_w.value),
            min_counts=int(self.ov_min_counts_w.value),
            max_counts=int(self.ov_max_counts_w.value),
            mt_prefix=str(self.ov_mt_prefix_w.value),
            max_pct_mt=float(self.ov_max_pct_mt_w.value),
        )

    def _sync_override_editor_from_selected_target(self, target_path):
        base = self._default_params()
        p = self.overrides.get(target_path, base)
        self.ov_min_genes_w.value  = p["min_genes"]
        self.ov_min_cells_w.value  = p["min_cells"]
        self.ov_min_counts_w.value = p["min_counts"]
        self.ov_max_counts_w.value = p["max_counts"]
        self.ov_mt_prefix_w.value  = p["mt_prefix"]
        self.ov_max_pct_mt_w.value = p["max_pct_mt"]
        self.use_override_w.value = (target_path in self.overrides)

    def _setup_callbacks(self):
        self.scan_btn.on_click(self.do_scan)
        self.apply_override_btn.on_click(self.apply_override)
        self.clear_override_btn.on_click(self.clear_override)
        self.run_btn.on_click(self.do_run)
        self.qc_btn.on_click(self.do_qc_plots)
        self.ov_target_dd.observe(self.on_target_change, names="value")

    def _build_ui(self):
        self.ui = widgets.VBox([
            widgets.HTML("<h3>🧬 scRNA-seq: per-sample filtering supported (override)</h3>"),
            widgets.HBox([self.sc_base_dir_w, self.scan_btn]),
            self.sc_list,
            widgets.HBox([self.read_mode_w, self.merge_join_w]),
            widgets.HBox([self.batch_key_w]),
            widgets.HTML("<hr><h4>Default Filters</h4>"),
            widgets.HBox([self.d_min_genes_w, self.d_min_cells_w, self.d_mt_prefix_w, self.d_max_pct_mt_w]),
            widgets.HBox([self.d_min_counts_w, self.d_max_counts_w]),
            widgets.HTML("<hr><h4>Per-sample Override</h4>"),
            self.ov_target_dd,
            self.use_override_w,
            widgets.HBox([self.ov_min_genes_w, self.ov_min_cells_w, self.ov_mt_prefix_w, self.ov_max_pct_mt_w]),
            widgets.HBox([self.ov_min_counts_w, self.ov_max_counts_w]),
            widgets.HBox([self.apply_override_btn, self.clear_override_btn]),
            widgets.HTML("<hr>"),
            widgets.HBox([self.run_btn, self.qc_btn]),
            self.out
        ])

    def display(self):
        display(self.ui)
        self.do_scan()

    def do_scan(self, _=None):
        with self.out:
            clear_output()
            base_dir = self.sc_base_dir_w.value
            print("🔍 scanning:", base_dir)
            hits = scan_sc_datasets(base_dir)
            if not hits:
                self.sc_list.options = []
                self.ov_target_dd.options = []
                print("⚠️ no datasets found")
                return
            base_abs = os.path.abspath(os.path.expanduser(base_dir))
            opts = []
            for p in hits:
                typ = detect_dataset_type(p)
                rel = os.path.relpath(p, base_abs) if p.startswith(base_abs) else p
                opts.append((f"[{typ}] {rel}", p))
            self.sc_list.options = opts
            self.ov_target_dd.options = opts
            self.ov_target_dd.value = opts[0][1]
            self._sync_override_editor_from_selected_target(self.ov_target_dd.value)
            print(f"✅ found {len(opts)} candidates")

    def on_target_change(self, change):
        if change["name"] == "value" and change["new"]:
            self._sync_override_editor_from_selected_target(change["new"])

    def apply_override(self, _=None):
        with self.out:
            clear_output()
            t = self.ov_target_dd.value
            if not t:
                print("⚠️ no target")
                return
            if self.use_override_w.value:
                self.overrides[t] = self._override_params_from_widgets()
                print("✅ override set for:", t)
                print(self.overrides[t])
            else:
                self.overrides.pop(t, None)
                print("✅ override disabled for:", t)

    def clear_override(self, _=None):
        with self.out:
            clear_output()
            t = self.ov_target_dd.value
            self.overrides.pop(t, None)
            self._sync_override_editor_from_selected_target(t)
            print("✅ cleared override for:", t)

    def do_run(self, _=None):
        with self.out:
            clear_output()
            selected = list(self.sc_list.value)
            if not selected:
                print("⚠️ select at least one dataset")
                return
            mode = self.read_mode_w.value
            join = self.merge_join_w.value
            bkey = self.batch_key_w.value.strip() or "batch"
            print("🚀 loading:", len(selected))
            adatas = []
            for i, pth in enumerate(selected):
                print(f"  [{i}] {pth}")
                a = read_one(pth, mode=mode)
                a.var_names_make_unique()
                adatas.append(a)
            print("🧪 QC + filtering (default + overrides)...")
            filtered = []
            for i, (pth, a) in enumerate(zip(selected, adatas)):
                base = self._default_params()
                p = self.overrides.get(pth, base)
                before = (a.n_obs, a.n_vars)
                a2 = qc_and_filter(a, p)
                after = (a2.n_obs, a2.n_vars)
                tag = "override" if pth in self.overrides else "default"
                print(f"  [{i}] {tag} {before} -> {after}  ({os.path.basename(pth.rstrip('/'))})")
                filtered.append(a2)
            print(f"🔗 merging: join={join}, batch_key={bkey}")
            if len(filtered) == 1:
                merged = filtered[0].copy()
                merged.obs[bkey] = "0"
            else:
                merged = ad.concat(
                    filtered,
                    join=join,
                    label=bkey,
                    keys=[str(i) for i in range(len(filtered))],
                    index_unique="-",
                )
            print("📦 set layers['counts'] and raw")
            merged.layers["counts"] = merged.X.copy()
            merged.raw = merged
            globals()["sc_adata_merged"] = merged
            globals()["sc_filter_overrides"] = self.overrides
            del adatas, filtered
            gc.collect()
            print("✅ done")
            print("sc_adata_merged:", merged.shape)
            print("overrides:", len(self.overrides))
            print("Data load/filter finished.")
            print("Done! Go ahead.")

    def do_qc_plots(self, _=None):
        with self.out:
            if "sc_adata_merged" not in globals():
                print("⚠️ sc_adata_merged not found. Run first.")
                return
            a = globals()["sc_adata_merged"]
            clear_output()
            plot_qc(a)

    def get_merged_adata(self):
        if "sc_adata_merged" in globals():
            return globals()["sc_adata_merged"]
        else:
            print("⚠️ sc_adata_merged not found. Run first.")
            return None


def show_sc_filter_save_widget(
    sc_adata_merged,
    annotation_dict,
    config_dict,
):
    """Show widget to filter, plot and save single-cell AnnData with optional config update."""
    sample_name = config_dict.get("SAMPLE_NAME")
    results_path = config_dict.get("RESULTS_PATH")
    b2c_config_save_path = config_dict.get("B2C_CONFIG_SAVE_PATH")

    sc_filtered_path = os.path.join(results_path, sample_name + "_single_cell_filtered.h5ad")
    save_plot_w = widgets.Checkbox(value=True, description="Show UMAP plots (leiden/cell_type)")
    rm_exist_w = widgets.Checkbox(value=True, description="Remove existing file")
    del_after_w = widgets.Checkbox(value=False, description="Delete filtered_sc_adata after save")
    write_celltype_w = widgets.Checkbox(value=True, description="Write .obs['cell_type'] from annotation_dict")
    exclude_w = widgets.Text(
        value="Doublet,Other",
        description="Exclude cell_type:",
        layout=widgets.Layout(width="420px"),
    )
    btn_run = widgets.Button(description="🧬 Filter + Plot + Save", button_style="success")
    btn_update_config = widgets.Button(description="📝 Update Config", button_style="info")
    out = widgets.Output()
    out_config = widgets.Output()

    def on_run(_):
        with out:
            clear_output(wait=True)
            if sc_adata_merged is None:
                print("❌ sc_adata_merged not found.")
                return
            if write_celltype_w.value and annotation_dict is None:
                print("❌ annotation_dict not found (needed for cell_type mapping).")
                return
            os.makedirs(results_path, exist_ok=True)
            print("=== Single-cell filtering & saving ===")
            print("save ->", sc_filtered_path)
            print()

            from huetracer.io.sc_filter_save import save_filtered_sc_adata

            result = save_filtered_sc_adata(
                ad=sc_adata_merged,
                annotation_dict=annotation_dict,
                sc_filtered_path=sc_filtered_path,
                exclude_labels=[x.strip() for x in exclude_w.value.split(",") if x.strip()],
                write_celltype=write_celltype_w.value,
                show_plots=save_plot_w.value,
                remove_existing=rm_exist_w.value,
                delete_after=del_after_w.value,
            )

            print(f"✅ filtered: {result['n_kept']} / {result['n_total']} cells kept")
            print("   excluded labels:", result["excluded"])
            if result.get("missing_clusters"):
                print("⚠️ Unmapped leiden clusters -> 'Other':", result["missing_clusters"])
            print(f"✅ saved ({result['file_size']})")
            print(f"\nDone. Total time: {result['elapsed']:.2f} sec")

    def on_update_config(_):
        with out_config:
            clear_output(wait=True)
            if b2c_config_save_path is None:
                print("❌ B2C_CONFIG_SAVE_PATH not provided.")
                return

            print("=== Updating config file ===")
            print("Config path:", b2c_config_save_path)

            from huetracer.io.config import update_config_file

            config = config_dict.copy()
            config["sc_filtered_path"] = sc_filtered_path
            config["annotation_dict"] = annotation_dict
            update_config_file(b2c_config_save_path, config, merge=True)

            print("✅ Config updated successfully!")
            print(f"   - sc_filtered_path: {sc_filtered_path}")
            print(f"   - annotation_dict keys: {len(annotation_dict) if annotation_dict else 0}")
            print("\nYou can verify with:")
            print(f"   cat {b2c_config_save_path}")

    btn_run.on_click(on_run)
    btn_update_config.on_click(on_update_config)
    display(
        widgets.VBox([
            widgets.HTML("<h3>🧬 Single-cell: filter (remove Doublet etc.) + save h5ad</h3>"),
            write_celltype_w,
            save_plot_w,
            widgets.Label(""),
            exclude_w,
            rm_exist_w,
            del_after_w,
            btn_run,
            out,
            widgets.HTML("<hr><h4>📝 Config file update</h4>"),
            btn_update_config,
            out_config,
        ])
    )
