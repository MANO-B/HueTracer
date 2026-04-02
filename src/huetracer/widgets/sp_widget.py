import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import os
from huetracer.core.reproducibility import set_global_seed, get_seed_from_env

class ScanpyQCPipelineWidget:
    """
    Interactive Scanpy QC / Filtering / Leiden pipeline widget.
    """
    def __init__(self, sp_adata, seed=None):
        self.sp_adata = sp_adata
        self.SEED = get_seed_from_env(default=42) if seed is None else int(seed)
        self.N_THREADS = 1
        self.out = widgets.Output()
        self._init_widgets()
        self._bind_events()
        self._display_ui()
        # Keep an untouched original copy for reset (first time only)
        self.sp_adata_original = sp_adata.copy()
        self.sp_adata_sponly = None

    def _set_reproducibility(self):
        """Apply deterministic settings for repeatable embeddings and clustering."""
        set_global_seed(self.SEED)

        # Keep CPU-level parallelism stable where possible.
        os.environ["OMP_NUM_THREADS"] = str(self.N_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.N_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(self.N_THREADS)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.N_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.N_THREADS)

        sc.settings.n_jobs = self.N_THREADS

    def _init_widgets(self):
        self.w_bin_min = widgets.IntSlider(value=6, min=0, max=200, step=1,
            description="min bin_count (>=)", layout=widgets.Layout(width="520px"))
        self.w_round_counts = widgets.Checkbox(value=True, description="Round X to integers (Seurat v3 HVG)")
        self.w_prefix_mt = widgets.Text(value="MT-", description="MT prefix", layout=widgets.Layout(width="320px"))
        self.w_mt_max = widgets.FloatSlider(value=10.0, min=0.0, max=50.0, step=0.5,
            description="max %MT (<)", layout=widgets.Layout(width="520px"))
        self.w_min_counts = widgets.IntSlider(value=200, min=0, max=200000, step=50,
            description="min total_counts (>=)", layout=widgets.Layout(width="520px"))
        self.w_max_counts = widgets.IntSlider(value=20000, min=0, max=500000, step=500,
            description="max total_counts (<=)", layout=widgets.Layout(width="520px"))
        self.w_qc_top_n = widgets.IntSlider(value=30, min=10, max=100, step=5,
            description="QC: highest_expr n_top", layout=widgets.Layout(width="520px"))
        self.w_mt_hist_max = widgets.FloatSlider(value=10.0, min=1.0, max=50.0, step=0.5,
            description="Plot MT% hist up to", layout=widgets.Layout(width="520px"))
        self.w_hvg_n = widgets.IntSlider(value=2000, min=500, max=10000, step=250,
            description="HVG n_top_genes", layout=widgets.Layout(width="520px"))
        self.w_pca_n = widgets.IntSlider(value=20, min=5, max=100, step=5,
            description="PCA n_comps", layout=widgets.Layout(width="520px"))
        self.w_leiden_res = widgets.FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1,
            description="Leiden resolution", layout=widgets.Layout(width="520px"))
        self.w_leiden_iters = widgets.IntSlider(value=2, min=1, max=10, step=1,
            description="Leiden n_iterations", layout=widgets.Layout(width="520px"))
        self.w_img_key = widgets.Text(value="hires", description="spatial img_key", layout=widgets.Layout(width="320px"))
        self.w_spot_size = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1,
            description="spatial spot_size", layout=widgets.Layout(width="520px"))
        self.w_size = widgets.IntSlider(value=20, min=1, max=200, step=1,
            description="spatial point size", layout=widgets.Layout(width="520px"))
        self.w_make_plots = widgets.Checkbox(value=False, description="Generate QC plots")
        self.btn_run = widgets.Button(description="▶ Run QC + Filtering + Leiden", button_style="success")
        self.btn_reset_raw = widgets.Button(description="↩ Reset sp_adata from original (copy)", button_style="")
        self.btn_help = widgets.Button(description="ℹ Print current params", button_style="info")

    def _bind_events(self):
        self.btn_run.on_click(self.run_pipeline)
        self.btn_reset_raw.on_click(self.reset_from_original)
        self.btn_help.on_click(self.show_params)

    def _display_ui(self):
        display(widgets.VBox([
            widgets.HTML("<h3>🧪 Scanpy QC / Filtering / Leiden pipeline (parameterized)</h3>"),
            widgets.HBox([self.btn_run, self.btn_reset_raw, self.btn_help]),
            widgets.HTML("<b>Early filter</b>"),
            self.w_bin_min,
            self.w_round_counts,
            self.w_prefix_mt,
            widgets.HTML("<b>QC + filters</b>"),
            self.w_min_counts,
            self.w_max_counts,
            self.w_mt_max,
            widgets.HTML("<b>Plots</b>"),
            self.w_qc_top_n,
            self.w_mt_hist_max,
            self.w_make_plots,
            widgets.HTML("<b>HVG / PCA / Leiden</b>"),
            self.w_hvg_n,
            self.w_pca_n,
            self.w_leiden_res,
            self.w_leiden_iters,
            widgets.HTML("<b>Spatial plot settings</b>"),
            self.w_img_key,
            self.w_size,
            self.w_spot_size,
            widgets.HTML("<hr>"),
            self.out
        ]))

    def _print_params(self):
        print("=== Params ===")
        print(f"bin_count >= {self.w_bin_min.value}")
        print(f"round X: {self.w_round_counts.value}")
        print(f"MT prefix: {self.w_prefix_mt.value}")
        print(f"%MT < {self.w_mt_max.value}")
        print(f"total_counts: {self.w_min_counts.value} .. {self.w_max_counts.value}")
        print(f"HVG n_top_genes: {self.w_hvg_n.value}")
        print(f"PCA n_comps: {self.w_pca_n.value}")
        print(f"Leiden resolution: {self.w_leiden_res.value}, n_iterations: {self.w_leiden_iters.value}")
        print(f"spatial img_key: {self.w_img_key.value}, size={self.w_size.value}, spot_size={self.w_spot_size.value}")
        print(f"make plots: {self.w_make_plots.value}")

    def _mt_hist(self, sp, xmax=10.0):
        vals = sp.obs["pct_counts_MT"].to_numpy()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals <= xmax]
        plt.figure(figsize=(7, 4))
        plt.hist(vals, bins=50)
        plt.title(f"pct_counts_MT (<= {xmax})")
        plt.xlabel("pct_counts_MT")
        plt.ylabel("n")
        plt.show()

    def reset_from_original(self, _=None):
        with self.out:
            clear_output(wait=True)
            self.sp_adata = self.sp_adata_original.copy()
            print("✅ sp_adata reset from sp_adata_original (fresh copy). Current n_obs:", self.sp_adata.n_obs)

    def show_params(self, _=None):
        with self.out:
            clear_output(wait=True)
            self._print_params()

    def run_pipeline(self, _=None):
        with self.out:
            clear_output(wait=True)
            plt.close("all")
            self._set_reproducibility()
            print(f"Reproducibility settings: seed={self.SEED}, threads={self.N_THREADS}")
            sp_adata = self.sp_adata
            sp_adata.var_names_make_unique()
            # bin_count filter
            if "bin_count" in sp_adata.obs.columns:
                before = sp_adata.n_obs
                sp_adata = sp_adata[sp_adata.obs["bin_count"] >= self.w_bin_min.value].copy()
                print(f"bin_count filter: {before:,} -> {sp_adata.n_obs:,} (>= {self.w_bin_min.value})")
            else:
                print("⚠️ sp_adata.obs['bin_count'] not found. Skipping bin_count filter.")
            # Round counts if requested
            if self.w_round_counts.value:
                sp_adata.X = np.round(sp_adata.X).copy()
            sp_adata.raw = sp_adata.copy()
            # MT annotation + QC metrics
            prefix = self.w_prefix_mt.value
            sp_adata.var["MT"] = sp_adata.var_names.str.startswith(prefix)
            sc.pp.calculate_qc_metrics(sp_adata, qc_vars=["MT"], percent_top=None, log1p=False, inplace=True)
            print(f"Total cells (after early steps): {sp_adata.n_obs:,}")
            # QC plots (before filtering)
            if self.w_make_plots.value:
                sc.set_figure_params(fontsize=20, figsize=[7, 7])
                sc.pl.highest_expr_genes(sp_adata, n_top=self.w_qc_top_n.value)
                sc.pl.violin(sp_adata, ["n_genes_by_counts", "total_counts", "pct_counts_MT"],
                    jitter=0.4, multi_panel=True)
                sc.pl.scatter(sp_adata, "total_counts", "n_genes_by_counts",
                    color="pct_counts_MT", size=40)
                sc.pl.scatter(sp_adata, x="total_counts", y="pct_counts_MT")
                self._mt_hist(sp_adata, xmax=self.w_mt_hist_max.value)
            # Filtering
            before = sp_adata.n_obs
            sc.pp.filter_cells(sp_adata, min_counts=int(self.w_min_counts.value))
            print(f"min_counts: {before:,} -> {sp_adata.n_obs:,} (>= {self.w_min_counts.value})")
            before = sp_adata.n_obs
            if "total_counts" not in sp_adata.obs.columns:
                sc.pp.calculate_qc_metrics(sp_adata, qc_vars=["MT"], percent_top=None, log1p=False, inplace=True)
            sp_adata = sp_adata[sp_adata.obs["total_counts"] <= int(self.w_max_counts.value)].copy()
            print(f"max_counts: {before:,} -> {sp_adata.n_obs:,} (<= {self.w_max_counts.value})")
            before = sp_adata.n_obs
            sp_adata = sp_adata[sp_adata.obs["pct_counts_MT"] < float(self.w_mt_max.value)].copy()
            print(f"MT% filter: {before:,} -> {sp_adata.n_obs:,} (< {self.w_mt_max.value})")
            if self.w_make_plots.value:
                sc.set_figure_params(fontsize=20, figsize=[7, 7])
                sc.pl.highest_expr_genes(sp_adata, n_top=self.w_qc_top_n.value)
                sc.pl.violin(sp_adata, ["n_genes_by_counts", "total_counts", "pct_counts_MT"],
                    jitter=0.4, multi_panel=True)
                sc.pl.scatter(sp_adata, "total_counts", "n_genes_by_counts",
                    color="pct_counts_MT", size=40)
                sc.pl.scatter(sp_adata, x="total_counts", y="pct_counts_MT")
                self._mt_hist(sp_adata, xmax=self.w_mt_hist_max.value)
            sp_adata.layers["counts"] = sp_adata.X.copy()
            sp_adata.raw = sp_adata
            self.sp_adata_sponly = sp_adata.copy()
            self.sp_adata_sponly.X = self.sp_adata_sponly.raw.X.copy()
            self.sp_adata_sponly.var = self.sp_adata_sponly.raw.var.copy()
            self.sp_adata_sponly.layers["counts"] = self.sp_adata_sponly.X.copy()
            sc.pp.normalize_total(self.sp_adata_sponly)
            sc.pp.log1p(self.sp_adata_sponly)
            sc.pp.highly_variable_genes(
                self.sp_adata_sponly,
                flavor="seurat_v3",
                n_top_genes=int(self.w_hvg_n.value),
                layer="counts",
                subset=False
            )
            print("Performing PCA...")
            sc.tl.pca(self.sp_adata_sponly, svd_solver="arpack",
                mask_var="highly_variable", n_comps=int(self.w_pca_n.value))
            sc.pp.neighbors(self.sp_adata_sponly, random_state=self.SEED)
            print("Performing UMAP...")
            sc.tl.umap(self.sp_adata_sponly, random_state=self.SEED)
            print("Performing Leiden...")
            sc.tl.leiden(
                self.sp_adata_sponly,
                resolution=float(self.w_leiden_res.value),
                flavor="igraph",
                n_iterations=int(self.w_leiden_iters.value),
                key_added="leiden",
                random_state=self.SEED
            )
            print("✅ Embeddings + Leiden done.")
            print("sp_adata_sponly.n_obs:", self.sp_adata_sponly.n_obs)
            sc.pl.umap(self.sp_adata_sponly, color=["leiden"], use_raw=False)
            sc.pl.umap(self.sp_adata_sponly, color=["total_counts"], use_raw=False)
            # Update self.sp_adata with filtered version
            self.sp_adata = sp_adata
            self.sp_adata.obs["leiden_nucleus"] = self.sp_adata_sponly.obs["leiden"].astype(str)
            self.sp_adata.obsm["X_PCA_nucleus"] = self.sp_adata_sponly.obsm["X_pca"].copy()
            self.sp_adata.obsm["X_umap_nucleus"] = self.sp_adata_sponly.obsm["X_umap"].copy()
            print("✅ Copied leiden + embeddings back to sp_adata.")
            print(f"Final Total number of cells: {self.sp_adata.n_obs:,}")
            print("Done! Go ahead.")


class SpatialZoomViewerWidget:
    def __init__(self, sp_adata, basis_key, img_key, color="leiden"):
        self.sp_adata = sp_adata
        self.basis_key = basis_key
        self.img_key = img_key
        self.color = color
        self.XY = sp_adata.obsm[basis_key]
        self.x = self.XY[:, 0]
        self.y = self.XY[:, 1]
        self.xmin, self.xmax = float(np.min(self.x)), float(np.max(self.x))
        self.ymin, self.ymax = float(np.min(self.y)), float(np.max(self.y))
        self.x_range = widgets.FloatRangeSlider(
            value=[self.xmin, self.xmax], min=self.xmin, max=self.xmax, step=(self.xmax-self.xmin)/500,
            description="X range", layout=widgets.Layout(width="650px")
        )
        self.y_range = widgets.FloatRangeSlider(
            value=[self.ymin, self.ymax], min=self.ymin, max=self.ymax, step=(self.ymax-self.ymin)/500,
            description="Y range", layout=widgets.Layout(width="650px")
        )
        self.btn_full = widgets.Button(description="↔ Full", button_style="")
        self.btn_zoom_in = widgets.Button(description="＋ Zoom in (x0.5)", button_style="info")
        self.btn_zoom_out = widgets.Button(description="－ Zoom out (x2)", button_style="info")
        self.btn_draw = widgets.Button(description="👀 Draw", button_style="success")
        self.out = widgets.Output()
        self._bind_events()
        self._display_ui()
        self.draw()

    def _set_ranges(self, x0, x1, y0, y1):
        x0 = max(self.xmin, min(x0, self.xmax)); x1 = max(self.xmin, min(x1, self.xmax))
        y0 = max(self.ymin, min(y0, self.ymax)); y1 = max(self.ymin, min(y1, self.ymax))
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        self.x_range.value = (x0, x1)
        self.y_range.value = (y0, y1)

    def draw(self, _=None):
        with self.out:
            clear_output(wait=True)
            progress_label = widgets.Label('Drawing...')
            display(progress_label)
            plt.close("all")
            sc.set_figure_params(fontsize=20, figsize=[7, 7])
            xr0, xr1 = self.x_range.value
            yr0, yr1 = self.y_range.value
            roi = (self.x >= xr0) & (self.x <= xr1) & (self.y >= yr0) & (self.y <= yr1)
            n = int(np.sum(roi))
            print(f"ROI cells: {n:,}/{self.sp_adata.n_obs:,}")
            if n == 0:
                print("⚠️ Empty ROI. Expand ranges.")
                return
            ad = self.sp_adata[roi].copy()
            try:
                fig = sc.pl.spatial(
                    ad,
                    color=self.color,
                    img_key=self.img_key,
                    basis=self.basis_key,
                    s=4,
                    frameon=False,
                    legend_fontsize=6,
                    title="Clustering for nuclei, sp data only",
                    show=False,
                    return_fig=True,
                )
            except TypeError:
                ax = sc.pl.spatial(
                    ad,
                    color=self.color,
                    img_key=self.img_key,
                    basis=self.basis_key,
                    s=4,
                    frameon=False,
                    legend_fontsize=6,
                    title="Clustering for nuclei, sp data only",
                    show=False,
                )
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            display(fig)
            plt.close(fig)

    def full(self, _=None):
        self._set_ranges(self.xmin, self.xmax, self.ymin, self.ymax)
        self.draw()

    def zoom(self, factor):
        xr0, xr1 = self.x_range.value
        yr0, yr1 = self.y_range.value
        xc = (xr0 + xr1) / 2
        yc = (yr0 + yr1) / 2
        xw = (xr1 - xr0) * factor / 2
        yw = (yr1 - yr0) * factor / 2
        self._set_ranges(xc - xw, xc + xw, yc - yw, yc + yw)
        self.draw()

    def _bind_events(self):
        self.btn_full.on_click(self.full)
        self.btn_zoom_in.on_click(lambda b: self.zoom(0.5))
        self.btn_zoom_out.on_click(lambda b: self.zoom(2.0))
        self.btn_draw.on_click(self.draw)

    def _display_ui(self):
        display(
            widgets.VBox([
                widgets.HTML("<h3>🔎 Spatial Zoom Viewer (ROI by basis ranges)</h3>"),
                widgets.HBox([self.btn_full, self.btn_zoom_in, self.btn_zoom_out, self.btn_draw]),
                self.x_range,
                self.y_range,
                self.out
            ])
        )
