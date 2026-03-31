"""Interactive widgets for exporting AnnData objects."""

from __future__ import annotations

import gc
from typing import Any, Optional

import ipywidgets as widgets
from IPython.display import clear_output, display

from ..io.saver import save_anndata_batch


class SaveAnnDataWidget:
    """Interactive widget for saving AnnData objects with optional memory cleanup.
    
    Allows users to:
    - Save cdata (cell-level AnnData)
    - Optionally save sp_adata (spatial bin AnnData)
    - Remove existing files
    - Delete objects from memory after saving
    """
    
    def __init__(
        self,
        cdata: Any,
        cdata_path: str,
        sp_adata: Optional[Any] = None,
        sp_adata_path: Optional[str] = None,
        runtime_namespace: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize SaveAnnDataWidget.
        
        Args:
            cdata: Cell-level AnnData object (required)
            cdata_path: Output path for cdata
            sp_adata: Spatial bin AnnData object (optional)
            sp_adata_path: Output path for sp_adata (optional)
            runtime_namespace: Runtime namespace (e.g., globals()) for memory cleanup
        """
        self.cdata = cdata
        self.cdata_path = cdata_path
        self.sp_adata = sp_adata
        self.sp_adata_path = sp_adata_path
        self.runtime_namespace = runtime_namespace
        
        self._init_ui_components()
        self.ui = self._build_ui()
    
    def _init_ui_components(self) -> None:
        self.save_sp_w = widgets.Checkbox(
            value=False,
            description="Also save sp_adata (2um bins)",
        )
        self.rm_exist_w = widgets.Checkbox(
            value=True,
            description="Remove existing files",
        )
        self.del_after_w = widgets.Checkbox(
            value=True,
            description="Delete objects after save",
        )
        
        self.btn_save = widgets.Button(
            description="💾 Save h5ad",
            button_style="success",
            icon="save",
        )
        self.output = widgets.Output()
        self.btn_save.on_click(self._on_save)
    
    def _build_ui(self) -> widgets.Widget:
        return widgets.VBox([
            widgets.HTML("<h3>💾 Save AnnData</h3>"),
            widgets.HTML(f"<b>cdata</b> → {self.cdata_path}<br>"
                        f"<b>sp_adata</b> → {self.sp_adata_path or 'N/A'}"),
            widgets.HBox([self.save_sp_w, self.rm_exist_w, self.del_after_w]),
            self.btn_save,
            self.output,
        ])
    
    def _on_save(self, _: Any) -> None:
        with self.output:
            clear_output(wait=True)
            
            if self.cdata is None:
                print("❌ cdata is required.")
                return
            
            adata_dict = {"cdata": self.cdata}
            output_paths = {"cdata": self.cdata_path}
            
            if self.save_sp_w.value and self.sp_adata is not None:
                if not self.sp_adata_path:
                    print("❌ sp_adata_path is required but not provided.")
                    return
                adata_dict["sp_adata"] = self.sp_adata
                output_paths["sp_adata"] = self.sp_adata_path
            
            try:
                save_anndata_batch(
                    adata_dict=adata_dict,
                    output_paths=output_paths,
                    remove_existing=self.rm_exist_w.value,
                    delete_after=False,  # Manual cleanup below
                    verbose=True,
                )
                
                # Memory cleanup if requested
                if self.del_after_w.value:
                    if self.runtime_namespace is not None:
                        if "cdata" in self.runtime_namespace:
                            del self.runtime_namespace["cdata"]
                        if self.save_sp_w.value and "sp_adata" in self.runtime_namespace:
                            del self.runtime_namespace["sp_adata"]
                    
                    gc.collect()
                    print("🧠 Memory freed from runtime namespace.")
                
                print("\n✅ All done!")
            except Exception as error:
                print(f"❌ Error: {error}")
    
    def display(self) -> None:
        display(self.ui)


def create_save_anndata_widget(
    cdata: Any,
    cdata_path: str,
    sp_adata: Optional[Any] = None,
    sp_adata_path: Optional[str] = None,
    runtime_namespace: Optional[dict[str, Any]] = None,
) -> SaveAnnDataWidget:
    """Create SaveAnnDataWidget instance.
    
    Args:
        cdata: Cell-level AnnData object (required)
        cdata_path: Output path for cdata
        sp_adata: Spatial bin AnnData object (optional)
        sp_adata_path: Output path for sp_adata (optional)
        runtime_namespace: Runtime namespace (e.g., globals()) for memory cleanup
        
    Returns:
        SaveAnnDataWidget instance
        
    Example:
        >>> widget = create_save_anndata_widget(
        ...     cdata=cdata,
        ...     cdata_path="results/sample_b2c.h5ad",
        ...     sp_adata=sp_adata,
        ...     sp_adata_path="results/sample_2um.h5ad",
        ...     runtime_namespace=globals(),
        ... )
        >>> widget.display()
    """
    return SaveAnnDataWidget(
        cdata=cdata,
        cdata_path=cdata_path,
        sp_adata=sp_adata,
        sp_adata_path=sp_adata_path,
        runtime_namespace=runtime_namespace,
    )
