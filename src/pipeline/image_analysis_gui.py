import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import tifffile
import os

from src.pipeline.segment import SegmentationModule
from src.pipeline.layer_detection import LayerDetectionModule
from src.pipeline.plot import (
    visualize_nodes_by_similarity,
    visualize_classification,
    visualize_graph_overlay
)
from src.utils.reinhard_normalizer import ReinhardNormalizer


class ImageAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Histopathology Image Analysis Pipeline")
        self.root.geometry("1200x600")
        
        self.image_path = None
        self.original_image = None
        self.normalized_image = None
        self.tissue_mask = None
        self.nuclei_data_dict = None
        self.nuclei_mask = None
        self.layer_module = None
        
        self.segmentation_module = None
        self.normalizer = None
        self.colorbar = None
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        control_container = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_container.rowconfigure(0, weight=1)
        control_container.columnconfigure(0, weight=1)
        
        canvas = tk.Canvas(control_container, width=280)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        control_frame = ttk.Frame(canvas)
        
        control_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=control_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        ttk.Label(control_frame, text="Image Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(control_frame, textvariable=self.path_var, width=30)
        path_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(control_frame, text="Browse...", command=self.browse_image).grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=5
        )
        
        ttk.Label(control_frame, text="Example Images:", font=('TkDefaultFont', 9)).grid(
            row=3, column=0, sticky=tk.W, pady=(10, 2)
        )
        
        ttk.Button(control_frame, text="Load Single Layer Example", 
                  command=self.load_single_layer_example).grid(
            row=4, column=0, sticky=(tk.W, tk.E), pady=2
        )
        
        ttk.Button(control_frame, text="Load Multi-Layer Example", 
                  command=self.load_multilayer_example).grid(
            row=5, column=0, sticky=(tk.W, tk.E), pady=2
        )
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=6, column=0, sticky=(tk.W, tk.E), pady=10
        )
        
        ttk.Label(control_frame, text="Pipeline Steps:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=7, column=0, sticky=tk.W, pady=(10, 5)
        )
        
        self.load_btn = ttk.Button(control_frame, text="1. Load Image", command=self.load_image)
        self.load_btn.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.normalize_btn = ttk.Button(control_frame, text="2. Normalize Image", 
                                       command=self.normalize_image, state='disabled')
        self.normalize_btn.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.segment_tissue_btn = ttk.Button(control_frame, text="3. Segment Tissue", 
                                            command=self.segment_tissue, state='disabled')
        self.segment_tissue_btn.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.segment_nuclei_btn = ttk.Button(control_frame, text="4. Segment Nuclei", 
                                            command=self.segment_nuclei, state='disabled')
        self.segment_nuclei_btn.grid(row=11, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.build_graph_btn = ttk.Button(control_frame, text="5. Build Graph", 
                                         command=self.build_graph, state='disabled')
        self.build_graph_btn.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=3)
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=13, column=0, sticky=(tk.W, tk.E), pady=10
        )
        
        ttk.Label(control_frame, text="Visualizations:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=14, column=0, sticky=tk.W, pady=(10, 5)
        )
        
        self.show_original_btn = ttk.Button(control_frame, text="Show Original", 
                                           command=lambda: self.show_visualization('original'),
                                           state='disabled')
        self.show_original_btn.grid(row=15, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_normalized_btn = ttk.Button(control_frame, text="Show Normalized", 
                                             command=lambda: self.show_visualization('normalized'),
                                             state='disabled')
        self.show_normalized_btn.grid(row=16, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_tissue_btn = ttk.Button(control_frame, text="Show Tissue Mask", 
                                         command=lambda: self.show_visualization('tissue'),
                                         state='disabled')
        self.show_tissue_btn.grid(row=17, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_nuclei_btn = ttk.Button(control_frame, text="Show Nuclei", 
                                         command=lambda: self.show_visualization('nuclei'),
                                         state='disabled')
        self.show_nuclei_btn.grid(row=18, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_graph_btn = ttk.Button(control_frame, text="Show Filtered Graph", 
                                        command=lambda: self.show_visualization('graph'),
                                        state='disabled')
        self.show_graph_btn.grid(row=19, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_top_n_graph_btn = ttk.Button(control_frame, text="Show Top-2 Graph", 
                                              command=lambda: self.show_visualization('top_n_graph'),
                                              state='disabled')
        self.show_top_n_graph_btn.grid(row=20, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_axes_btn = ttk.Button(control_frame, text="Show Nuclei Axes", 
                                       command=lambda: self.show_visualization('axes'),
                                       state='disabled')
        self.show_axes_btn.grid(row=21, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_similarity_btn = ttk.Button(control_frame, text="Show Similarity Map", 
                                             command=lambda: self.show_visualization('similarity'),
                                             state='disabled')
        self.show_similarity_btn.grid(row=22, column=0, sticky=(tk.W, tk.E), pady=3)
        
        self.show_classification_btn = ttk.Button(control_frame, text="Show Classification", 
                                                 command=lambda: self.show_visualization('classification'),
                                                 state='disabled')
        self.show_classification_btn.grid(row=23, column=0, sticky=(tk.W, tk.E), pady=3)
        
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=24, column=0, sticky=(tk.W, tk.E), pady=10
        )
        
        self.run_all_btn = ttk.Button(control_frame, text="Run Full Pipeline", 
                                     command=self.run_full_pipeline, state='disabled')
        self.run_all_btn.grid(row=25, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.status_var = tk.StringVar(value="Ready. Please load an image.")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=26, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        display_frame = ttk.LabelFrame(main_frame, text="Display", padding="10")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.ax.text(0.5, 0.5, 'Load an image to begin', 
                    ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def browse_image(self):
        """Open file dialog to select an image."""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.path_var.set(filename)
            self.image_path = filename
    
    def load_single_layer_example(self):
        """Load the single layer example image."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "gui_example_images",
            "E2+DHT_1_M13_3L_0002.tif"
        )
        if os.path.exists(example_path):
            self.path_var.set(example_path)
            self.image_path = example_path
            self.load_image()
        else:
            messagebox.showerror("Error", f"Example image not found at: {example_path}")
    
    def load_multilayer_example(self):
        """Load the multi-layer example image."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "gui_example_images",
            "E2+DHT_1_M13_3L_0008.tif"
        )
        if os.path.exists(example_path):
            self.path_var.set(example_path)
            self.image_path = example_path
            self.load_image()
        else:
            messagebox.showerror("Error", f"Example image not found at: {example_path}")
            
    def load_image(self):
        """Load the selected image."""
        try:
            if not self.path_var.get():
                messagebox.showerror("Error", "Please select an image first.")
                return
            
            self.status_var.set("Loading image...")
            self.root.update()
            
            image_path = self.path_var.get()
            if image_path.lower().endswith(('.tif', '.tiff')):
                self.original_image = tifffile.imread(image_path)
            else:
                self.original_image = np.array(Image.open(image_path))
            
            self.normalized_image = None
            self.tissue_mask = None
            self.nuclei_data_dict = None
            self.nuclei_mask = None
            self.layer_module = None
            
            self.normalize_btn.config(state='normal')
            self.run_all_btn.config(state='normal')
            self.show_original_btn.config(state='normal')
            
            self.show_visualization('original')
            
            self.status_var.set(f"Image loaded: {os.path.basename(image_path)} - Shape: {self.original_image.shape}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image.")
            
    def normalize_image(self):
        """Normalize the loaded image using Reinhard normalization."""
        try:
            if self.original_image is None:
                messagebox.showerror("Error", "Please load an image first.")
                return
            
            self.status_var.set("Normalizing image...")
            self.root.update()
            
            if self.normalizer is None:
                self.normalizer = ReinhardNormalizer()
            
            self.normalized_image = self.normalizer.normalize(self.original_image)
            
            self.segment_tissue_btn.config(state='normal')
            self.show_normalized_btn.config(state='normal')
            
            self.show_visualization('normalized')
            
            self.status_var.set("Image normalized successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to normalize image: {str(e)}")
            self.status_var.set("Error normalizing image.")
            
    def segment_tissue(self):
        """Segment tissue regions in the image."""
        try:
            if self.normalized_image is None:
                messagebox.showerror("Error", "Please normalize the image first.")
                return
            
            self.status_var.set("Segmenting tissue... (this may take a moment)")
            self.root.update()
            
            if self.segmentation_module is None:
                self.status_var.set("Loading segmentation model... (first time only)")
                self.root.update()
                self.segmentation_module = SegmentationModule()
            
            org_res = (self.normalized_image.shape[0], self.normalized_image.shape[1])
            self.tissue_mask = self.segmentation_module.segment_tissue(
                self.normalized_image, org_res=org_res
            )
            
            self.segment_nuclei_btn.config(state='normal')
            self.show_tissue_btn.config(state='normal')
            
            self.show_visualization('tissue')
            
            self.status_var.set("Tissue segmentation completed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to segment tissue: {str(e)}")
            self.status_var.set("Error segmenting tissue.")
            
    def segment_nuclei(self):
        """Segment nuclei in the image."""
        try:
            if self.tissue_mask is None:
                messagebox.showerror("Error", "Please segment tissue first.")
                return
            
            self.status_var.set("Segmenting nuclei... (this may take a moment)")
            self.root.update()
            
            self.nuclei_data_dict = self.segmentation_module.segment_nuclei(
                self.normalized_image
            )
            
            self.build_graph_btn.config(state='normal')
            self.show_nuclei_btn.config(state='normal')
            
            self.show_visualization('nuclei')
            
            self.status_var.set(f"Nuclei segmentation completed. Found {len(self.nuclei_data_dict['points'])} nuclei.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to segment nuclei: {str(e)}")
            self.status_var.set("Error segmenting nuclei.")
            
    def build_graph(self):
        """Build the neighbor graph and classify nuclei."""
        try:
            if self.nuclei_data_dict is None:
                messagebox.showerror("Error", "Please segment nuclei first.")
                return
            
            self.status_var.set("Building graph and classifying nuclei... (this may take a moment)")
            self.root.update()
            
            self.layer_module = LayerDetectionModule(
                image=self.normalized_image,
                nuclei_data_dict=self.nuclei_data_dict,
                tissue_mask=self.tissue_mask,
                area_th=0.5
            )
            
            self.show_graph_btn.config(state='normal')
            self.show_top_n_graph_btn.config(state='normal')
            self.show_axes_btn.config(state='normal')
            self.show_similarity_btn.config(state='normal')
            self.show_classification_btn.config(state='normal')
            
            self.show_visualization('graph')
            
            n_organized = sum(1 for c in self.layer_module.classifications.values() if c == "organized")
            n_unorganized = len(self.layer_module.classifications) - n_organized
            
            self.status_var.set(
                f"Graph built. Classified: {n_organized} organized, {n_unorganized} unorganized nuclei."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build graph: {str(e)}")
            self.status_var.set("Error building graph.")
            
    def show_visualization(self, viz_type):
        """Display the requested visualization."""
        try:
            if self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except Exception:
                    pass
                self.colorbar = None
            
            self.ax.clear()
            self.ax.axis('off')
            
            if viz_type == 'original':
                if self.original_image is None:
                    messagebox.showerror("Error", "No original image loaded.")
                    return
                self.ax.imshow(self.original_image)
                self.ax.set_title("Original Image", fontsize=14, pad=10)
                
            elif viz_type == 'normalized':
                if self.normalized_image is None:
                    messagebox.showerror("Error", "Image not normalized yet.")
                    return
                self.ax.imshow(self.normalized_image)
                self.ax.set_title("Normalized Image", fontsize=14, pad=10)
                
            elif viz_type == 'tissue':
                if self.tissue_mask is None:
                    messagebox.showerror("Error", "Tissue not segmented yet.")
                    return
                self.ax.imshow(self.normalized_image)
                self.ax.imshow(self.tissue_mask, alpha=0.5, cmap='jet')
                self.ax.set_title("Tissue Segmentation", fontsize=14, pad=10)
                
            elif viz_type == 'nuclei':
                if self.nuclei_data_dict is None:
                    messagebox.showerror("Error", "Nuclei not segmented yet.")
                    return
                self.ax.imshow(self.normalized_image)
                points = self.nuclei_data_dict["points"]
                n_points = len(points)
                random_values = np.random.rand(n_points)
                self.ax.scatter(points[:, 1], points[:, 0], 
                               c=random_values, cmap='tab20', s=5, alpha=0.8)
                self.ax.set_title(f"Nuclei Segmentation ({n_points} nuclei)", fontsize=14, pad=10)
                
            elif viz_type == 'graph':
                if self.layer_module is None:
                    messagebox.showerror("Error", "Graph not built yet.")
                    return
                visualize_graph_overlay(
                    self.normalized_image, 
                    self.layer_module.filtered_graph,
                    node_size=15,
                    edge_width=1.0,
                    ax=self.ax
                )
                self.ax.set_title(f"Filtered Graph Overlay ({self.layer_module.filtered_graph.number_of_edges()} edges)", fontsize=14, pad=10)
                
            elif viz_type == 'top_n_graph':
                if self.layer_module is None:
                    messagebox.showerror("Error", "Graph not built yet.")
                    return
                visualize_graph_overlay(
                    self.normalized_image, 
                    self.layer_module.top_n_filtered_graph,
                    node_size=15,
                    edge_width=1.0,
                    ax=self.ax
                )
                self.ax.set_title(f"Top-2 Filtered Graph ({self.layer_module.top_n_filtered_graph.number_of_edges()} edges)", fontsize=14, pad=10)
                
            elif viz_type == 'axes':
                if self.layer_module is None:
                    messagebox.showerror("Error", "Graph not built yet.")
                    return
                from src.pipeline.plot import visualize_nuclei_axes
                axes = self.layer_module._get_axis_for_nuclei(self.layer_module.boundary_points)
                visualize_nuclei_axes(
                    self.normalized_image,
                    self.nuclei_data_dict["points"],
                    axes,
                    line_length=25.0,
                    point_size=15,
                    ax=self.ax
                )
                self.ax.set_title("Nuclei Orientation Axes", fontsize=14, pad=10)
                
            elif viz_type == 'similarity':
                if self.layer_module is None:
                    messagebox.showerror("Error", "Graph not built yet.")
                    return
                
                self.ax.imshow(self.normalized_image)
                
                similarities = []
                positions = []
                
                for node in self.layer_module.filtered_graph.nodes():
                    similarity = self.layer_module.filtered_graph.nodes[node].get('best_similarity', 0.0)
                    pos = self.layer_module.filtered_graph.nodes[node]['pos']
                    similarities.append(similarity)
                    positions.append([pos[0], pos[1]])
                
                if len(similarities) > 0:
                    similarities = np.array(similarities)
                    positions = np.array(positions)
                    
                    scatter = self.ax.scatter(positions[:, 1], positions[:, 0], 
                                            s=15, c=similarities, 
                                            cmap='coolwarm', alpha=0.7, zorder=5, vmin=0, vmax=1)
                    
                    self.colorbar = self.fig.colorbar(scatter, ax=self.ax, fraction=0.03, pad=0.02)
                    self.colorbar.set_label('Best Similarity', rotation=270, labelpad=15)
                
                self.ax.set_title("Similarity Map", fontsize=14, pad=10)
                
            elif viz_type == 'classification':
                if self.layer_module is None:
                    messagebox.showerror("Error", "Graph not built yet.")
                    return
                visualize_classification(
                    self.normalized_image,
                    self.layer_module.filtered_graph,
                    self.layer_module.classifications,
                    node_size=15,
                    ax=self.ax
                )
                self.ax.set_title("Nuclei Classification", fontsize=14, pad=10)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show visualization: {str(e)}")
            
    def run_full_pipeline(self):
        """Run the complete pipeline from start to finish."""
        try:
            if self.original_image is None:
                messagebox.showerror("Error", "Please load an image first.")
                return
            
            self.normalize_image()
            self.segment_tissue()
            self.segment_nuclei()
            self.build_graph()
            
            self.show_visualization('classification')
            
        except Exception as e:
            messagebox.showerror("Error", f"Pipeline failed: {str(e)}")
    
    def on_closing(self):
        """Handle window closing event."""
        try:
            plt.close('all')
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = ImageAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

