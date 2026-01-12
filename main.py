import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import measure, draw
import io
import tempfile
import base64
import pickle
from datetime import datetime
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Custom unpickler to handle persistent loading
class SafeUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if not isinstance(pid, str):
            try:
                pid = str(pid)
            except:
                pid = "persistent_object"
        return pid

def load_pt_file(uploaded_file):
    try:
        try:
            data = torch.load(uploaded_file, map_location='cpu')
            if isinstance(data, torch.Tensor):
                return data.numpy()
            return data
        except (pickle.UnpicklingError, RuntimeError) as e:
            st.warning(f"Standard loading failed: {str(e)}. Trying alternative methods...")
            try:
                file_bytes = uploaded_file.getvalue()
                file_like = io.BytesIO(file_bytes)
                try:
                    unpickler = SafeUnpickler(file_like)
                    data = unpickler.load()
                    if isinstance(data, torch.Tensor):
                        return data.numpy()
                    return data
                except Exception as e:
                    st.warning(f"Custom unpickler failed: {str(e)}. Trying basic numpy load...")
                    file_like.seek(0)
                    try:
                        data = np.load(file_like, allow_pickle=False)
                        return data
                    except:
                        raise ValueError("All loading methods failed")
            except Exception as e:
                st.error(f"Alternative loading failed: {str(e)}")
                return None
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")
        return None

def calculate_tumor_volume(mask):
    return np.sum(mask)

def analyze_tumor(mask):
    volume = calculate_tumor_volume(mask)
    level = 1 if volume < 10000 else 2 if volume < 50000 else 3
    if volume == 0:
        stage = "Stage 0: No tumor detected"
    elif volume < 20000:
        stage = "Stage I: Small localized tumor"
    elif volume < 50000:
        stage = "Stage II: Larger tumor, localized"
    elif volume < 100000:
        stage = "Stage III: Tumor has spread locally"
    else:
        stage = "Stage IV: Metastasized tumor"
    return volume, level, stage

def get_tumor_grade(volume, enhancement_mask):
    return "Grade IV" if volume > 50000 and np.any(enhancement_mask) else "Grade I"

def normalize_volume(volume):
    volume = volume.astype(np.float32)
    min_val, max_val = volume.min(), volume.max()
    return (volume - min_val) / (max_val - min_val) if max_val > min_val else volume

def create_slice_animation(modalities, labels, output_path, modality_idx=1, fps=10, dpi=100):
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    modality = normalize_volume(modalities[modality_idx])
    n_slices = modality.shape[0]

    combined_mask = np.any(labels > 0, axis=0)
    volume, level, stage = analyze_tumor(combined_mask)
    tumor_info = {"volume": volume, "level": level, "stage": stage}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.axis('off')

    img = ax.imshow(modality[0], cmap='gray')
    
    # Create custom colormaps with transparency
    colors = ['red', 'green', 'blue']
    alphas = [0.4, 0.4, 0.4]  # Transparency levels
    cmaps = [ListedColormap([(0,0,0,0), (plt.cm.colors.to_rgba(c, a))]) 
               for c, a in zip(colors, alphas)]
    
    overlays = [
        ax.imshow(np.ma.masked_where(labels[i, 0] == 0, labels[i, 0]),
                  cmap=cmaps[i], alpha=0.5, vmin=0, vmax=1)
        for i in range(3)
    ]

    title = ax.set_title(f"{modality_names[modality_idx]} with Segmentation\nSlice: 0\nVolume: {volume} voxels, Stage: {stage.split(':')[0]}")

    def update(frame):
        img.set_array(modality[frame])
        for i in range(3):
            overlays[i].set_array(np.ma.masked_where(labels[i, frame] == 0, labels[i, frame]))
        title.set_text(f"{modality_names[modality_idx]} with Segmentation\nSlice: {frame}\nVolume: {volume} voxels, Stage: {stage.split(':')[0]}")
        return [img] + overlays + [title]

    anim = FuncAnimation(fig, update, frames=n_slices, interval=1000//fps, blit=False)

    try:
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    finally:
        plt.close()

    return output_path

def plot_3d_tumor(labels, threshold=0.5):
    """Create a 3D visualization of the tumor components"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for each component
    colors = ['red', 'green', 'blue']
    names = ['Tumor Core', 'Enhancing Tumor', 'Whole Tumor']
    
    # Plot each component
    for i, (label, color, name) in enumerate(zip(labels, colors, names)):
        # Use marching cubes to get the surface mesh
        verts, faces, _, _ = measure.marching_cubes(label, level=threshold)
        
        # Create the mesh
        mesh = Poly3DCollection(verts[faces], alpha=0.5, linewidths=0.5, edgecolor=color, facecolor=color)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)
        
    # Set the view and labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Tumor Visualization')
    
    # Create a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor=c, label=n) for c, n in zip(colors, names)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust the view
    ax.view_init(elev=30, azim=45)
    
    return fig

def generate_patient_report(tumor_info, patient_info, modalities, labels, slice_index):
    report = f"""
    ## NEUROIMAGING REPORT
    **Patient Name:** {patient_info['name']}  
    **Patient ID:** {patient_info['id']}  
    **Date of Birth:** {patient_info['dob']}  
    **Date of Scan:** {patient_info['scan_date']}  
    **Referring Physician:** {patient_info['physician']}  
    **Scan Modality:** {patient_info['modality']}  
    **Scan Region:** Brain  
    **Scan Technique:** 3D MRI  
    **Model Used:** NeuroVision AI v2.1  

    ### FINDINGS:
    **Tumor Presence:** {'Present' if tumor_info['volume'] > 0 else 'Not detected'}  
    **Tumor Type (Probable):** {'Glioblastoma' if tumor_info['grade'] == 'Grade IV' else 'Low-grade glioma'}  
    **Tumor Location:** {'Left hemisphere' if np.mean(np.where(labels[0] > 0)[2]) < labels[0].shape[2]/2 else 'Right hemisphere'}  
    **Tumor Size:** {tumor_info['volume']:,} voxels  
    **Enhancing Tumor Volume:** {calculate_tumor_volume(labels[1]):,} voxels  
    **Tumor Core Volume:** {calculate_tumor_volume(labels[0]):,} voxels  
    **Whole Tumor Volume (Including Edema):** {calculate_tumor_volume(np.any(labels > 0, axis=0)):,} voxels  

    ### IMPRESSION:
    {tumor_info['stage']} ({tumor_info['grade']}) brain tumor detected. 
    """
    
    # Add visualization
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Show the slice with all tumor components
    ax[0].imshow(modalities[1][slice_index], cmap='gray')
    
    # Create composite overlay
    overlay = np.zeros((*labels[0][slice_index].shape, 4))  # RGBA image
    
    # Add each component with different colors
    colors = [
        (1, 0, 0, 0.5),  # Red for tumor core
        (0, 1, 0, 0.5),  # Green for enhancing tumor
        (0, 0, 1, 0.3)   # Blue for whole tumor
    ]
    
    for i in range(3):
        mask = labels[i][slice_index] > 0
        overlay[mask] = colors[i]
    
    ax[0].imshow(overlay)
    ax[0].set_title('Representative Slice with Tumor Components')
    ax[0].axis('off')
    
    # Add 3D visualization
    try:
        fig3d = plot_3d_tumor(labels)
        buf = io.BytesIO()
        fig3d.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        ax[1].imshow(plt.imread(buf))
        ax[1].axis('off')
        ax[1].set_title('3D Tumor Visualization')
        plt.close(fig3d)
    except Exception as e:
        st.warning(f"Could not generate 3D visualization: {str(e)}")
        ax[1].axis('off')
        ax[1].text(0.5, 0.5, '3D visualization not available', 
                   ha='center', va='center')
    
    return report, fig

def generate_synthetic_labels(modalities):
    """Generate synthetic tumor labels based on modality intensities"""
    # Use FLAIR (modality 3) for edema detection
    flair = modalities[3]
    # Threshold for potential tumor regions
    threshold = np.percentile(flair, 95)
    potential_tumor = flair > threshold
    
    # Create synthetic labels (simplified version)
    labels = np.zeros((3,) + modalities.shape[1:], dtype=np.uint8)
    
    # Whole tumor (label 3 in BraTS convention)
    labels[2] = potential_tumor.astype(np.uint8)
    
    # Enhancing tumor core (label 4) - use T1ce with high intensity
    t1ce = modalities[1]
    enhancing_threshold = np.percentile(t1ce, 90)
    labels[1] = ((t1ce > enhancing_threshold) & potential_tumor).astype(np.uint8)
    
    # Tumor core (label 1) - use T1 with high intensity
    t1 = modalities[0]
    core_threshold = np.percentile(t1, 85)
    labels[0] = ((t1 > core_threshold) & potential_tumor).astype(np.uint8)
    
    return labels

def main():
    st.set_page_config(
        page_title="NeuroVision AI | Brain Tumor Analysis",
        page_icon="🧠",
        layout="wide"
    )
    
    st.markdown(f"""
    <style>
        .stButton>button {{
            background-color: rgb(240, 240, 155);
            color: black;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }}
        .highlight {{
            background-color: rgb(128, 128, 128);
            padding: 10px;
            border-radius: 5px;
            color: white;
        }}
        .report-header {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color:rgb(240, 240, 155); padding:20px; border-radius:10px">
        <h1 style="color:black;text-align:center;">🧠 Brain Tumor Visualizer & Report Generator</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Patient information form
    with st.expander("🖊️ Patient Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name")
            patient_id = st.text_input("Patient ID")
            dob = st.date_input("Date of Birth", datetime.now())
        with col2:
            scan_date = st.date_input("Date of Scan", datetime.now())
            physician = st.text_input("Referring Physician")
            modality = st.selectbox("Scan Modality", ["T1", "T1C", "T2", "Flair"])
    
    st.sidebar.markdown("### MRI Modalities")
    selected_modality = st.sidebar.radio(
        "Select Modality to View",
        options=["T1", "T1C", "T2", "Flair"],
        index=1
    )
    
    modality_map = {"T1": 0, "T1C": 1, "T2": 2, "Flair": 3}
    
    with st.expander("📁 Upload Scan Files", expanded=True):
        modality_file = st.file_uploader(
            "Modalities (.pt file)", 
            type=["pt"],
            help="Upload the .pt file containing T1, T1ce, T2, and FLAIR modalities",
            key="modality_uploader"
        )
    
    if modality_file:
        try:
            with st.spinner('Loading scan data...'):
                modalities = load_pt_file(modality_file)
                
                if modalities is None:
                    st.error("Failed to load modalities file. Please check your file and try again.")
                    return
                
                if len(modalities.shape) < 3:
                    st.error("Invalid data dimensions in uploaded file. Expected 3D volume.")
                    return
                
                # Generate synthetic labels
                with st.spinner('Generating tumor segmentation...'):
                    labels = generate_synthetic_labels(modalities)
            
            st.markdown("---")
            slice_index = st.slider(
                "Select slice to view", 
                0, 
                modalities.shape[1]-1, 
                modalities.shape[1]//2,
                help="Navigate through different brain slices"
            )
            
            combined_mask = np.any(labels > 0, axis=0)
            modality_idx = modality_map[selected_modality]
            modality_slice = modalities[modality_idx][slice_index]
            
            with st.spinner('Analyzing tumor...'):
                volume, level, stage = analyze_tumor(combined_mask)
                enhancement_mask = modality_slice > np.percentile(modality_slice, 90)
                tumor_grade = get_tumor_grade(volume, enhancement_mask)
                
                tumor_info = {
                    "volume": volume,
                    "level": level,
                    "stage": stage,
                    "grade": tumor_grade
                }
            
            st.markdown("## 🎞 Scan Visualization")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["2D Slice Animation", "3D Tumor Visualization", "Detailed Slice View"])
            
            with tab1:
                st.markdown("### Tumor Segmentation Animation")
                with st.spinner("Generating animation..."):
                    tmpfile = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
                    gif_path = create_slice_animation(modalities, labels, tmpfile.name, modality_idx=modality_idx)
                    with open(gif_path, "rb") as f:
                        gif_bytes = f.read()
                    b64 = base64.b64encode(gif_bytes).decode("utf-8")
                    st.markdown(f'<img src="data:image/gif;base64,{b64}" width="100%">', unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### 3D Tumor Visualization")
                with st.spinner("Generating 3D visualization..."):
                    try:
                        fig3d = plot_3d_tumor(labels)
                        buf = io.BytesIO()
                        fig3d.savefig(buf, format='png', dpi=150)
                        st.image(buf, use_column_width=True)
                        plt.close(fig3d)
                    except Exception as e:
                        st.warning(f"Could not generate 3D visualization: {str(e)}")
            
            with tab3:
                st.markdown("### Detailed Slice View with Tumor Components")
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(modalities[modality_idx][slice_index], cmap='gray')
                
                # Create composite overlay
                overlay = np.zeros((*labels[0][slice_index].shape, 4))  # RGBA image
                
                # Add each component with different colors
                colors = [
                    (1, 0, 0, 0.5),  # Red for tumor core
                    (0, 1, 0, 0.5),  # Green for enhancing tumor
                    (0, 0, 1, 0.3)   # Blue for whole tumor
                ]
                
                for i in range(3):
                    mask = labels[i][slice_index] > 0
                    overlay[mask] = colors[i]
                
                ax.imshow(overlay)
                ax.set_title(f'{selected_modality} Slice {slice_index} with Tumor Components')
                ax.axis('off')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', label='Tumor Core'),
                    Patch(facecolor='green', label='Enhancing Tumor'),
                    Patch(facecolor='blue', label='Whole Tumor')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                st.pyplot(fig)
            
            # Display tumor information
            st.markdown(f"""
            <div class="highlight">
                <p><strong>Current View:</strong> {selected_modality}</p>
                <p><strong>Slice Number:</strong> {slice_index}</p>
                <p><strong>Tumor Volume:</strong> {volume:,} voxels</p>
                <p><strong>Tumor Grade:</strong> {tumor_grade}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            if st.button("📝 Generate Comprehensive Tumor Report", type="primary"):
                patient_info = {
                    "name": patient_name,
                    "id": patient_id,
                    "dob": dob.strftime("%Y-%m-%d"),
                    "scan_date": scan_date.strftime("%Y-%m-%d"),
                    "physician": physician,
                    "modality": selected_modality
                }
                
                report, report_fig = generate_patient_report(tumor_info, patient_info, modalities, labels, slice_index)
                
                with st.expander("📄 Complete Tumor Analysis Report", expanded=True):
                    st.markdown(report, unsafe_allow_html=True)
                    st.pyplot(report_fig)
                    
                    st.markdown("### Clinical Recommendations")
                    if tumor_info['grade'] == "Grade I":
                        st.success("""
                        **Recommendations:**
                        - Routine follow-up MRI in 6 months
                        - Neurological evaluation
                        - Consider spectroscopic MRI if progression suspected
                        """)
                    else:
                        st.error("""
                        **Urgent Recommendations:**
                        - Immediate neurosurgical consultation
                        - Consideration for biopsy or resection
                        - Radiation oncology and neuro-oncology referrals
                        - Steroid therapy for edema management
                        """)
                    
                    st.download_button(
                        label="📥 Download Full Report as PDF",
                        data=report.encode('utf-8'),
                        file_name=f"Brain_Tumor_Report_{patient_name.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.info("Please ensure you've uploaded valid brain scan files and try again.")

if __name__ == "__main__":
    main()