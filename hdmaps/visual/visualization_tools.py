import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def embed_vs_actual(
    diffusion_coords,
    data_samples,
    num_samples=4,
    embed_scale=0.00001,
    actual_scale=0.001,
):
    sample_length = data_samples[0].shape[0]
    print(f"Sample length: {sample_length}, Amount: {num_samples}")
    tab20 = plt.colormaps.get_cmap("tab20").resampled(20)
    set1 = plt.colormaps.get_cmap("Set1").resampled(9)
    set2 = plt.colormaps.get_cmap("Set2").resampled(8)
    set3 = plt.colormaps.get_cmap("Set3").resampled(12)
    pastel1 = plt.colormaps.get_cmap("Pastel1").resampled(9)
    pastel2 = plt.colormaps.get_cmap("Pastel2").resampled(8)
    dark2 = plt.colormaps.get_cmap("Dark2").resampled(8)
    accent = plt.colormaps.get_cmap("Accent").resampled(8)

    # Extract colors and combine
    all_colors = []
    for cmap, n in [
        (tab20, 20),
        (set1, 9),
        (set2, 8),
        (set3, 12),
        (pastel1, 9),
        (pastel2, 8),
        (dark2, 8),
        (accent, 8),
    ]:
        all_colors.extend([cmap(i) for i in range(n)])
    colors_dynamic = all_colors[:sample_length]
    combined_cmap = ListedColormap(colors_dynamic)

    wings = list(range(num_samples))
    embedded_wings = len(wings)
    enable_labels = False

    # Create a plotter with 2 subplots (1 row, 2 columns)
    plotter1 = pv.Plotter(shape=(1, 1))
    plotter2 = pv.Plotter(shape=(2, 2))

    right_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    plotter2_text = ["circle", "square", "cylinder", "cone"]

    shape_type2 = [
        pv.Sphere(radius=actual_scale),
        pv.Cube(x_length=actual_scale, y_length=actual_scale, z_length=actual_scale),
        pv.Cylinder(radius=actual_scale, height=2 * actual_scale),
        pv.Cone(radius=actual_scale, height=2 * actual_scale),
    ]
    shape_type1 = [
        pv.Sphere(radius=embed_scale),
        pv.Cube(x_length=embed_scale, y_length=embed_scale, z_length=embed_scale),
        pv.Cylinder(radius=embed_scale, height=2 * embed_scale),
        pv.Cone(radius=embed_scale, height=2 * embed_scale),
    ]

    scalars1 = np.tile(np.arange(sample_length), num_samples)
    for i in wings:
        start_idx = i * sample_length
        end_idx = (i + 1) * sample_length

        # First subplot
        point_cloud1 = pv.PolyData(diffusion_coords[start_idx:end_idx, :3])

        # Add scalars to the original point cloud BEFORE glyphing
        point_cloud1["colors"] = scalars1[:sample_length]

        # Create glyphed mesh
        glyphed1 = point_cloud1.glyph(geom=shape_type1[i], scale=False)

        # Add the glyphed mesh (remove point_size and render_points_as_spheres)
        plotter1.add_mesh(
            glyphed1,
            cmap=combined_cmap,
            clim=[0, sample_length - 1],
            show_scalar_bar=False,
        )

        # Add labels for each point (use original point cloud for labels)
        if enable_labels:
            labels1 = [
                str(j) for j in range(sample_length)
            ]  # Changed i to j to avoid confusion
            plotter1.add_point_labels(
                point_cloud1,
                labels=labels1,
                point_size=20,
                font_size=12,
                always_visible=True,
                text_color="black",
                fill_shape=False,
            )
        plotter1.add_text("Embedded wings", position="upper_left")

        # Second subplot
        plotter2.subplot(right_positions[i][0], right_positions[i][1])

        points_2d = data_samples[i]
        points_3d = np.column_stack([points_2d, np.zeros(len(points_2d))])
        point_cloud2 = pv.PolyData(points_3d)

        # Add scalars to the original point cloud BEFORE glyphing
        scalars2 = np.tile(np.arange(sample_length), num_samples)
        point_cloud2["colors"] = scalars2[:sample_length]

        # Create glyphed mesh
        glyphed2 = point_cloud2.glyph(geom=shape_type2[i], scale=False)

        # Add the glyphed mesh (not the original point_cloud2)
        plotter2.add_mesh(
            glyphed2,
            cmap=combined_cmap,
            clim=[0, sample_length - 1],
            show_scalar_bar=False,
        )

        # Add labels for each point (use original point cloud for labels)
        if enable_labels:
            labels2 = [str(j) for j in range(sample_length)]  # Changed i to j
            plotter2.add_point_labels(
                point_cloud2,
                labels2,
                point_size=20,
                font_size=12,
                always_visible=True,
                text_color="black",
                fill_shape=False,
            )

        plotter2.add_text(f"{plotter2_text[i]}", position="upper_left")

    plotter1.show(interactive=False, auto_close=False)
    plotter2.show()
    pv.close_all()
    del plotter1
    del plotter2
