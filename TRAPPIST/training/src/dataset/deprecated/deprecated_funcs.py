if standardize_coords:
    # Standardize the coordinates of the hits
    coord_cart_hits, scaler = standardize_coordinates(coord_cart_hits)
    coord_cart_hits_norm, scaler_norm = standardize_coordinates(coord_cart_hits_norm)
    pos_xyz_hits, scaler_norm_xyz = standardize_coordinates(pos_xyz_hits)
    if scaler_norm is not None:
        y_coords_std = scaler_norm.transform(y_data_graph[:, :3])
        y_data_graph[:, :3] = torch.tensor(y_coords_std).float()


# in create_inputs_from_table
coord_cart_hits = spherical_to_cartesian(theta, phi, r, normalized=False)
coord_cart_hits_norm = spherical_to_cartesian(theta, phi, r, normalized=True)


tracks = (hit_type_feature == 0) | (hit_type_feature == 1)
# no_tracks = ~tracks
# no_tracks[0] = True

# theta = pf_features_hits[:, 0]
# phi = pf_features_hits[:, 1]
# r = p_hits.view(-1)