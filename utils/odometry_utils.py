import numpy as np



def compute_headings( utm_coords, motion_threshold=1e-3):
    """
    Compute headings from UTM coordinates with improved handling of stationary periods.
    
    Args:
        utm_coords: Array of [x, y] coordinates
        motion_threshold: Minimum distance to consider as actual motion
    
    Returns:
        headings: Array of headings in degrees, same length as input
    """
    if len(utm_coords) < 2:
        return np.zeros(len(utm_coords))
    
    deltas = np.diff(utm_coords, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    
    # Compute raw headings only where there's significant motion
    raw_headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    
    # Create output array
    headings = np.zeros(len(utm_coords))
    
    # Strategy 1: Forward fill valid headings for stationary periods
    valid_motion = distances >= motion_threshold
    
    if np.any(valid_motion):
        # Set headings where there's valid motion
        headings[1:][valid_motion] = raw_headings[valid_motion]
        
        # Forward fill the first valid heading to the beginning
        first_valid_idx = np.where(valid_motion)[0][0] + 1  # +1 because we're indexing into headings
        if first_valid_idx < len(headings):
            headings[0] = headings[first_valid_idx]
        
        # Forward fill for stationary periods
        last_valid_heading = headings[0]
        for i in range(1, len(headings)):
            if valid_motion[i-1]:  # i-1 because valid_motion is one element shorter
                last_valid_heading = headings[i]
            else:
                headings[i] = last_valid_heading
    else:
        # No significant motion detected, set all headings to 0
        headings[:] = 0.0
    
    return headings



def compute_angular_velocity( headings, times, window_size=100):
    """
    Compute smoothed angular velocity using heading difference over a sliding window.
    
    Args:
        headings: Array of heading angles in radians
        times: Array of timestamps
        window_size: Number of timesteps to use (e.g., 100 = 1s for dt=0.01)
    
    Returns:
        angular_velocities: Array of angular velocities in rad/s
    """
    dt = np.mean(np.diff(times))  # Should be ~0.01s
    unwrapped_headings = np.unwrap(headings)
    
    angular_velocities = np.zeros_like(headings)
    
    for i in range(window_size+1,len(headings)):
        i_start = max(0, i - window_size)
        delta_heading = unwrapped_headings[i] - unwrapped_headings[i_start]
        delta_time = times[i] - times[i_start]
        # delta_time = max(delta_time, 1e-6)  # Avoid division by zero
        angular_velocities[i] = delta_heading / delta_time
    angular_velocities=np.clip(angular_velocities, -np.pi/6, np.pi/6, out=angular_velocities)  # Clip to [-pi, pi]
    
    return angular_velocities



def compute_speeds( utm_coords, times):
    """
    Compute instantaneous speeds from UTM coordinates and timestamps.
    
    Args:
        utm_coords: Array of shape (N, 2) with UTM coordinates
        times: Array of shape (N,) with timestamps
        
    Returns:
        speeds: Array of shape (N,) with speeds in m/s
    """
    # Compute distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(utm_coords, axis=0)**2, axis=1))
    
    # Compute time differences
    time_diffs = np.diff(times)
    
    # Avoid division by zero
    time_diffs = np.maximum(time_diffs, 1e-6)
    
    # Compute speeds
    speeds = distances / time_diffs
    
    # Pad with first speed value to maintain same length as input
    speeds = np.concatenate([[speeds[0]], speeds])
    
    return speeds



def compute_acceleration( speeds, angular_velocities, uniform_times, window_size=100):
    """
    Alternative acceleration computation using local polynomial fitting.
    This is more robust to noise than simple gradient.
    
    Args:
        speeds: Speed array
        angular_velocities: Angular velocity array
        uniform_times: Time array
        window_size: Size of local fitting window
    
    Returns:
        linear_acc, angular_acc: Acceleration arrays
    """
    n = len(speeds)
    linear_acc = np.zeros(n)
    angular_acc = np.zeros(n)
    
    half_window = window_size // 2
    
    for i in range(n):
        # Define window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Extract local data
        local_times = uniform_times[start:end] - uniform_times[i]  # Center around 0
        local_speeds = speeds[start:end]
        local_angular_vels = angular_velocities[start:end]
        
        if len(local_times) >= 3:
            # Fit local polynomial (degree 2 for acceleration)
            try:
                # Linear acceleration
                coeffs_linear = np.polyfit(local_times, local_speeds, min(2, len(local_times)-1))
                if len(coeffs_linear) >= 2:
                    linear_acc[i] = coeffs_linear[-2]  # Coefficient of t^1 term
                
                # Angular acceleration  
                coeffs_angular = np.polyfit(local_times, local_angular_vels, min(2, len(local_times)-1))
                if len(coeffs_angular) >= 2:
                    angular_acc[i] = coeffs_angular[-2]  # Coefficient of t^1 term
                    
            except np.linalg.LinAlgError:
                # Fallback to simple difference if fitting fails
                if i > 0:
                    dt = uniform_times[i] - uniform_times[i-1]
                    linear_acc[i] = (speeds[i] - speeds[i-1]) / dt
                    angular_acc[i] = (angular_velocities[i] - angular_velocities[i-1]) / dt
    
    return linear_acc, angular_acc      



def adaptive_bin_size(
    ang_vel, lin_speed, ang_acc, lin_acc,
    ang_vel_max=0.5, lin_speed_max=20.0,
    ang_acc_max=0.5, lin_acc_max=2.5,
    min_bins=1, max_bins=20,
    weights=(0.5, 0.2, 0.3,0), use_exponential=False):
    """
    Computes adaptive bin size from motion dynamics.
    Uses both acceleration and deceleration (via abs(acc)) to reduce bin size when motion changes.

    Parameters:
    - ang_vel, lin_speed: instantaneous motion
    - ang_acc, lin_acc: rate of change (both positive and negative magnitudes matter)
    """
    w_ang_vel, w_lin_speed, w_ang_acc, w_lin_acc = weights

    # Normalize absolute values of motion quantities
    ang_vel_norm = np.clip(np.abs(ang_vel) / ang_vel_max, 0, 1)
    lin_speed_norm = np.clip(lin_speed / lin_speed_max, 0, 1)
    ang_acc_norm = np.clip(np.abs(ang_acc) / ang_acc_max, 0, 1)
    lin_acc_norm = np.clip(np.abs(lin_acc) / lin_acc_max, 0, 1)

    # Combine motion into a single intensity score
    motion_intensity = (w_ang_vel * ang_vel_norm +
        w_lin_speed * lin_speed_norm +
        w_ang_acc * ang_acc_norm +
        w_lin_acc * lin_acc_norm
    )

    # If nearly stationary, allow longer bins
    if lin_speed < 0.5 and np.abs(ang_vel) < 0.05:
        return max_bins
        

    if use_exponential:
        alpha = np.log(max_bins / min_bins)
        bin_size = max_bins * np.exp(-alpha * motion_intensity)
        bin_size = int(round(bin_size))
    else:
        bin_size = max_bins - motion_intensity * (max_bins - min_bins)
    return max(min_bins, min(bin_size, max_bins))



def plot_vels_accs_bins(bin_mid_times, bin_durations, bin_speeds, bin_angular_velocities, uniform_times, linear_acc, angular_acc, x,y, sequence_name):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    # Plot 1: Bin durations colored by duration, colormap fixed between 0 and 0.5
    sc1 = axes[0, 0].scatter(x, y, c=bin_durations, cmap='plasma', s=1.0)
    axes[0, 0].set_title(f'Adaptive Bin Durations: {sequence_name}')
    axes[0, 0].set_xlabel('X (UTM)')
    axes[0, 0].set_ylabel('Y (UTM)')
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True)
    plt.colorbar(sc1, ax=axes[0, 0], label='Bin Duration (s)')

    # Plot 2: Speeds
    sc2 = axes[0, 1].scatter(x, y, c=bin_speeds, cmap='coolwarm', s=1.0)
    axes[0, 1].set_title('Speed at Bin Locations')
    axes[0, 1].set_xlabel('X (UTM)')
    axes[0, 1].set_ylabel('Y (UTM)')
    axes[0, 1].axis('equal')
    axes[0, 1].grid(True)
    plt.colorbar(sc2, ax=axes[0, 1], label='Speed (m/s)')

    # Plot 3: Angular velocity
    sc3 = axes[1, 0].scatter(x, y, c=bin_angular_velocities, cmap='viridis', s=1.0)
    axes[1, 0].set_title('Angular Velocity at Bin Locations')
    axes[1, 0].set_xlabel('X (UTM)')
    axes[1, 0].set_ylabel('Y (UTM)')
    axes[1, 0].axis('equal')
    axes[1, 0].grid(True)
    plt.colorbar(sc3, ax=axes[1, 0], label='Angular Velocity (rad/s)')

    # Plot 4: Speed vs angular velocity
    sc4 = axes[1, 1].scatter(bin_speeds, bin_angular_velocities, c=bin_durations,
                            cmap='viridis', s=2.0, alpha=0.6)
    axes[1, 1].set_title('Speed vs Angular Velocity (colored by bin duration)')
    axes[1, 1].set_xlabel('Speed (m/s)')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].grid(True)
    plt.colorbar(sc4, ax=axes[1, 1], label='Bin Duration (s)')


    # Interpolate to bin midpoints
    interp_ang_acc = np.interp(bin_mid_times, uniform_times, angular_acc)
    interp_lin_acc = np.interp(bin_mid_times, uniform_times, linear_acc)

    # Plot 5: Angular acceleration
    sc5 = axes[2, 0].scatter(x, y, c=interp_ang_acc, cmap='cool', s=1.0)
    axes[2, 0].set_title('Angular Acceleration at Bin Locations')
    axes[2, 0].set_xlabel('X (UTM)')
    axes[2, 0].set_ylabel('Y (UTM)')
    axes[2, 0].axis('equal')
    axes[2, 0].grid(True)
    plt.colorbar(sc5, ax=axes[2, 0], label='Angular Acceleration (rad/s²)')

    # Plot 6: Linear acceleration
    sc6 = axes[2, 1].scatter(x, y, c=interp_lin_acc, cmap='autumn', s=1.0)
    axes[2, 1].set_title('Linear Acceleration at Bin Locations')
    axes[2, 1].set_xlabel('X (UTM)')
    axes[2, 1].set_ylabel('Y (UTM)')
    axes[2, 1].axis('equal')
    axes[2, 1].grid(True)
    plt.colorbar(sc6, ax=axes[2, 1], label='Linear Acceleration (m/s²)')

    plt.tight_layout()
    plt.savefig(f'./plots/adaptive_binning_{sequence_name}.png', dpi=150)


