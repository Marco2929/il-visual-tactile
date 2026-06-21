# --- Input: Senseglove Joint Ranges ---
data_glove_ranges = [
    {
        "thumb": {
            "mcp": {"lower_rad": -0.520, "upper_rad": 0.087},
            "pip": {"lower_rad": 0.139, "upper_rad": 0.698},
            "dip": {"lower_rad": 0.314, "upper_rad": 1.571},
        }
    },
    {
        "index": {
            "mcp": {"lower_rad": 0.000, "upper_rad": 1.222},
            "pip": {"lower_rad": 0.000, "upper_rad": 1.745},
            "dip": {"lower_rad": 0.000, "upper_rad": 1.571},
        }
    },
    {
        "middle": {
            # Fixed! Now starts at 0.0
            "mcp": {"lower_rad": 0.000, "upper_rad": 1.222},
            "pip": {"lower_rad": 0.000, "upper_rad": 1.745},
            "dip": {"lower_rad": 0.000, "upper_rad": 1.571},
        }
    },
    {
        "ring": {
            "mcp": {"lower_rad": 0.000, "upper_rad": 1.222},
            "pip": {"lower_rad": 0.000, "upper_rad": 1.745},
            "dip": {"lower_rad": 0.000, "upper_rad": 1.571},
        }
    },
    {
        "pinky": {
            "mcp": {"lower_rad": 0.000, "upper_rad": 1.222},
            "pip": {"lower_rad": 0.000, "upper_rad": 1.745},
            "dip": {"lower_rad": 0.000, "upper_rad": 1.571},
        }
    }
]

hand_ranges = [
    {'thumb_1': {'lower': 0, 'upper': 1000}},
    {'thumb_2': {'lower': 0, 'upper': 1000}},
    {'index': {'lower': 0, 'upper': 1000}},
    {'middle': {'lower': 0, 'upper': 1000}},
    {'ring': {'lower': 0, 'upper': 1000}},
    {'pinky': {'lower': 0, 'upper': 1000}}
]

hand_init_joints = [
    {'thumb_1': 734},  # rotation
    {'thumb_2': 482},  # flex
    {'index': 911},
    {'middle': 203},
    {'ring': 650},
    {'pinky': 78}
]

hand_ranges_map = {list(item.keys())[0]: list(item.values())[0] for item in hand_ranges}


def normalize_values(value, range_):
    """Normalizes a value from its range to [0, 1]"""
    norm = (value - range_['lower_rad']) / (range_['upper_rad'] - range_['lower_rad'])
    # Clamp the value between 0 and 1
    return max(0.0, min(1.0, norm))


# Function to scale the normalized [0, 1] value to the target robot range ---
def remap_value(normalized_value, target_range):
    """Scales a normalized [0, 1] value to a target range."""
    return normalized_value * (target_range['upper'] - target_range['lower']) + target_range['lower']


def teleoperation_mapping(data_glove_joints):
    hand_mapping = []

    # --- Thumb mapping ---
    data_glove_thumb_joints = data_glove_joints[0]["thumb"]
    data_glove_thumb_ranges = data_glove_ranges[0]["thumb"]

    hand_thumb_1 = "thumb_1"
    hand_thumb_2 = "thumb_2"

    # MCP -> Thumb_1
    mcp_val = data_glove_thumb_joints['mcp']
    mcp_range = data_glove_thumb_ranges['mcp']
    thumb1_norm = normalize_values(mcp_val, mcp_range)
    thumb1_norm = 1.0 - thumb1_norm

    # PIP + DIP -> Thumb_2
    pip_val = data_glove_thumb_joints['pip']
    dip_val = data_glove_thumb_joints['dip']
    pip_range = data_glove_thumb_ranges['pip']
    dip_range = data_glove_thumb_ranges['dip']

    # --- Remap to robot's range ---
    thumb1_scaled = remap_value(thumb1_norm, hand_ranges_map[hand_thumb_1])
    hand_mapping.append({hand_thumb_1: thumb1_scaled})

    mcp_norm_raw = normalize_values(mcp_val, mcp_range)
    pip_norm_raw = normalize_values(pip_val, pip_range)
    dip_norm_raw = normalize_values(dip_val, dip_range)

    # Define influence weights (Must sum to 1.0)
    w_mcp = 0.4
    w_pip = 0.3
    w_dip = 0.3

    # Calculate weighted average
    thumb2_avg_norm = (w_mcp * mcp_norm_raw) + (w_pip * pip_norm_raw) + (w_dip * dip_norm_raw)

    thumb2_norm = 1.0 - thumb2_avg_norm

    # Remap to robot's range ---
    thumb2_scaled = remap_value(thumb2_norm, hand_ranges_map[hand_thumb_2])
    hand_mapping.append({hand_thumb_2: thumb2_scaled})

    # --- Normal fingers mapping ---
    normal_finger_data_glove_joints = data_glove_joints[1:]

    normal_finger_hand_joints = [item.copy() for item in hand_init_joints[2:]]
    normal_finger_data_glove_ranges = data_glove_ranges[1:]

    for i in range(len(normal_finger_data_glove_joints)):

        finger_name = list(normal_finger_hand_joints[i].keys())[0]


        mcp_val, pip_val, dip_val = list(normal_finger_data_glove_joints[i].values())[0].values()

        mcp_val_range, pip_val_range, dip_val_range = list(normal_finger_data_glove_ranges[i].values())[0].values()

        # Just use mcp values cause this resulted in a better performance
        finger_norm = normalize_values(mcp_val, mcp_val_range)

        finger_norm = 1.0 - finger_norm

        finger_scaled = remap_value(finger_norm, hand_ranges_map[finger_name])

        normal_finger_hand_joints[i][finger_name] = finger_scaled

    hand_mapping = hand_mapping + normal_finger_hand_joints

    return hand_mapping


if __name__ == '__main__':
    data_glove_joints_min = [
        {"thumb": {"mcp": -0.174533, "pip": 0.0, "dip": -0.174533,
                   }},
        {"index": {"mcp": -0.523599, "pip": 0.0, "dip": -0.087266,
                   }},
        {"middle": {"mcp": -0.523599, "pip": 0.0, "dip": -0.087266,
                    }},
        {"ring": {"mcp": -0.523599, "pip": 0.0, "dip": -0.087266,
                  }},
        {"pinky": {"mcp": -0.523599, "pip": 0.0, "dip": -0.087266,
                   }}
    ]

    print(f"Minimum values: {teleoperation_mapping(data_glove_joints=data_glove_joints_min)}")

    data_glove_joints_max = [
        {'thumb': {'mcp': 0.610865, 'pip': 0.872665, 'dip': 1.570796}},
        {'index': {'mcp': 1.570796, 'pip': 1.745329, 'dip': 1.570796}},
        {'middle': {'mcp': 1.570796, 'pip': 1.745329, 'dip': 1.570796}},
        {'ring': {'mcp': 1.570796, 'pip': 1.745329, 'dip': 1.570796}},
        {'pinky': {'mcp': 1.570796, 'pip': 1.745329, 'dip': 1.570796}},
    ]

    print(f"Maximum values: {teleoperation_mapping(data_glove_joints=data_glove_joints_max)}")
