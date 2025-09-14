import pandas as pd
import numpy as np
FEATURE_SCORE_MAP = {
    1: 5,   # Swimming Pool
    2: 4,   # Power Backup
    3: 4,   # Club house
    4: 3,   # Parking
    6: 2,   # Park
    9: 4,   # Security Personnel
    11: 1,  # ATM
    12: 4,  # Gymnasium
    21: 3,  # Lift
    25: 2,  # Waste disposal
    29: 3,  # Gas Pipeline
    34: 3,  # Wheelchair Accessibility
    35: 2,  # DG Availability
    39: 1   # Near bank
}
def score_features(feature_str):
    if pd.isna(feature_str):
        return 0
    ids = [int(i.strip()) for i in feature_str.split(',') if i.strip().isdigit()]
    return sum(FEATURE_SCORE_MAP.get(i, 0) for i in ids)

AMENITY_SCORE_MAP = {
    1: 18,   # Swimming Pool
    2: 10,   # Power Back-up
    3: 5,   # Club house / Community Center
    5: 4,   # Feng Shui / Vaastu Compliant
    6: 15,   # Park
    8: 15,   # Private Garden / Terrace
    9: 13,   # Security Personnel
    10: 17,  # Centrally Air Conditioned
    11: 5,  # ATM
    12: 10,  # Fitness Centre / GYM
    13: 15,  # Cafeteria / Food Court
    15: 15,  # Bar / Lounge
    16: 10,  # Conference room
    17: 17,  # Security / Fire Alarm
    19: 5,  # Visitor Parking
    20: 5,  # Intercom Facility
    21: 4,  # Lift(s)
    22: 5,  # Service / Goods Lift
    23: 5,  # Maintenance Staff
    24: 10,  # Water Storage
    25: 15,  # Waste Disposal
    26: 5,  # Rain Water Harvesting
    27: 17,  # Access to High Speed Internet
    28: 10,  # Bank Attached Property
    29: 10,  # Piped-gas
    32: 10,  # Water purifier
    33: 15,  # Shopping Centre
    34: 5,  # WheelChair Accessibilitiy
    35: 5,  # DG Availability
    36: 15,  # CCTV Surveillance
    37: 15,  # Grade A Building
    38: 15,  # Grocery Shop
    39: 5   # Near Bank
}

def score_amenities(amenity_str):
    if pd.isna(amenity_str):
        return 0
    ids = [int(i.strip()) for i in amenity_str.split(',') if i.strip().isdigit()]
    return sum(AMENITY_SCORE_MAP.get(i, 0) for i in ids)


# Facing mapping dictionary (case-insensitive, flexible input)

def preprocess_input(raw_input,furnish_categories,facing_categories,property_type_categories,locality_categories):
    
    FURNISH_MAP = {
        "furnished": 1,
        "unfurnished": 2,
        "semi-furnished": 4,
        "semifurnished": 4   # allow both with/without hyphen
    }
    FACING_MAP = {
        "north": 1,
        "south": 2,
        "east": 3,
        "west": 4,
        "north-east": 5,
        "northeast": 5,
        "ne": 5,
        "north-west": 6,
        "northwest": 6,
        "nw": 6,
        "south-east": 7,
        "southeast": 7,
        "se": 7,
        "south-west": 8,
        "southwest": 8,
        "sw": 8
    }
    input_df = pd.DataFrame([raw_input])

    if 'FURNISH' in input_df.columns:
        furnish_val = str(input_df.at[0, 'FURNISH']).lower().strip()
        if furnish_val in FURNISH_MAP:
            input_df.at[0, 'FURNISH'] = FURNISH_MAP[furnish_val]
        else:
            raise ValueError(f"❌ Unknown FURNISH value: {furnish_val}. Expected one of {list(FURNISH_MAP.keys())}")
        
    if 'FACING' in input_df.columns:
        facing_val = str(input_df.at[0, 'FACING']).lower().strip()
        if facing_val in FACING_MAP:
            input_df.at[0, 'FACING'] = FACING_MAP[facing_val]
        else:
            raise ValueError(f"❌ Unknown FACING value: {facing_val}. Expected one of {list(FACING_MAP.keys())}")

    # Apply same categories as training
    input_df['FURNISH'] = input_df['FURNISH'].astype(pd.CategoricalDtype(categories=furnish_categories))
    input_df['FACING'] = input_df['FACING'].astype(pd.CategoricalDtype(categories=facing_categories))
    input_df['PROPERTY_TYPE'] = input_df['PROPERTY_TYPE'].astype(pd.CategoricalDtype(categories=property_type_categories))
    input_df['LOCALITY'] = input_df['LOCALITY'].astype(pd.CategoricalDtype(categories=locality_categories))

    # Clean AREA
    if 'AREA' in input_df:
        input_df['AREA'] = input_df['AREA'].astype(str).str.replace('sq.ft.', '', regex=False).str.strip()
        input_df['AREA'] = pd.to_numeric(input_df['AREA'], errors='coerce')

    # Encode categorical using same mapping
    input_df['FURNISH_CODE'] = input_df['FURNISH'].cat.codes
    input_df['FACING_CODE'] = input_df['FACING'].cat.codes
    input_df['PROPERTY_TYPE_CODE'] = input_df['PROPERTY_TYPE'].cat.codes
    input_df['LOCALITY_CODE'] = input_df['LOCALITY'].cat.codes

    # Handle TRANSACT_TYPE
    if 'TRANSACT_TYPE' in input_df:
        input_df['TRANSACT_TYPE'] = input_df['TRANSACT_TYPE'].fillna(0).astype(int)

    # Handle FLOOR_NUM (replace 'G' with 0, NaNs → -1)
    if 'FLOOR_NUM' in input_df:
        input_df['FLOOR_NUM'] = input_df['FLOOR_NUM'].replace('G', 0)
        input_df['FLOOR_NUM'] = pd.to_numeric(input_df['FLOOR_NUM'], errors='coerce').fillna(-1).astype(int)

    # Feature Score
    def score_features(feature_str):
        if pd.isna(feature_str):
            return 0
        ids = [int(i.strip()) for i in str(feature_str).split(',') if i.strip().isdigit()]
        return sum(FEATURE_SCORE_MAP.get(i, 0) for i in ids)

    if 'FEATURES' in input_df:
        input_df['FEATURE_SCORE'] = input_df['FEATURES'].apply(score_features)

    # Amenity Score
    def score_amenities(amenity_str):
        if pd.isna(amenity_str):
            return 0
        ids = [int(i.strip()) for i in str(amenity_str).split(',') if i.strip().isdigit()]
        return sum(AMENITY_SCORE_MAP.get(i, 0) for i in ids)

    if 'AMENITIES' in input_df:
        input_df['AMENITY_SCORE'] = input_df['AMENITIES'].apply(score_amenities)

    # Debug print
    print("\n🔎 Input Encoding Verification:")
    print("PROPERTY_TYPE:", input_df['PROPERTY_TYPE'].iloc[0], "->", input_df['PROPERTY_TYPE_CODE'].iloc[0])
    print("LOCALITY:", input_df['LOCALITY'].iloc[0], "->", input_df['LOCALITY_CODE'].iloc[0])
    print("FURNISH:", input_df['FURNISH'].iloc[0], "->", input_df['FURNISH_CODE'].iloc[0])
    print("FACING:", input_df['FACING'].iloc[0], "->", input_df['FACING_CODE'].iloc[0])

    return input_df

def recommend_properties(user_input, model, df, ml_features,
                         furnish_categories, facing_categories,
                         property_type_categories, locality_categories,
                         top_n=10):
    """
    Recommend similar properties based on predicted price and feature similarity with custom scoring.
    """
    FURNISH_MAP = {
        "furnished": 1,
        "unfurnished": 2,
        "semi-furnished": 4,
        "semifurnished": 4   # allow both with/without hyphen
    }

    # --- Preprocess user input like training ---
    user_df = preprocess_input(user_input, furnish_categories,
                               facing_categories, property_type_categories,
                               locality_categories)
    user_df = user_df[ml_features]

    # Predict price
    predicted_price = model.predict(user_df)[0]
    lower = predicted_price * 0.9
    upper = predicted_price * 1.1

    # Filter dataset by price range
    candidate_df = df[(df['TARGET_PRICE'] >= lower) & (df['TARGET_PRICE'] <= upper)].copy()
    if candidate_df.empty:
        return pd.DataFrame()

    # --- Use encoded values from user_df ---
    user_vals = user_df.iloc[0]  # row as Series

    # Feature weights
    weights = {
        'BEDROOM_NUM': 2,
        'BATHROOM_NUM': 1.5,
        'BALCONY_NUM': 1,
        'AREA': 2.5,
        'FURNISH': 1,  # already encoded in preprocess
        'PROPERTY_TYPE_CODE': 1,
        'LOCALITY_CODE': 1,
        'TRANSACT_TYPE': 0.5,
        'OWNTYPE': 0.5
    }
    total_weight = sum(weights.values())

    def compute_match_score(row):
        score = 0
        for col, weight in weights.items():
            user_val = user_vals[col]
            row_val = row[col]

            if col == 'AREA':
                rel_diff = abs(user_val - row_val) / (abs(user_val) + 1e-5)
                score += weight * (1 - min(rel_diff / 0.15, 1))
            else:
                score += weight * int(user_val == row_val)
        return (score / total_weight) * 100

    candidate_df['MATCH_SCORE'] = candidate_df.apply(compute_match_score, axis=1)

    # --- Decode FURNISH numeric code back to string ---
    INVERSE_FURNISH_MAP = {v: k for k, v in FURNISH_MAP.items()}
    # Ensure we pick the first key if multiple map to same value
    candidate_df['FURNISH'] = candidate_df['FURNISH'].apply(lambda x: INVERSE_FURNISH_MAP.get(x, str(x)))

    # Return top matches
    return candidate_df.sort_values(by='MATCH_SCORE', ascending=False).head(top_n)[
        ['TARGET_PRICE', 'BEDROOM_NUM', 'AREA', 'FURNISH', 'MATCH_SCORE']
    ]

