from django.shortcuts import render
import pandas as pd
import joblib
import os
import json

# Load category mappings
mapping_path = os.path.join(os.path.dirname(__file__), "category_mappings.json")
with open(mapping_path, "r") as f:
    category_mappings = json.load(f)

furnish_categories = category_mappings["furnish"]
facing_categories = category_mappings["facing"]
property_type_categories = category_mappings["property_type"]
locality_categories = category_mappings["locality"]

data_path = os.path.join(os.path.dirname(__file__), "processed_data.csv")
df = pd.read_csv(data_path)

# Load model (make sure house_price_model.pkl is inside predictor/)
model_path = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
model = joblib.load(model_path)

FEATURE_ID_MAP = { "Swimming Pool": 1,"Power Backup": 2,"Club house": 3,"Parking": 4,"Park": 6,"Security Personnel": 9,"ATM": 11,"Gymnasium": 12,"Lift": 21,
    "Waste disposal": 25,"Gas Pipeline": 29,"Wheelchair Accessibility": 34,"DG Availability": 35,"Near bank": 39
}

AMENITY_ID_MAP = { "Swimming Pool": 1,"Power Back-up": 2,"Club house / Community Center": 3,"Feng Shui / Vaastu Compliant": 5,"Park": 6,"Private Garden / Terrace": 8,
    "Security Personnel": 9,"Centrally Air Conditioned": 10,"ATM": 11,"Fitness Centre / GYM": 12,"Cafeteria / Food Court": 13,"Bar / Lounge": 15,"Conference room": 16,
    "Security / Fire Alarm": 17,"Visitor Parking": 19,"Intercom Facility": 20,"Lift(s)": 21,"Service / Goods Lift": 22,"Maintenance Staff": 23,"Water Storage": 24,
    "Waste Disposal": 25,"Rain Water Harvesting": 26,"Access to High Speed Internet": 27,"Bank Attached Property": 28,"Piped-gas": 29,"Water purifier": 32,
    "Shopping Centre": 33,"WheelChair Accessibilitiy": 34,"DG Availability": 35,"CCTV Surveillance": 36,"Grade A Building": 37,"Grocery Shop": 38,"Near Bank": 39
}


from Real_Estate_Price import preprocess_input,recommend_properties  

def index(request):
    predicted_price = None
    recommendations = None 

    if request.method == "POST":
        selected_features = request.POST.getlist("FEATURES")  
        selected_amenities = request.POST.getlist("AMENITIES") 
        raw_input = {
            'PROPERTY_TYPE': request.POST.get("PROPERTY_TYPE"),
            'LOCALITY': request.POST.get("LOCALITY"),
            'TRANSACT_TYPE': int(request.POST.get("TRANSACT_TYPE")),
            'OWNTYPE': int(request.POST.get("OWNTYPE")),
            'AGE': int(request.POST.get("AGE")),
            'BEDROOM_NUM': int(request.POST.get("BEDROOM_NUM")),
            'BATHROOM_NUM': int(request.POST.get("BATHROOM_NUM")),
            'BALCONY_NUM': int(request.POST.get("BALCONY_NUM")),
            'AREA': request.POST.get("AREA"),
            'FURNISH': request.POST.get("FURNISH"),
            'FACING': request.POST.get("FACING"),  
            'FLOOR_NUM': int(request.POST.get("FLOOR_NUM")),
            'TOTAL_FLOOR': int(request.POST.get("TOTAL_FLOOR")),
            'TOTAL_LANDMARK_COUNT': int(request.POST.get("TOTAL_LANDMARK_COUNT")),
            'QUALITY_SCORE': float(request.POST.get("QUALITY_SCORE")),
            'CITY_ID': int(request.POST.get("CITY_ID")),
            "FEATURES": ",".join([str(FEATURE_ID_MAP[name]) for name in selected_features if name in FEATURE_ID_MAP]),
            "AMENITIES": ",".join([str(AMENITY_ID_MAP[name]) for name in selected_amenities if name in AMENITY_ID_MAP]),
        }
        ml_features = [
            'PROPERTY_TYPE_CODE', 'LOCALITY_CODE', 'TRANSACT_TYPE', 'OWNTYPE', 'AGE',
            'BEDROOM_NUM', 'BATHROOM_NUM', 'BALCONY_NUM', 'AREA', 'FURNISH',
            'FLOOR_NUM', 'TOTAL_FLOOR', 'TOTAL_LANDMARK_COUNT',
            'QUALITY_SCORE', 'CITY_ID', 'FEATURE_SCORE', 'AMENITY_SCORE'
        ]


        # preprocess input → DataFrame
        input_df = preprocess_input(raw_input,furnish_categories,facing_categories,property_type_categories,locality_categories)
        input_df = input_df[ml_features]  
        # predict
        predicted_price = model.predict(input_df)[0]
        # Get recommendations
        recs_df = recommend_properties(raw_input, model, df, ml_features,furnish_categories,facing_categories,property_type_categories,locality_categories, top_n=10)
        recommendations = recs_df.to_dict(orient="records")  

    return render(request, "index.html", {
        "predicted_price": predicted_price,
        "furnish_categories": furnish_categories,
        "facing_categories": facing_categories,
        "property_type_categories": property_type_categories,
        "locality_categories": locality_categories,
        "FEATURE_ID_MAP":FEATURE_ID_MAP,
        "AMENITY_ID_MAP":AMENITY_ID_MAP,
        "recommendations": recommendations,
    })
