CATEGORICAL_FEATURES = [
    "Target",
    "Workclass",
    "Education",
    "Martial_Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Country"
]

NUMERIC_FEATURES = [
    "Age",
    "fnlwgt",
    "Education_Num",
    "Capital_Gain",
    "Capital_Loss",
    "Hours_per_week"
]

ALL_FEATURES = [
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES
]

ALL_FEATURES_COLLECTION = {
    1: "Age",
    2: "fnlwgt",
    3: "Education_Num",
    4: "Capital_Gain",
    5: "Capital_Loss",
    6: "Hours_per_week",
    7: "Target",
    8: "Workclass",
    9: "Education",
    10: "Martial_Status",
    11: "Occupation",
    12: "Relationship",
    13: "Race",
    14: "Sex",
    15: "Country"
}
