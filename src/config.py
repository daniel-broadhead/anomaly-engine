"""
Domain-specific column configs.

This is the only file you should need to touch when adding a new domain later:
just add a new key with 'numeric' and 'categorical' column lists.
"""

CONFIGS = {
    'healthcare': {
        'numeric': [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses'
        ],
        'categorical': ['race', 'gender', 'age', 'admission_type_id']
    },
    'financial': {
        'numeric': [f'V{i}' for i in range(1, 29)] + ['Amount'],
        'categorical': []
    },
    'realestate': {
        'numeric': [
            'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
            'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars',
            'SalePrice'
        ],
        'categorical': ['Neighborhood', 'BldgType', 'HouseStyle']
    }
}

# Contamination (estimated anomaly fraction) per domain.
# Financial has a known ground-truth rate (~0.17% fraud); others are
# exploratory defaults you should tune after inspecting flagged rows.
CONTAMINATION = {
    'healthcare': 0.05,
    'financial': 0.002,
    'realestate': 0.05,
}
