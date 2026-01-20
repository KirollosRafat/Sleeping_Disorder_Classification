import numpy as np

def add_bp_category(df, bp_col='Blood Pressure', output_col='BP_Category'):
    # Split BP into Systolic and Diastolic
    df[['Systolic', 'Diastolic']] = df[bp_col].str.split('/', expand=True).astype(int)
    
    # Vectorized conditions
    conditions = [
        (df['Systolic'] < 120) & (df['Diastolic'] < 80),
        (df['Systolic'] < 130) & (df['Diastolic'] < 80),
        (df['Systolic'] < 140) | (df['Diastolic'] < 90)
    ]
    
    choices = ['Normal', 'Elevated', 'Hypertension Stage 1']
    
    # Apply categories
    df[output_col] = np.select(conditions, choices, default='Hypertension Stage 2')
    
    return df

def assign_age_group(age):
    if age <= 30:
        return 'Young'
    elif 31 <= age <= 40:
        return 'Adult'
    elif 41 <= age <= 50:
        return 'Middle-Aged'
    else:
        return 'Senior'
