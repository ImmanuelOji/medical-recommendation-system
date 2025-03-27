from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
import pandas as pd
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import os
from difflib import get_close_matches  # Add this import for finding similar symptoms

print("Starting the Flask app...")
# flask app
app = Flask(__name__)

# load database datasets===================================
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# load model===========================================
svc = pickle.load(open('svc.pkl','rb'))

# helper function========================================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Add this function to find similar symptoms
def find_similar_symptoms(unknown_symptom, threshold=0.6):
    """Find symptoms in our database that are similar to the unknown symptom."""
    return get_close_matches(unknown_symptom, symptoms_dict.keys(), n=3, cutoff=threshold)

symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21,
    'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
    'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
    'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
    'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

def get_predicted_value(patient_symptoms):
    # Check if all symptoms are valid
    unknown_symptoms = [symptom for symptom in patient_symptoms if symptom not in symptoms_dict]
    if unknown_symptoms:
        return None, unknown_symptoms
    
    # If all symptoms are valid, proceed with prediction
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]], []


@app.context_processor
def utility_processor():
    return {'now': datetime.now}

# ---------------------------------------------------
# Routes
# ---------------------------------------------------


@app.route("/")
def index():
    # Pass the list of valid symptoms to the template
    valid_symptoms = sorted(list(symptoms_dict.keys()))
    return render_template("index.html", valid_symptoms=valid_symptoms)

# Update the home route to handle unknown symptoms
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms == "Symptoms":
            message = "Please either write symptoms or check for any spelling mistakes."
            return render_template('index.html', message=message)
        else:
            # Process input symptoms (assumes comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            
            # Get prediction and check for unknown symptoms
            predicted_disease, unknown_symptoms = get_predicted_value(user_symptoms)
            
            # If there are unknown symptoms, show an error message with suggestions
            if unknown_symptoms:
                suggestions = {}
                for symptom in unknown_symptoms:
                    similar = find_similar_symptoms(symptom)
                    if similar:
                        suggestions[symptom] = similar
                
                # Get a sample of valid symptoms to show as examples
                sample_symptoms = list(symptoms_dict.keys())[:5]
                
                message = f"Sorry, we don't recognize these symptoms: {', '.join(unknown_symptoms)}."
                
                # Add suggestions if available
                if suggestions:
                    message += " Did you mean:"
                    for unknown, similars in suggestions.items():
                        message += f" {unknown} → {', '.join(similars)}?"
                
                message += f" Some valid symptoms include: {', '.join(sample_symptoms)}."
                
                # Pass the list of valid symptoms to the template
                valid_symptoms = sorted(list(symptoms_dict.keys()))
                return render_template('index.html', message=message, valid_symptoms=valid_symptoms)
            
            # If all symptoms are valid, proceed with the prediction
            dis_des, precautions_data, medications, rec_diet, workout = helper(predicted_disease)
            my_precautions = []
            for i in precautions_data[0]:
                my_precautions.append(i)
                
            # Pass the list of valid symptoms to the template
            valid_symptoms = sorted(list(symptoms_dict.keys()))
            return render_template('index.html', predicted_disease=predicted_disease,
                                   dis_des=dis_des, my_precautions=my_precautions,
                                   medications=medications, my_diet=rec_diet,
                                   workout=workout, valid_symptoms=valid_symptoms)
    
    # Pass the list of valid symptoms to the template for GET requests
    valid_symptoms = sorted(list(symptoms_dict.keys()))
    return render_template('index.html', valid_symptoms=valid_symptoms)

@app.route('/download', methods=['POST'])
def download_pdf():
    # Retrieve data from the hidden form fields
    predicted_disease = request.form.get('predicted_disease')
    dis_des = request.form.get('dis_des')
    my_precautions = request.form.getlist('my_precautions')
    medications = request.form.getlist('medications')
    my_diet = request.form.getlist('my_diet')
    workout = request.form.getlist('workout')

    # Create PDF with ReportLab
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    pdf_path = temp_file.name
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c7da0'),
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c7da0'),
        spaceAfter=6
    )
    
    normal_style = styles['Normal']
    
    # Build the PDF content
    elements = []
    
    # Title
    elements.append(Paragraph("Medical Recommendation Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Disease
    elements.append(Paragraph("Diagnosis", heading_style))
    elements.append(Paragraph(predicted_disease, normal_style))
    elements.append(Spacer(1, 12))
    
    # Description
    elements.append(Paragraph("Description", heading_style))
    elements.append(Paragraph(dis_des, normal_style))
    elements.append(Spacer(1, 12))
    
    # Precautions
    elements.append(Paragraph("Recommended Precautions", heading_style))
    precaution_items = [ListItem(Paragraph(p, normal_style)) for p in my_precautions]
    elements.append(ListFlowable(precaution_items, bulletType='bullet'))
    elements.append(Spacer(1, 12))
    
    # Medications
    elements.append(Paragraph("Medications", heading_style))
    medication_items = [ListItem(Paragraph(m, normal_style)) for m in medications]
    elements.append(ListFlowable(medication_items, bulletType='bullet'))
    elements.append(Spacer(1, 12))
    
    # Diet
    elements.append(Paragraph("Dietary Recommendations", heading_style))
    diet_items = [ListItem(Paragraph(d, normal_style)) for d in my_diet]
    elements.append(ListFlowable(diet_items, bulletType='bullet'))
    elements.append(Spacer(1, 12))
    
    # Workout
    elements.append(Paragraph("Exercise Recommendations", heading_style))
    workout_items = [ListItem(Paragraph(w, normal_style)) for w in workout]
    elements.append(ListFlowable(workout_items, bulletType='bullet'))
    elements.append(Spacer(1, 12))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontSize=8,
        textColor=colors.gray
    )
    elements.append(Paragraph("DISCLAIMER: This report is generated based on the symptoms provided and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.", disclaimer_style))
    
    # Build the PDF
    doc.build(elements)
    
    return send_file(pdf_path, as_attachment=True)

# Additional pages
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

