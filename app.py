from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model pipeline
try:
    model = joblib.load('ultimate_champion_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Pre-populate dropdowns with realistic options ---
# In a real app, you might load these from a file or the original dataset.
# For simplicity, we define them here. These should match the values your model was trained on.
qualifications = ['PhD', 'BBA', 'BA', 'BCA', 'BS', 'BE', 'B.Tech', 'B.Com', 'M.Tech', 'MBA']
locations = ['New Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata']
work_types = ['Full-Time', 'Contract', 'Intern', 'Part-Time']
job_titles = ['Interaction Designer', 'UX Designer', 'UI/UX Designer', 'Software Engineer', 'Data Scientist', 'Project Manager', 'Business Analyst', 'Graphic Designer']
sectors = ['Information Technology', 'Finance', 'Aerospace & Defense', 'Healthcare', 'Automotive', 'Professional Services']
industries = ['Computer Software', 'Internet', 'Financial Services', 'Hospital & Health Care', 'Automotive', 'Information Technology and Services']


@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    
    if request.method == 'POST':
        try:
            # --- Extract form data ---
            min_experience = float(request.form.get('min_experience'))
            max_experience = float(request.form.get('max_experience'))
            company_size = int(request.form.get('company_size'))
            qualification = request.form.get('qualification')
            location = request.form.get('location')
            work_type = request.form.get('work_type')
            job_title = request.form.get('job_title')
            sector = request.form.get('sector')
            industry = request.form.get('industry')
            skills = request.form.get('skills')
            
            # --- Create a DataFrame with the correct column names ---
            # This is the most critical step. The column names and order
            # MUST EXACTLY MATCH the training data's columns.
            # We provide default values for columns not in the form.
            
            input_data = pd.DataFrame({
                'Qualifications': [qualification],
                'location': [location],
                'Country': ['India'], # Default value
                'Work Type': [work_type],
                'Preference': ['Any'], # Default value
                'Job Title': [job_title],
                'Role': ['Engineer'], # Default value, Role can be simplified or linked to Job Title
                'Job Portal': ['LinkedIn'], # Default value
                'Sector': [sector],
                'Industry': [industry],
                'Salary_Spread': [50000], # Assume a default spread, as it's hard for users to input
                'Min_Experience': [min_experience],
                'Max_Experience': [max_experience],
                'Company Size': [company_size],
                'Posting_Year': [2023], # Default to a recent year
                'Posting_Month': [10], # Default to a recent month
                'skills': [skills],
                'Benefits': ['Health Insurance, PTO'], # Default value
                'Full_Job_Text': [skills] # Use skills as a proxy for the full job text in this simple UI
            })

            # --- Make Prediction ---
            if model:
                prediction = model.predict(input_data)[0]
                prediction_text = f"${prediction:,.2f}" # Format as currency
            else:
                prediction_text = "Error: Model not loaded."

        except Exception as e:
            prediction_text = f"An error occurred: {e}"

    return render_template('index.html', 
                           prediction_text=prediction_text,
                           qualifications=qualifications,
                           locations=locations,
                           work_types=work_types,
                           job_titles=job_titles,
                           sectors=sectors,
                           industries=industries)

if __name__ == '__main__':
    app.run(debug=True)