from flask import Flask, render_template_string, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained Naive Bayes model[cite: 1]
model_path = os.path.join(os.path.dirname(__file__), 'naive.pkl')
with open(model_path, 'rb') as f:
  model = pickle.load(f)

# Features expected by the model: 
# ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'][cite: 1]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f3f4f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { border: none; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); background: #ffffff; }
        .btn-primary { background-color: #6366f1; border: none; border-radius: 8px; padding: 12px; font-weight: 600; }
        .btn-primary:hover { background-color: #4f46e5; }
        .form-control, .form-select { border-radius: 8px; padding: 10px; border: 1px solid #d1d5db; }
        .form-control:focus, .form-select:focus { border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15); }
        .result-box { background: #eef2ff; color: #4338ca; border-radius: 8px; padding: 15px; font-weight: 600; text-align: center; margin-top: 20px; border: 1px solid #c7d2fe; }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-md-8 col-lg-6">
                <div class="card p-4 p-md-5">
                    <h3 class="text-center mb-4 text-dark fw-bold">Employee Status Predictor</h3>
                    <form method="POST">
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">Education</label>
                            <select name="Education" class="form-select" required>
                                <option value="0">Bachelors</option>
                                <option value="1">Masters</option>
                                <option value="2">PHd</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">Joining Year</label>
                            <input type="number" name="JoiningYear" class="form-control" placeholder="e.g. 2018" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">City</label>
                            <select name="City" class="form-select" required>
                                <option value="0">Bangalore</option>
                                <option value="1">Pune</option>
                                <option value="2">New Delhi</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">Payment Tier</label>
                            <select name="PaymentTier" class="form-select" required>
                                <option value="1">Tier 1</option>
                                <option value="2">Tier 2</option>
                                <option value="3">Tier 3</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">Age</label>
                            <input type="number" name="Age" class="form-control" placeholder="e.g. 28" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">Gender</label>
                            <select name="Gender" class="form-select" required>
                                <option value="0">Male</option>
                                <option value="1">Female</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">Ever Benched</label>
                            <select name="EverBenched" class="form-select" required>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label text-muted small fw-bold">Experience In Current Domain</label>
                            <input type="number" name="ExperienceInCurrentDomain" class="form-control" placeholder="e.g. 3" required>
                        </div>
                        <button type="submit" class="btn btn-primary w-100 mt-2">Run Prediction</button>
                    </form>
                    {% if prediction is not none %}
                    <div class="result-box">
                        Prediction Result: {{ prediction }}
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['Education']),
                float(request.form['JoiningYear']),
                float(request.form['City']),
                float(request.form['PaymentTier']),
                float(request.form['Age']),
                float(request.form['Gender']),
                float(request.form['EverBenched']),
                float(request.form['ExperienceInCurrentDomain'])
            ]
            final_features = np.array([features])
            pred = model.predict(final_features)
            prediction = pred[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
