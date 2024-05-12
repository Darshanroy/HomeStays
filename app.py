from flask import Flask, render_template,request
import pickle
import pandas as pd
app = Flask(__name__)



# Define function to load label encoders
def load_label_encoders():
    with open('trained_label_encoder/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return label_encoders


# Define function to load trained models
def load_models():
    models = {}

    with open('trained_models/catboost_model.pkl', 'rb') as f:
        catboost_model = pickle.load(f)
    models['catboost_model'] = catboost_model

    with open('trained_models/lgbm_model.pkl', 'rb') as f:
        lgbm_model = pickle.load(f)
    models['lgbm_model'] = lgbm_model

    with open('trained_models/xgboost_model.pkl', 'rb') as f:
        xgboost_model = pickle.load(f)
    models['xgboost_model'] = xgboost_model

    return models


@app.route("/")
def hello_world():
    """Renders a basic 'Hello, World!' message."""
    return render_template("index.html")  # Assumes a template named index.html


@app.route('/<service_name>')
def redirect(service_name):
    # Here you can perform any necessary operations based on the service name
    # and return the appropriate HTML file
    if service_name == 'Monitoring':
        return render_template('Monitoring_Index.html')
    elif service_name == 'notebooks':
        return render_template('notebooks_index.html')
    elif service_name == 'prediction':
        return render_template('predection.html')
    elif service_name == 'service4':
        return render_template('service4.html')
    else:
        return "Invalid service name"


@app.route('/Monitoring/<service_name>')
def monitoring(service_name):
    # Here you can perform any necessary operations based on the service name
    # and return the appropriate HTML file
    if service_name == 'dataquality':
        return render_template('file.html')
    elif service_name == 'testsuite':
        return render_template('test_suite.html')

    else:
        return "Invalid service name"


@app.route('/notebooks/<service_name>')
def notebooks(service_name):
    # Here you can perform any necessary operations based on the service name
    # and return the appropriate HTML file
    if service_name == 'EDA':
        return render_template('HomeStay_EDA.html')
    elif service_name == 'OHE':
        return render_template('HomeStay_OHE_MT.html')
    elif service_name == 'label-encoding':
        return render_template('Homestay_LabelEncode_Mt.html')
    else:
        return "Invalid service name"


@app.route('/prediction', methods=['POST'])
def process_data():
    if request.method == 'POST':
        # Retrieve form data
        room_type = request.form['room_type']
        accommodates = int(request.form['accommodates'])
        bathrooms = int(request.form['bathrooms'])
        city = request.form['city']
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        zipcode = int(request.form['zipcode'])
        bedrooms = int(request.form['bedrooms'])
        beds = int(request.form['beds'])
        host_tenure = int(request.form['host_tenure'])

        # Load label encoders
        label_encoders = load_label_encoders()

        # Perform label encoding for Room Type and City
        room_type_encoder = label_encoders['room_type']
        room_type_encoded = room_type_encoder.transform([room_type])[0]

        city_encoder = label_encoders['city']
        city_encoded = city_encoder.transform([city])[0]

        # Create DataFrame from form data
        data = {
            'room_type': [room_type_encoded],
            'accommodates': [accommodates],
            'bathrooms': [bathrooms],
            'city': [city_encoded],
            'latitude': [latitude],
            'longitude': [longitude],
            'zipcode': [zipcode],
            'bedrooms': [bedrooms],
            'beds': [beds],
            'host_tenure': [host_tenure]
        }
        df = pd.DataFrame(data)

        # Load trained models
        models = load_models()

        # Make predictions
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(df)

        # Store predicted values in a list
        predicted_values = [predictions[model_name][0] for model_name in models]

        return render_template("result.html", predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
