import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error
import joblib
from scipy.spatial import cKDTree
import time
from datetime import datetime, timedelta
import os


class ChargingStationPredictor:
    def __init__(self, data):
        """Initialize with charging station dataset"""
        self.data = pd.DataFrame(data)
        self.scalers = {}
        self.label_encoders = {}
        self.models = {}
        self.feature_columns = [
            'EV_Level1_EVSE_Num', 'EV_Level2_EVSE_Num', 'EV_DC_Fast_Count',
            'EV_Network', 'Latitude', 'Longitude', 'Open_Date',
            'EV_Connector_Types', 'Country', 'Access_Code', 'Facility_Type'
        ]

    def preprocess_data(self):
        """Prepare features for all models"""
        print("Starting data preprocessing...")

        # Handle missing values
        numeric_cols = ['EV_Level1_EVSE_Num', 'EV_Level2_EVSE_Num', 'EV_DC_Fast_Count']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)

        # Calculate total ports
        self.data['Total_Ports'] = (self.data['EV_Level1_EVSE_Num'] +
                                    self.data['EV_Level2_EVSE_Num'] +
                                    self.data['EV_DC_Fast_Count'])

        # Process Open_Date
        self.data['Open_Date'] = pd.to_datetime(self.data['Open_Date'], errors='coerce')
        self.data['Days_Since_Opening'] = (datetime.now() - self.data['Open_Date']).dt.days
        self.data['Days_Since_Opening'] = self.data['Days_Since_Opening'].fillna(
            self.data['Days_Since_Opening'].median())

        # Print unique values for categorical columns
        categorical_cols = ['EV_Network', 'EV_Connector_Types', 'Country',
                            'Access_Code', 'Facility_Type']
        print("\nUnique values in categorical columns:")
        for col in categorical_cols:
            unique_vals = self.data[col].fillna('UNKNOWN').unique()
            print(f"\n{col}: {sorted(unique_vals)}")

        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[f'{col}_Encoded'] = le.fit_transform(self.data[col].fillna('UNKNOWN'))
            self.label_encoders[col] = le

        print("\nData preprocessing completed!")

    def calculate_station_density(self, lat, lon, radius_miles=3):
        """Calculate charging station density for a single point"""
        # Convert to radians
        lat_rad, lon_rad = np.radians([lat, lon])
        coords_rad = np.radians(self.data[['Latitude', 'Longitude']].values)

        # Create KD-tree
        tree = cKDTree(coords_rad)

        # Convert radius to radians (Earth's radius ≈ 3959 miles)
        radius_rad = radius_miles / 3959.0

        # Find points within radius
        indices = tree.query_ball_point([lat_rad, lon_rad], radius_rad)

        # Calculate densities
        area = np.pi * (radius_miles ** 2)
        station_density = len(indices) / area
        port_density = self.data['Total_Ports'].iloc[indices].sum() / area

        return station_density, port_density

    def prepare_features(self):
        """Prepare features for model training"""
        print("Preparing features...")
        print("This may take several minutes, please wait...")
        start_time = time.time()

        # Calculate densities for all points
        densities = []
        total_points = len(self.data)

        # Process in batches to show progress
        batch_size = 1000
        for i in range(0, total_points, batch_size):
            batch_start = time.time()
            end_idx = min(i + batch_size, total_points)

            # Process current batch
            batch_densities = [
                self.calculate_station_density(lat, lon)
                for lat, lon in self.data[['Latitude', 'Longitude']].iloc[i:end_idx].values
            ]
            densities.extend(batch_densities)

            # Calculate and display progress
            processed = end_idx
            percent_complete = (processed / total_points) * 100
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total_points - processed) / rate if rate > 0 else 0

            print(f"\rProgress: {processed}/{total_points} ({percent_complete:.1f}%) | "
                  f"Speed: {rate:.1f} locations/sec | "
                  f"Est. remaining: {str(timedelta(seconds=int(remaining)))}", end='')

        print("\nFeature preparation completed!")
        print(f"Total time: {str(timedelta(seconds=int(time.time() - start_time)))}")

        # Create feature DataFrame
        print("Building feature matrix...")
        features = pd.DataFrame({
            'station_density': [d[0] for d in densities],
            'port_density': [d[1] for d in densities],
            'Days_Since_Opening': self.data['Days_Since_Opening'],
            'Latitude': self.data['Latitude'],
            'Longitude': self.data['Longitude'],
            'EV_Level1': self.data['EV_Level1_EVSE_Num'],
            'EV_Level2': self.data['EV_Level2_EVSE_Num'],
            'DC_Fast': self.data['EV_DC_Fast_Count'],
            'total_ports': self.data['Total_Ports']
        })

        # Add encoded categorical features
        for col in self.label_encoders.keys():
            features[f'{col}_Encoded'] = self.data[f'{col}_Encoded']

        print("Feature matrix completed!")
        return features

    def train_all_models(self):
        """Train all prediction models"""
        print("\nStarting to train all models...")

        # Prepare features
        features = self.prepare_features()

        # Train density prediction model (Need new station?)
        print("\nTraining density prediction model...")
        X = features.copy()
        median_density = features['station_density'].median()
        y = (features['station_density'] < median_density).astype(int)

        print("Splitting data for density model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training Random Forest classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        self.scalers['density'] = scaler
        self.models['density'] = model

        density_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"Density prediction model accuracy: {density_accuracy:.2f}")

        # Train charger type prediction model
        print("\nTraining charger type prediction model...")
        print("Preparing charger type labels...")
        conditions = [
            self.data['EV_DC_Fast_Count'] > 0,
            self.data['EV_Level2_EVSE_Num'] > 0,
            self.data['EV_Level1_EVSE_Num'] > 0
        ]
        choices = ['DC Fast', 'Level 2', 'Level 1']
        charger_type = np.select(conditions, choices, default='Level 2')

        X = features.copy()
        y = charger_type

        print("Splitting data for charger type model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training Random Forest classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        self.scalers['charger_type'] = scaler
        self.models['charger_type'] = model

        charger_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"Charger type prediction model accuracy: {charger_accuracy:.2f}")

        # Train port number prediction model
        print("\nTraining port number prediction model...")
        X = features.copy()
        y = features['total_ports']

        print("Splitting data for port number model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training Random Forest regressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        self.scalers['ports'] = scaler
        self.models['ports'] = model

        ports_rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))
        print(f"Port number prediction model RMSE: {ports_rmse:.2f}")

        return {
            'density_accuracy': density_accuracy,
            'charger_accuracy': charger_accuracy,
            'ports_rmse': ports_rmse
        }

    def predict(self, latitude, longitude, access_type='public', facility_type='PARKING_LOT',
                ev_network='Non-Networked', connector_types='J1772', country='US'):
        """Make predictions for a new location"""
        print(f"\nMaking predictions for location ({latitude}, {longitude})...")

        # Calculate density features
        station_density, port_density = self.calculate_station_density(latitude, longitude)

        # Prepare features for prediction
        features = {
            'station_density': station_density,
            'port_density': port_density,
            'Days_Since_Opening': 0,  # New station
            'Latitude': latitude,
            'Longitude': longitude,
            'EV_Level1': 0,
            'EV_Level2': 0,
            'DC_Fast': 0,
            'total_ports': 0
        }

        # Map input values to encoded values
        value_mapping = {
            'EV_Network': ev_network,
            'EV_Connector_Types': connector_types,
            'Country': country,
            'Access_Code': access_type,
            'Facility_Type': facility_type
        }

        # Encode categorical features with error handling
        for col, encoder in self.label_encoders.items():
            value = value_mapping[col]
            try:
                features[f'{col}_Encoded'] = encoder.transform([value])[0]
            except ValueError as e:
                print(f"\nWarning: Unknown {col} value: {value}")
                print(f"Available values are: {sorted(encoder.classes_)}")
                print(f"Using most common value instead.")
                # Use the most common value from training data
                common_value = encoder.inverse_transform([0])[0]
                features[f'{col}_Encoded'] = encoder.transform([common_value])[0]

        # Create feature array
        X = pd.DataFrame([features])

        # Make predictions
        # 1. Need new station prediction
        X_scaled = self.scalers['density'].transform(X)
        needs_station = bool(self.models['density'].predict(X_scaled)[0])

        # 2. Charger type prediction
        X_scaled = self.scalers['charger_type'].transform(X)
        best_charger = self.models['charger_type'].predict(X_scaled)[0]

        # 3. Number of ports prediction
        X_scaled = self.scalers['ports'].transform(X)
        recommended_ports = max(1, int(round(self.models['ports'].predict(X_scaled)[0])))
        
        predictions = {
            'needs_station': needs_station,
            'best_charger_type': best_charger,
            'recommended_ports': recommended_ports,
            'station_density': station_density,
            'port_density': port_density
        }

        results=f"\nPrediction Results:Need new charging station: {'Yes' if needs_station else 'No'}\nRecommended charger type: {best_charger}\nRecommended number of ports: {recommended_ports}\nCurrent area station density: {station_density:.2f} stations/mi²\nCurrent area port density: {port_density:.2f} ports/mi²"

        print("\nPrediction Results:")
        print(f"Need new charging station: {'Yes' if needs_station else 'No'}")
        print(f"Recommended charger type: {best_charger}")
        print(f"Recommended number of ports: {recommended_ports}")
        print(f"Current area station density: {station_density:.2f} stations/mi²")
        print(f"Current area port density: {port_density:.2f} ports/mi²")
        
        return results
    def save_models(self, path='charging_station_models'):
        """Save all trained models and preprocessors"""
        print(f"\nSaving models to {path}.joblib ...")
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, f'{path}.joblib')
        print("Models saved successfully!")

    def load_models(self, path='charging_station_models'):
        """Load trained models and preprocessors"""
        print(f"Loading models from {path}.joblib ...")
        model_data = joblib.load(f'{path}.joblib')
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.label_encoders = model_data['label_encoders']
        print("Models loaded successfully!")
def main():
    # Read data
    print("Reading data...")
    data_path = 'alt_fuel_stations.csv'
    data = pd.read_csv(data_path, low_memory=False)
    print(f"Read {len(data)} charging station records")

    # Create predictor
    predictor = ChargingStationPredictor(data)

    # Preprocess data
    predictor.preprocess_data()

    predictor.load_models("charging_station_models")

    # Save models
    # predictor.save_models()

    # Test prediction
    print("\nTesting prediction functionality...")
    # Using Los Angeles downtown coordinates as a test
    test_prediction = predictor.predict(
        latitude=34.0522,
        longitude=-118.2437,

    )
    print(predictor.models)


    # Test prediction
    print("\nTesting prediction functionality...")
    # Using Los Angeles downtown coordinates as a test
    test_prediction = predictor.predict(
        latitude=34.0522,
        longitude=-118.2437,

    )
    print(predictor.models)
if __name__ == "__main__":
    main()