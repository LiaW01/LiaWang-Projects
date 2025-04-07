import argparse
import pandas as pd
from xgboost import XGBClassifier
from prediction import predictFunc

# python3 harness.py --input_csv train_demo_subset.csv --output_csv final_output.csv

def harness(input_csv, output_csv):
    # Load the model
    model = XGBClassifier()
    model.load_model('model_xgb.json')  # Load pre-trained model

    # Load new data
    new_data = pd.read_csv(input_csv)

    # Perform predictions using the external prediction function
    predictions = predictFunc(model, new_data)

    # Save predictions to output file
    pd.Series(predictions).to_csv(output_csv, index=False, header=False)

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run harness function for default probability prediction.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")

    args = parser.parse_args()
    harness(args.input_csv, args.output_csv)
